from torch_util.hypers_base import HypersBase
from torch_util.transformer_optimize import TransformerOptimize
from transformers import RagTokenizer, RagSequenceForGeneration, RagTokenForGeneration
import torch
import os
from util.line_corpus import read_lines, block_shuffle
import ujson as json
import random
from slot_filling.rag_util import prepare_seq2seq_batch
from slot_filling.corpus_client import CorpusClient
from slot_filling.dataset_stats import get_relations_by_fold, get_relation_from_inst
import logging
import signal

logger = logging.getLogger(__name__)


class RagHypers(HypersBase):
    def __init__(self):
        super().__init__()
        self.kilt_data = ''
        self.model_name = 'facebook/rag-token-nq'
        self.model_path = ''
        self.no_leading_space = False
        self.corpus_endpoint = ''
        self.n_docs = 5
        self.fold = ''  # {1-n}of{n}
        # only used for train
        self.num_instances = -1
        self.warmup_fraction = 0.1
        self.__required_args__ = ['kilt_data', 'num_instances', 'output_dir', 'corpus_endpoint']

    def _post_init(self):
        super()._post_init()
        # launch the server with a fork if corpus_endpoint is a directory
        self._server_pid = CorpusClient.ensure_server(self)

    def cleanup_corpus_server(self):
        if not hasattr(self, '_server_pid') or self._server_pid < 0:
            # no corpus server was started
            return
        if self.world_size > 1:
            torch.distributed.barrier()  # wait for everyone to finish before killing the corpus server
        if self._server_pid > 0:
            os.kill(self._server_pid, signal.SIGKILL)


hypers = RagHypers().fill_from_args()

# initialize the model and index
tokenizer = RagTokenizer.from_pretrained(hypers.model_name)
if 'rag-token' in hypers.model_name:
    model = RagTokenForGeneration.from_pretrained(hypers.model_path if hypers.model_path else hypers.model_name)
elif 'rag-sequence' in hypers.model_name:
    model = RagSequenceForGeneration.from_pretrained(hypers.model_path if hypers.model_path else hypers.model_name)
else:
    raise AssertionError

model = model.to(hypers.device)
model.train()
# construct rest retriever after the model
rest_retriever = CorpusClient(hypers.corpus_endpoint, model, tokenizer)
optimizer = TransformerOptimize(hypers, hypers.num_train_epochs * hypers.num_instances, model)
batch_count = 0
if hypers.n_gpu < 1:
    raise ValueError('Must have GPU')
# torch.autograd.set_detect_anomaly(True)


def retrieve(queries, answers):
    input_dict = prepare_seq2seq_batch(tokenizer, queries, answers, return_tensors="pt")
    input_ids = input_dict['input_ids'].to(model.device)
    attention_mask = input_dict['attention_mask'].to(model.device)
    context_input_ids, context_attention_mask, doc_scores, docs = \
        rest_retriever.retrieve(input_ids, attention_mask, n_docs=hypers.n_docs)
    return context_input_ids.reshape(len(queries) * hypers.n_docs, -1), \
           context_attention_mask.reshape(len(queries) * hypers.n_docs, -1), \
           doc_scores.reshape(len(queries), hypers.n_docs), \
           input_ids, attention_mask, input_dict['labels'].to(optimizer.hypers.device)


def one_batch(queries, answers):
    global batch_count
    context_input_ids, context_attention_mask, doc_scores, input_ids, attention_mask, labels = retrieve(queries, answers)

    outputs = optimizer.model(labels=labels,
                              context_input_ids=context_input_ids, context_attention_mask=context_attention_mask,
                              doc_scores=doc_scores)
    if batch_count == 0:
        print(f'logits shape = {outputs.logits.shape}')
    batch_count += 1
    optimizer.step_loss(outputs.loss.mean(),
                        retrieval_time=rest_retriever.retrieval_time/(batch_count * hypers.per_gpu_train_batch_size))


def train():
    rand = random.Random(hypers.seed)
    if hypers.fold:
        fold_num, fold_count = [int(part.strip()) for part in hypers.fold.split('of')]
        assert fold_num <= fold_count
        assert fold_count >= 1
        rel_by_fold, count_by_fold = get_relations_by_fold(hypers.kilt_data, fold_count)
        print(f'instance distribution by fold = {count_by_fold/count_by_fold.sum()}')
    else:
        fold_num, fold_count = 1, 1
        rel_by_fold = []
    query_batch = []
    answer_batch = []
    while True:
        optimizer.model.train()
        dataset = read_lines(hypers.kilt_data, shuffled_files=rand)
        for line_ndx, line in enumerate(block_shuffle(dataset, rand=rand, block_size=100000)):
            if line_ndx % hypers.world_size != hypers.global_rank:
                continue
            inst = json.loads(line)
            if hypers.fold:
                relation = get_relation_from_inst(inst)
                if relation in rel_by_fold[fold_num-1]:
                    # we exclude one fold
                    continue
            input_text = inst['input']
            answers = inst['answers'] if 'answers' in inst else [ai['answer'] for ai in inst['output']]
            query_batch.append(input_text)
            if hypers.no_leading_space:
                answer_batch.append(answers[0])
            else:
                answer_batch.append(' ' + answers[0])
            if len(query_batch) == hypers.per_gpu_train_batch_size * hypers.n_gpu:
                one_batch(query_batch, answer_batch)
                if not optimizer.should_continue():
                    return
                query_batch = []
                answer_batch = []


train()
(optimizer.model.module if hasattr(optimizer.model, "module") else optimizer.model).save_pretrained(hypers.output_dir)

hypers.cleanup_corpus_server()
