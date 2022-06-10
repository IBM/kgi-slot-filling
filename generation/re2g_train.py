from torch_util.transformer_optimize import TransformerOptimize, LossHistory
from transformers import RagTokenizer, RagSequenceForGeneration, RagTokenForGeneration
from dpr.retriever_dpr_bm25_reranker import RetrieverDPRReranker
import torch
from util.line_corpus import read_lines, block_shuffle, jsonl_lines
import ujson as json
import random
from generation.rag_util import prepare_seq2seq_batch_labels, prefered_answers
import logging
from eval.convert_for_kilt_eval import kilt_answers
from generation.re2g_hypers import Re2gHypers


logger = logging.getLogger(__name__)


assert __name__ == "__main__"
hypers = Re2gHypers().fill_from_args()

# initialize the model and index
tokenizer = RagTokenizer.from_pretrained(hypers.model_name)


def get_rag_model(hypers):
    if 'rag-token' in hypers.model_name:
        model = RagTokenForGeneration.from_pretrained(hypers.model_path if hypers.model_path else hypers.model_name)
    elif 'rag-sequence' in hypers.model_name:
        model = RagSequenceForGeneration.from_pretrained(hypers.model_path if hypers.model_path else hypers.model_name)
    else:
        raise AssertionError

    return model


retriever = RetrieverDPRReranker(hypers.reranker, hypers.dpr, hypers.bm25)

model = get_rag_model(hypers)
model.rag.question_encoder = None
model = model.to(hypers.device)
optimizer = TransformerOptimize(hypers, hypers.num_train_epochs * hypers.num_instances, model)
loss_history = LossHistory(hypers.num_instances //
                           (hypers.full_train_batch_size // hypers.gradient_accumulation_steps))

batch_count = 0
if hypers.n_gpu != 1:
    raise ValueError('Must have GPU, multi-GPU must be DistributedDataParallel')
# torch.autograd.set_detect_anomaly(True)

id2positive_pids = None
if hypers.positive_pids:
    id2positive_pids = dict()
    for line in jsonl_lines(hypers.positive_pids):
        jobj = json.loads(line)
        id2positive_pids[jobj['id']] = jobj['positive_pids']


def retrieve(queries, inst_ids, *, rdocs=None):
    positive_pids = [id2positive_pids[inst_id] if inst_id in id2positive_pids else None for inst_id in inst_ids] \
        if id2positive_pids is not None else None
    context_input_ids, context_attention_mask, doc_scores, docs, ifb = \
        retriever.retrieve_forward(queries, positive_pids=positive_pids, rdocs=rdocs)

    return context_input_ids.reshape(len(queries) * hypers.reranker.n_docs, -1).to(optimizer.hypers.device), \
           context_attention_mask.reshape(len(queries) * hypers.reranker.n_docs, -1).to(optimizer.hypers.device), \
           doc_scores.reshape(len(queries), hypers.reranker.n_docs), ifb, docs


def one_batch(queries, answers, inst_ids):
    n_docs = hypers.reranker.n_docs
    global batch_count
    gen_batch_size = hypers.per_gpu_train_batch_size
    ret_batch_size = hypers.dpr.per_gpu_train_batch_size
    assert ret_batch_size > gen_batch_size and ret_batch_size % gen_batch_size == 0
    assert len(queries) == len(answers) == ret_batch_size
    context_input_ids, context_attention_mask, doc_scores, ifb, docs = retrieve(queries, inst_ids)
    # select either first possible answer or the answer that occurs in the highest ranked document
    panswers = prefered_answers(answers, docs, prefer_extractive=hypers.prefer_extractive)  # CONSIDER: prefer_short=hypers.prefer_short_answer
    labels = prepare_seq2seq_batch_labels(tokenizer, panswers, return_tensors="pt",
                                          max_target_length=hypers.max_target_length).to(optimizer.hypers.device)
    doc_scores_grad = torch.zeros((ret_batch_size, n_docs), dtype=torch.float32, device=hypers.device)
    assert doc_scores.shape == doc_scores_grad.shape
    assert ret_batch_size//gen_batch_size == hypers.retrieve_batch_factor

    batch_loss = 0
    prev_step = optimizer.global_step
    for sub_batch_i in range(ret_batch_size//gen_batch_size):
        bstart, bend = gen_batch_size * sub_batch_i, gen_batch_size * (sub_batch_i + 1)
        doc_scores_sbi = doc_scores[bstart:bend]
        doc_scores_sbi.requires_grad = True
        outputs = optimizer.model(labels=labels[bstart:bend],
                                  context_input_ids=context_input_ids[bstart * n_docs:bend * n_docs],
                                  context_attention_mask=context_attention_mask[bstart * n_docs:bend * n_docs],
                                  doc_scores=doc_scores_sbi)
        if batch_count == 0 or random.random() < 1/10000:
            contexts = [c.replace('<pad>', '') for c in tokenizer.batch_decode(context_input_ids[bstart * n_docs:bend * n_docs])]
            targets = [t.replace('<pad>', '') for t in tokenizer.batch_decode(labels[bstart:bend])]
            # print(f'rag_retriever_train logits shape = {outputs.logits.shape}')
            logger.info(f'contexts = {contexts}\ntargets = {targets}')
                        # f'contexts.shape = {context_input_ids[bstart * n_docs:bend * n_docs].shape}\n'
                        # f'labels.shape = {labels[bstart:bend].shape}')
        batch_count += 1
        loss_val = optimizer.backward_on_loss(outputs.loss.mean())
        batch_loss += loss_val
        loss_history.note_loss(loss_val)
        doc_scores_grad[bstart:bend, :] = doc_scores_sbi.grad.detach()
        optimizer.optimizer_step()
    # I think we multiply by gradient_accumulation_steps here,
    #   since we will divide the loss by gradient accumulation steps again in the reranker optimizer
    retriever.retrieve_backward(ifb, doc_scores_grad=doc_scores_grad*hypers.reranker.gradient_accumulation_steps)
    # if in debug mode, re-run this in forward and check that loss is lower
    if hypers.debug:
        assert optimizer.global_step > prev_step
        # We pass in the rdocs we retrieved before - we are not redoing retrieval
        context_input_ids, context_attention_mask, doc_scores, ifb, docs = retrieve(queries, inst_ids, rdocs=ifb.dpr_ifb.rdocs)
        labels = prepare_seq2seq_batch_labels(tokenizer, panswers, return_tensors="pt").to(optimizer.hypers.device)
        post_update_loss = 0
        for sub_batch_i in range(ret_batch_size // gen_batch_size):
            bstart, bend = gen_batch_size * sub_batch_i, gen_batch_size * (sub_batch_i + 1)
            doc_scores_sbi = doc_scores[bstart:bend]
            doc_scores_sbi.requires_grad = True
            outputs = optimizer.model(labels=labels[bstart:bend],
                                      context_input_ids=context_input_ids[bstart * n_docs:bend * n_docs],
                                      context_attention_mask=context_attention_mask[bstart * n_docs:bend * n_docs],
                                      doc_scores=doc_scores_sbi)

            post_update_loss += outputs.loss.mean().item()
        logger.info(f'Batch loss = {batch_loss} -> {post_update_loss}')


def train():
    rand = random.Random(hypers.seed)
    query_batch, answer_batch, id_batch = [], [], []
    missing_positive_pids = 0
    while True:
        optimizer.model.train()
        dataset = read_lines(hypers.kilt_data, shuffled_files=rand)
        for line_ndx, line in enumerate(block_shuffle(dataset, rand=rand, block_size=100000)):
            if line_ndx % hypers.world_size != hypers.global_rank:
                continue
            inst = json.loads(line)

            query_batch.append(inst['input'])
            inst_id = inst['id']
            id_batch.append(inst_id)
            if id2positive_pids and inst_id not in id2positive_pids:
                id2positive_pids[inst_id] = []
                missing_positive_pids += 1

            answer_batch.append(kilt_answers(inst,
                                             normalize_train_answer=hypers.normalize_train_answer,
                                             prefer_extractive=hypers.prefer_extractive,
                                             no_leading_space=hypers.no_leading_space))

            if len(query_batch) == hypers.dpr.per_gpu_train_batch_size:
                one_batch(query_batch, answer_batch, id_batch)
                if not optimizer.should_continue():
                    if missing_positive_pids > 0:
                        logger.info(f'Missing {missing_positive_pids} positive pids')
                    return
                query_batch, answer_batch, id_batch = [], [], []


train()
if hypers.global_rank == 0:
    (optimizer.model.module if hasattr(optimizer.model, "module") else optimizer.model).save_pretrained(hypers.output_dir)
    retriever.save()
logger.info(f'loss_history = {loss_history.loss_history}')
retriever.cleanup_corpus_server()