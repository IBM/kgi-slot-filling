from transformers import (PreTrainedTokenizer, BartTokenizerFast)
from transformers import (RagSequenceForGeneration, RagTokenForGeneration, RagTokenizer)
import torch
from util.line_corpus import read_lines, write_open
import ujson as json
from typing import List
import numpy as np
import functools
from slot_filling.rag_client_server_train import RagHypers
from util.reporting import Reporting
import logging
from slot_filling.corpus_client import CorpusClient
from slot_filling.rag_util import prepare_seq2seq_batch, normalize_answer
from slot_filling.dataset_stats import get_relations_by_fold, get_relation_from_inst


logger = logging.getLogger(__name__)


class Options(RagHypers):
    def __init__(self):
        super().__init__()
        self.output = ''
        self.use_candidates = False
        self.no_context = False
        self.num_beams = 4
        self.num_return_sequences = 4
        self.n_docs_for_provenance = 20  # we'll supply this many document ids for reporting provenance
        self.limit = -1
        self.retrieve_batch_size = 8
        # self.batch_size = 8
        self.__required_args__ = ['kilt_data', 'output', 'corpus_endpoint']

    def _post_argparse(self):
        self.num_beams = max(self.num_beams, self.num_return_sequences)


opts = Options().fill_from_args()
torch.set_grad_enabled(False)

# initialize the model and index
tokenizer = RagTokenizer.from_pretrained(opts.model_name)
# rag_conf = RagConfig.from_pretrained(opts.model_name)
if 'rag-token' in opts.model_name:
    model = RagTokenForGeneration.from_pretrained(opts.model_path if opts.model_path else opts.model_name)
elif 'rag-sequence' in opts.model_name:
    model = RagSequenceForGeneration.from_pretrained(opts.model_path if opts.model_path else opts.model_name)
else:
    raise AssertionError

model = model.to(opts.device)
model.eval()
# construct rest retriever after the model
rest_retriever = CorpusClient(opts.corpus_endpoint, model, tokenizer)

generated_not_in_candidate_count = 0
report = Reporting()
sum_rr = 0
count = 0


def tokenize_candidates(generator_tokenizer: PreTrainedTokenizer, batch_candidates: List[List[str]]):
    batch_candidates_ids = []
    for candidates in batch_candidates:
        candidates_ids = []
        for cand in candidates:
            if not opts.no_leading_space:
                cand = ' ' + cand
            candidates_ids.append(np.array(generator_tokenizer.convert_tokens_to_ids(generator_tokenizer.tokenize(cand)), dtype=np.int32))
        batch_candidates_ids.append(candidates_ids)
    return batch_candidates_ids


def prefix_allowed_tokens_fn(gtokenizer: BartTokenizerFast, candidates_text, batch_candidates: List[List[np.ndarray]],
                             batch_id: int, input_ids: torch.Tensor) -> List[int]:
    assert len(input_ids.shape) == 1
    prefix = input_ids.cpu().detach().numpy()
    candidates = batch_candidates[batch_id]
    allowed = set()
    # CONSIDER: always allow the special tokens, create cand_prefix as the non-special tokens
    if len(prefix) == 0:
        allowed.add(gtokenizer.eos_token_id)
    if len(prefix) == 1:
        allowed.add(gtokenizer.bos_token_id)
    if len(prefix) > 2 and prefix[-1] == gtokenizer.eos_token_id:
        allowed.add(gtokenizer.pad_token_id)
    if len(prefix) > 1:
        cand_prefix = prefix[2:]
        for cand in candidates:
            if len(cand) < len(cand_prefix):
                continue
            if all(cand[:len(cand_prefix)] == cand_prefix):
                if len(cand) > len(cand_prefix):
                    allowed.add(cand[len(cand_prefix)])
                else:
                    allowed.add(gtokenizer.eos_token_id)
    if not allowed:
        allowed.add(gtokenizer.eos_token_id)
        allowed.add(gtokenizer.pad_token_id)
    return list(allowed)


def retrieve(queries):
    input_dict = prepare_seq2seq_batch(tokenizer, queries, return_tensors="pt")
    input_ids = input_dict['input_ids'].to(model.device)
    attention_mask = input_dict['attention_mask'].to(model.device)
    with torch.no_grad():
        # retrieve support docs
        context_input_ids, context_attention_mask, doc_scores, docs = \
            rest_retriever.retrieve(input_ids, attention_mask,
                                    n_docs=opts.n_docs, n_docs_for_provenance=opts.n_docs_for_provenance)
        if 'id' in docs[0]:
            retrieved_doc_ids = [dd['id'] for dd in docs]
        elif 'pid' in docs[0]:
            retrieved_doc_ids = [dd['pid'] for dd in docs]
        else:
            retrieved_doc_ids = [[0] * len(dd['text']) for dd in docs]  # dummy ids
    # .reshape(len(queries), opts.n_docs, -1)
    return context_input_ids.reshape(len(queries), opts.n_docs, -1), \
           context_attention_mask.reshape(len(queries), opts.n_docs, -1), \
           doc_scores.reshape(len(queries), opts.n_docs), retrieved_doc_ids


def generate_one_instance(candidates, context_input_ids, context_attention_mask, doc_scores):
    """
    :param candidates: list of strings
    :param context_input_ids: n_docs x seq_len
    :param context_attention_mask: n_docs x seq_len
    :param doc_scores: n_docs
    :return:
    """
    # CONSIDER: try leading space for query too?
    global generated_not_in_candidate_count
    with torch.no_grad():
        special_token_ids = {tokenizer.generator.eos_token_id,
                             tokenizer.generator.bos_token_id,
                             tokenizer.generator.pad_token_id}
        if candidates:
            batch_candidate_ids = tokenize_candidates(tokenizer.generator, [candidates])
            prefix_allowed_tokens = functools.partial(prefix_allowed_tokens_fn,
                                                      tokenizer.generator,
                                                      [candidates],
                                                      batch_candidate_ids)
        else:
            batch_candidate_ids = None
            prefix_allowed_tokens = None

        if opts.use_candidates:
            assert prefix_allowed_tokens is not None

        num_return_sequences = opts.num_return_sequences if not candidates else len(candidates)
        # because it runs out of memory if there are too many
        if num_return_sequences > 16:
            num_return_sequences = 16
        # generate answers
        beam_search_output = model.generate(
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            num_beams=max(int(1.5 * num_return_sequences), opts.num_beams),
            num_return_sequences=num_return_sequences,
            min_length=2 if not candidates else min([len(c) for cands in batch_candidate_ids for c in cands]),
            max_length=64 if not candidates else 3 + max([len(c) for cands in batch_candidate_ids for c in cands]),
            length_penalty=1.0,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            return_dict_in_generate=True, output_scores=True
        )
        # BeamSearchDecoderOnlyOutput: sequences, sequences_scores
        generated_ids = beam_search_output.sequences.detach().cpu().numpy()
        generated_scores = beam_search_output.sequences_scores.detach().cpu().numpy()

        answer_strings = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if candidates:
            answer_strings = []
            candidate_ids = [list(c) for c in batch_candidate_ids[0]]
            for gen_ids in generated_ids:
                gen = [i for i in gen_ids if i not in special_token_ids]
                try:
                    ndx = candidate_ids.index(gen)
                    if candidates[ndx] not in answer_strings:
                        answer_strings.append(candidates[ndx])
                except ValueError:
                    if generated_not_in_candidate_count == 0:
                        logger.warning(f'not found: {gen} in {candidate_ids}')
                    generated_not_in_candidate_count += 1
                    pass

        return answer_strings, generated_scores.tolist()


def record_one_instance(output, inst_id, input_text, candidates, answers, pred_text, pred_scores, doc_ids, context_input_ids):
    global sum_rr
    global count
    pred_record = {'id': inst_id, 'input': input_text,
                   'predictions': pred_text, 'predictions_scores': pred_scores, 'doc_ids': doc_ids}
    if answers:
        pred_record['answers'] = answers
    if not opts.no_context:
        contexts = tokenizer.batch_decode(context_input_ids, skip_special_tokens=True)
        pred_record['contexts'] = [c[:c.rfind('//')] for c in contexts]

    rank = float("Inf")
    if opts.use_candidates:
        pred_record['candidates'] = candidates
        rank = len(candidates)
    if answers:
        norm_pred_text = [normalize_answer(a) for a in pred_text]
        norm_answers = [normalize_answer(a) for a in answers]
        for ans in norm_answers:
            try:
                ndx = norm_pred_text.index(ans)
                if ndx + 1 < rank:
                    rank = ndx + 1
            except ValueError:
                pass
        sum_rr += 1.0 / rank
        count += 1
    output.write(json.dumps(pred_record) + '\n')
    if report.is_time():
        metrics = f' MRR = {sum_rr/count}' if count > 0 else ''
        print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second. '
              f'{1000 * rest_retriever.retrieval_time/report.check_count} msecs per retrieval.{metrics}')


def one_batch(id_batch, query_batch, candidate_batch, answer_batch, output):
    context_input_ids, context_attention_mask, doc_scores, retrieved_doc_ids = retrieve(query_batch)
    # print(f'retrieved shapes: {context_input_ids.shape}, {context_attention_mask.shape}, {doc_scores.shape}, {retrieved_doc_ids}')
    for bi in range(len(query_batch)):
        context_ids_i = context_input_ids[bi]
        answer_strings, answer_scores = generate_one_instance(candidate_batch[bi], context_ids_i,
                                                              context_attention_mask[bi], doc_scores[bi:bi+1])
        record_one_instance(output, id_batch[bi], query_batch[bi], candidate_batch[bi], answer_batch[bi],
                            answer_strings, answer_scores, retrieved_doc_ids[bi], context_ids_i)


def get_answers(inst):
    if 'answers' in inst:
        return inst['answers']
    elif 'output' in inst:
        return [ai['answer'] for ai in inst['output']]
    else:
        return []


if opts.world_size > 1:
    raise ValueError('Distributed not supported')
has_answers = False
# TODO: use hypers.output_dir and filename with global_rank
with write_open(opts.output) as output:
    if opts.fold:
        fold_num, fold_count = [int(part.strip()) for part in opts.fold.split('of')]
        assert fold_num <= fold_count
        assert fold_count >= 1
        rel_by_fold, count_by_fold = get_relations_by_fold(opts.kilt_data, fold_count)
        print(f'instance distribution by fold = {count_by_fold/count_by_fold.sum()}')
    else:
        fold_num, fold_count = 1, 1
        rel_by_fold = []
    id_batch, query_batch, candidate_batch, answer_batch = [], [], [], []
    for line_ndx, line in enumerate(read_lines(opts.kilt_data)):
        # TODO: skip line_ndx not for our rank
        inst = json.loads(line)
        if opts.fold:
            relation = get_relation_from_inst(inst)
            if relation not in rel_by_fold[fold_num-1]:
                # we apply on one fold
                continue
        id_batch.append(inst['id'])
        query_batch.append(inst['input'])
        answers = get_answers(inst)
        if answers:
            has_answers = True
        answer_batch.append(answers)
        candidate_batch.append(inst['candidates'] if opts.use_candidates else None)

        if len(query_batch) == opts.retrieve_batch_size:
            one_batch(id_batch, query_batch, candidate_batch, answer_batch, output)
            id_batch, query_batch, candidate_batch, answer_batch = [], [], [], []
        if opts.limit > 0 and line_ndx+1 >= opts.limit:
            break
    if len(query_batch) > 0:
        one_batch(id_batch, query_batch, candidate_batch, answer_batch, output)
    metrics = f' MRR = {sum_rr/count}' if count > 0 else ''
    print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second. '
          f'{1000 * rest_retriever.retrieval_time/report.check_count} msecs per retrieval.{metrics}')

if generated_not_in_candidate_count > 0:
    print(f'Generated {generated_not_in_candidate_count} not in candidate list')

opts.cleanup_corpus_server()
