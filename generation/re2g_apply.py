from transformers import RagTokenizer, RagSequenceForGeneration, RagTokenForGeneration
from dpr.retriever_dpr_bm25_reranker import RetrieverDPRReranker
import torch
from util.line_corpus import read_lines, jsonl_lines, write_open
import ujson as json
from generation.rag_util import tokenize_candidates, prefix_allowed_tokens_fn, extractive_allowed_tokens_fn
import logging
import functools
from eval.kilt.kilt_eval import evaluate, normalize_answer
from util.reporting import Reporting
from eval.convert_for_kilt_eval import convert_for_kilt_eval, get_answers
from generation.re2g_hypers import Re2gApplyHypers


logger = logging.getLogger(__name__)

assert __name__ == "__main__"
hypers = Re2gApplyHypers().fill_from_args()

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


model = get_rag_model(hypers)
model.rag.question_encoder = None
model = model.to(hypers.device)
model.eval()

generated_not_in_candidate_count = 0
report = Reporting()
sum_rr = 0
count = 0

retriever = RetrieverDPRReranker(hypers.reranker, hypers.dpr, hypers.bm25, apply_mode=True)


batch_count = 0
if hypers.n_gpu != 1:
    raise ValueError('Must have GPU, multi-GPU must be DistributedDataParallel')

id2positive_pids = None
if hypers.positive_pids:
    id2positive_pids = dict()
    for line in jsonl_lines(hypers.positive_pids):
        jobj = json.loads(line)
        id2positive_pids[jobj['id']] = jobj['positive_pids']


def retrieve(queries, inst_ids):
    positive_pids = [id2positive_pids[inst_id] for inst_id in inst_ids] if id2positive_pids is not None else None
    context_input_ids, context_attention_mask, doc_scores, docs, ifb = \
        retriever.retrieve_forward(queries, positive_pids=positive_pids)

    retrieved_doc_ids = [dd['pid'] for dd in docs]
    return context_input_ids.reshape(len(queries), hypers.reranker.n_docs, -1).to(hypers.device), \
           context_attention_mask.reshape(len(queries), hypers.reranker.n_docs, -1).to(hypers.device), \
           doc_scores.reshape(len(queries), hypers.reranker.n_docs), retrieved_doc_ids


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
        elif hypers.extractive_only:
            prefix_allowed_tokens = functools.partial(extractive_allowed_tokens_fn,
                                                      tokenizer.generator,
                                                      [context_input_ids.reshape(model.config.n_docs, -1)])
        else:
            batch_candidate_ids = None
            prefix_allowed_tokens = None

        if hypers.use_candidates or hypers.extractive_only:
            assert prefix_allowed_tokens is not None

        num_return_sequences = hypers.num_return_sequences if not candidates else len(candidates)
        # because it runs out of memory if there are too many
        if num_return_sequences > 16:
            num_return_sequences = 16
        # generate answers
        beam_search_output = model.generate(
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            num_beams=max(int(1.5 * num_return_sequences), hypers.num_beams),
            num_return_sequences=num_return_sequences,
            min_length=2 if not candidates else min([len(c) for cands in batch_candidate_ids for c in cands]),
            max_length=hypers.max_target_length if not candidates else 3 + max([len(c) for cands in batch_candidate_ids for c in cands]),
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
    if not hypers.no_context:
        contexts = tokenizer.batch_decode(context_input_ids, skip_special_tokens=True)
        pred_record['contexts'] = [c[:c.rfind('//')] for c in contexts]

    rank = float("Inf")
    if hypers.use_candidates:
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
        print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second.{metrics}')


def one_batch(id_batch, query_batch, candidate_batch, answer_batch, output):
    context_input_ids, context_attention_mask, doc_scores, retrieved_doc_ids = retrieve(query_batch, id_batch)
    # print(f'retrieved shapes: {context_input_ids.shape}, {context_attention_mask.shape}, {doc_scores.shape}, {retrieved_doc_ids}')
    for bi in range(len(query_batch)):
        context_ids_i = context_input_ids[bi]
        answer_strings, answer_scores = generate_one_instance(candidate_batch[bi], context_ids_i,
                                                              context_attention_mask[bi], doc_scores[bi:bi+1])
        record_one_instance(output, id_batch[bi], query_batch[bi], candidate_batch[bi], answer_batch[bi],
                            answer_strings, answer_scores, retrieved_doc_ids[bi], context_ids_i)


if hypers.world_size > 1:
    raise ValueError('Distributed not supported')
has_answers = False
# TODO: use hypers.output_dir and filename with global_rank
with write_open(hypers.output) as output:
    id_batch, query_batch, candidate_batch, answer_batch = [], [], [], []
    for line_ndx, line in enumerate(read_lines(hypers.kilt_data)):
        # TODO: skip line_ndx not for our rank
        inst = json.loads(line)
        id_batch.append(inst['id'])
        query_batch.append(inst['input'])
        answers = get_answers(inst)
        if answers:
            has_answers = True
        answer_batch.append(answers)
        candidate_batch.append(inst['candidates'] if hypers.use_candidates else None)

        if len(query_batch) == hypers.retrieve_batch_factor:
            one_batch(id_batch, query_batch, candidate_batch, answer_batch, output)
            id_batch, query_batch, candidate_batch, answer_batch = [], [], [], []
        if hypers.limit > 0 and line_ndx+1 >= hypers.limit:
            break
    if len(query_batch) > 0:
        one_batch(id_batch, query_batch, candidate_batch, answer_batch, output)
    metrics = f' MRR = {sum_rr/count}' if count > 0 else ''
    print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second.{metrics}')

if generated_not_in_candidate_count > 0:
    print(f'Generated {generated_not_in_candidate_count} not in candidate list')

if has_answers:
    # only evaluate on full set
    result = evaluate(gold=hypers.kilt_data, guess=hypers.output)
    result_file = hypers.output[:-6]+'_eval.json' if hypers.output[-6:] == '.jsonl' else hypers.output+'_eval.json'
    with write_open(result_file) as f:
        json.dump(result, f, indent=4)

kilt_format_output = hypers.output[:-6] + '_kilt_format.jsonl' if hypers.output[-6:] == '.jsonl' \
    else hypers.output + '_kilt_format.jsonl'
# chain into convert_for_kilt_eval
convert_for_kilt_eval(hypers.output, kilt_format_output, hypers.kilt_data if has_answers else '')

retriever.cleanup_corpus_server()
