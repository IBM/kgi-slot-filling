from util.line_corpus import write_open
from util.args_help import fill_from_args
from eval.kilt.eval_downstream import get_gold_answers, _exact_match_score, _metric_max_over_ground_truths, _f1_score, _rougel_score, retrieval_metrics, kilt_utils
import numpy as np
import scipy.stats as st
from random import Random


class Options:
    def __init__(self):
        self.eval_file = ''
        self.baseline_file = ''
        self.iter_count = 10000
        self.gold_file = ''
        self.output = ''
        self.style = 'wow'
        self.__required_args__ = ['gold_file', 'eval_file']


opts = Options()
fill_from_args(opts)


def per_instance_metrics(gold_records, guess_records, write_to_file):

    assert len(gold_records) == len(guess_records), \
        "different size gold: {} guess: {}".format(len(gold_records), len(guess_records))
    print(f'Evaluating over {len(gold_records)} instances')
    results = []
    out = write_open(write_to_file) if write_to_file else None
    for guess_item, gold_item in zip(guess_records, gold_records):

        # check ids
        assert (
            str(gold_item["id"]).strip() == str(guess_item["id"]).strip()
        ), "Items must have same order with same IDs"
        inst_id = str(gold_item["id"]).strip()

        # check if each output of guess file exist in set of candidate answers
        gold_candidate_answers = get_gold_answers(gold_item)

        conditions = (len(guess_item["output"]) == 1) and (
            "answer" in guess_item["output"][0]
        )
        assert (
            conditions
        ), f"you should provide exactly one valid answer for {guess_item['id']}"
        guess_answer = str(guess_item["output"][0]["answer"]).strip()

        # 0. accuracy = strict exact match
        local_accuracy = 0
        if guess_answer in gold_candidate_answers:
            local_accuracy = 1

        # 1. normalized exact match
        local_em = _metric_max_over_ground_truths(
            _exact_match_score, guess_answer, gold_candidate_answers
        )

        # 2. normalized f1
        local_f1 = _metric_max_over_ground_truths(
            _f1_score, guess_answer, gold_candidate_answers
        )

        # 3. rougel
        local_rougel = _metric_max_over_ground_truths(
            _rougel_score, guess_answer, gold_candidate_answers
        )

        # KILT-metrics
        Rprec = retrieval_metrics.rprecision(
            guess_item, gold_item, rank_keys=["wikipedia_id"]
        )
        rank, num_distinct_evidence_sets = retrieval_metrics.get_rank(guess_item, gold_item, 5, rank_keys=["wikipedia_id"])
        if num_distinct_evidence_sets > 0:
            recall5 = retrieval_metrics._recall_at_k(rank, num_distinct_evidence_sets, 5)
        else:
            print('Error!')
            recall5 = 0
        if Rprec == 1:
            kilt_accuracy = local_accuracy
            kilt_em = local_em
            kilt_f1 = local_f1
            kilt_rougel = local_rougel
        else:
            kilt_accuracy = 0
            kilt_em = 0
            kilt_f1 = 0
            kilt_rougel = 0
        if out is not None:
            out.write(f'{inst_id}\t{local_accuracy}\t{local_f1}\t{local_rougel}\t{Rprec}\t{recall5}\t{kilt_accuracy}\t{kilt_f1}\t{kilt_rougel}\n')
        results.append([local_accuracy, local_f1, local_rougel, Rprec, recall5, kilt_accuracy, kilt_f1, kilt_rougel])
    if out is not None:
        out.close()
    return np.array(results, dtype=np.float32)


gold_records = kilt_utils.load_data(opts.gold_file)
guess_records = kilt_utils.load_data(opts.eval_file)

metric_names = ['accuracy', 'f1', 'rougel', 'Rprec', 'recall5', 'kilt_accuracy', 'kilt_f1', 'kilt_rougel']
results = per_instance_metrics(gold_records, guess_records, opts.output)
if opts.baseline_file:
    if opts.style == 'wow':
        metrics = [3, 4, 2, 1, 7, 6]
    elif opts.style == 'trex':
        metrics = [3, 4, 0, 1, 5, 6]
    elif opts.style == 'fever':
        metrics = [3, 4, 0, 5]
    else:
        raise ValueError('no such style')
    base_results = per_instance_metrics(gold_records, kilt_utils.load_data(opts.baseline_file), '')
    assert results.shape == base_results.shape
    for metric_ndx in metrics:
        mean = np.mean(results[:, metric_ndx])
        base_mean = np.mean(base_results[:, metric_ndx])
        diff = abs(mean - base_mean)
        rand = Random(123)
        diff_geq_count = 0.0
        for iter_num in range(opts.iter_count):
            m1, m2 = 0, 0
            for inst_ndx in range(results.shape[0]):
                r1, r2 = (results, base_results) if rand.randint(0, 1) == 0 else (base_results, results)
                m1 += r1[inst_ndx, metric_ndx]
                m2 += r2[inst_ndx, metric_ndx]
            m1 /= results.shape[0]
            m2 /= results.shape[0]
            # if iter_num % 2000 == 0:
            #     print(f'    {iter_num}: {m1} vs {m2} | {abs(m1 - m2)} {">=" if abs(m1 - m2) >= diff else "<"} {diff}')
            if abs(m1 - m2) >= diff:
                diff_geq_count += 1.0
        print(f'for {metric_names[metric_ndx]}, {mean} vs {base_mean}, p = {diff_geq_count/opts.iter_count}')
else:
    prefix = '{\\small $\\pm$'
    suffix = '}'
    latex = dict()
    for metric_ndx in range(results.shape[1]):
        mean = np.mean(results[:, metric_ndx])
        minv, maxv = st.t.interval(0.95, results.shape[0]-1, loc=mean, scale=st.sem(results[:, metric_ndx]))
        print(f'{metric_names[metric_ndx]}: {mean * 100:.2f}, {(mean-minv) * 100:.2f}, {(maxv-mean) * 100:.2f}')
        latex[metric_ndx] = f'{mean * 100:.2f}{prefix}{(mean-minv) * 100:.2f}{suffix}'
        #print(f'{metric_names[metric_ndx]}: {mean}, {mean-minv}, {mean-maxv}, {minv}, {maxv}')
    if opts.style == 'wow':
        print(f'{latex[3]} & {latex[4]} & {latex[2]} & {latex[1]} & {latex[7]} & {latex[6]}')
    elif opts.style == 'trex':
        print(f'{latex[3]} & {latex[4]} & {latex[0]} & {latex[1]} & {latex[5]} & {latex[6]}')
    elif opts.style == 'fever':
        print(f'{latex[3]} & {latex[4]} & {latex[0]} &  & {latex[5]} & ')
