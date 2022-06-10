from util.line_corpus import read_lines, write_open
from util.args_help import fill_from_args
import ujson as json
import numpy as np
from eval.kilt.kilt_eval import normalize_answer
from eval.kilt.eval_downstream import evaluate
import os

# FROM:
# "predictions":[" 3 February 1993"," 3 September 1993"," 3 October 1993"," 3 March 1993"],
# "predictions_scores":[-0.04270831122994423,-0.971720814704895,-0.9760023951530457,-0.9918134808540344],
# "doc_ids":["36387881::[0,2]","557227::[0,2]","36387881::[3,3]","557227::[19,19)","557227::[29,30]"]

# TO:
# "output": [{"answer": "3 February 1993", "provenance": [{"wikipedia_id": "36387881"}]}]


class Options:
    def __init__(self):
        self.apply_file = ''
        self.eval_file = ''
        self.gold_file = ''
        self.__required_args__ = ['apply_file', 'eval_file']


def get_answers(inst):
    if 'answers' in inst:
        return inst['answers']
    elif 'output' in inst:
        return [ai['answer'] for ai in inst['output'] if 'answer' in ai]  # [ai['answer'] for ai in inst['output']]
    else:
        return []


def kilt_answers(inst, *, normalize_train_answer=False, prefer_extractive=False, no_leading_space=False):
    answers = get_answers(inst)
    assert len(answers) > 0
    if normalize_train_answer:
        nanswers = [normalize_answer(a) for a in answers]
        nanswers = [a.strip() for a in nanswers if len(a.strip()) > 0]
        if len(nanswers) > 0:
            if prefer_extractive:
                answers = answers + nanswers
            else:
                answers = nanswers
    if no_leading_space:
        return answers
    else:
        return [' ' + a for a in answers]


def to_distinct_doc_ids(passage_ids):
    doc_ids = []
    for pid in passage_ids:
        doc_id = pid[:pid.find(':')]
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def convert_for_kilt_eval(apply_file, eval_file, gold_file):
    doc_counts = np.zeros(5, dtype=np.int32)
    with write_open(eval_file) as f:
        for line in read_lines(apply_file):
            jobj = json.loads(line)
            predictions = jobj['predictions']
            doc_ids = jobj['doc_ids']
            wids = to_distinct_doc_ids(doc_ids)  # convert to Wikipedia document ids
            doc_count = len(wids)
            doc_counts[-1 + (doc_count if doc_count <= 5 else 5)] += 1
            output = [{'answer': predictions[0],
                       'provenance': [{'wikipedia_id': wid} for wid in wids]}]
            f.write(json.dumps({'id': jobj['id'], 'output': output}) + '\n')

    print(f'documents retrieved (1-5+) = {doc_counts/doc_counts.sum()}')
    if gold_file:
        run_eval(gold_file, eval_file)


def run_eval(gold_file, eval_file):
    result = evaluate(gold_file, eval_file)
    """ Example:
    {'downstream': {'accuracy': 0.9062619685944082,
                    'em': 0.9062619685944082,
                    'f1': 0.9062619685944082,
                    'rougel': 0.9062619640629098},
     'kilt': {'KILT-accuracy': 0.787437763309077,
              'KILT-em': 0.787437763309077,
              'KILT-f1': 0.787437763309077,
              'KILT-rougel': 0.7874377593717473},
     'retrieval': {'Rprec': 0.8835863836151089, 'recall@5': 0.8846174305289582}}
    """
    rprec = result['retrieval']['Rprec']
    recall5 = result['retrieval']['recall@5']
    em = result['downstream']['em']
    acc = result['downstream']['accuracy']
    rougel = result['downstream']['rougel']
    f1 = result['downstream']['f1']
    kilt_em = result['kilt']['KILT-em']
    kilt_acc = result['kilt']['KILT-accuracy']
    kilt_rougel = result['kilt']['KILT-rougel']
    kilt_f1 = result['kilt']['KILT-f1']
    # QA style
    dataset = os.path.split(gold_file)[1]
    dataset = dataset[:dataset.index('-')]
    if dataset in ['nq', 'triviaqa', 'eli5', 'hotpotqa']:
        print(f'{rprec:.6f}\t{recall5:.6f}\t{em:.6f}\t{f1:.6f}\t{kilt_em:.6f}\t{kilt_f1:.6f}')
    elif dataset in ['trex', 'structured_zeroshot']:
        print(f'{rprec:.6f}\t{recall5:.6f}\t{acc:.6f}\t{f1:.6f}\t{kilt_acc:.6f}\t{kilt_f1:.6f}')
    elif dataset in ['fever']:
        print(f'{rprec:.6f}\t{recall5:.6f}\t{acc:.6f}\t{kilt_acc:.6f}')
    elif dataset in ['wow']:
        print(f'{rprec:.6f}\t{recall5:.6f}\t{rougel:.6f}\t{f1:.6f}\t{kilt_rougel:.6f}\t{kilt_f1:.6f}')
    else:
        print(f'unknown dataset {dataset}')
        print(f'{rprec:.6f}\t{recall5:.6f}\t{em:.6f}\t{f1:.6f}\t{kilt_em:.6f}\t{kilt_f1:.6f}')


if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)

    convert_for_kilt_eval(opts.apply_file, opts.eval_file, opts.gold_file)

