from collections import Counter
from util.line_corpus import read_lines
from util.args_help import fill_from_args
import ujson as json
import numpy as np


class Options:
    def __init__(self):
        self.train = ''
        self.dev = ''
        self.test = ''


def get_relation_counts(kilt_file: str):
    rel_counts = Counter()
    inst_count = 0
    for line in read_lines(kilt_file):
        jobj = json.loads(line)
        rel = jobj['input'].split(' [SEP] ')[1]
        rel_counts[rel] += 1
        inst_count += 1
    rel_counts = [(rel, count) for rel, count in rel_counts.items()]
    rel_counts.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return rel_counts, inst_count


def get_relation_from_inst(json_inst):
    return json_inst['input'].split(' [SEP] ')[1]


def get_relations_by_fold(kilt_file: str, num_folds: int):
    rel_counts, inst_count = get_relation_counts(kilt_file)
    rel_by_fold = [set() for _ in range(num_folds)]
    inst_count_by_fold = np.zeros(num_folds)
    for ndx, (r, c) in enumerate(rel_counts):
        fold = ndx % num_folds
        rel_by_fold[fold].add(r)
        inst_count_by_fold[fold] += c
    # print(f'fold distribution = {inst_count_by_fold/inst_count}')
    return rel_by_fold, inst_count_by_fold


if __name__ == "__main__":
    opts = Options()
    fill_from_args(opts)

    for file in [opts.train, opts.dev, opts.test]:
        if not file:
            continue
        print(f'{file}\n'+'='*80)
        rel_counts, inst_count = get_relation_counts(file)
        for r, c in rel_counts:
            print(f'{r}\t{c}')
        print('='*80+f'\n{len(rel_counts)} relations, {inst_count} instances\n')
        # get_relations_by_fold(file, 2)
