import os
from util.line_corpus import read_lines, jsonl_lines, write_open
from collections import defaultdict
from util.args_help import fill_from_args
import ujson as json
import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

# convert KILT datasets to dataloader_biencoder format
# the provenance for KILT answers has a wikipedia_id, and start/end paragraph ids, we need to map these to a set of overlapping passages
"""
{"id": "935392b3-c206-4036-94b0-09bc35319c45",
"input": "Cirith Ungol [SEP] genre",
"output": [{"answer": "heavy metal", "provenance": [{"wikipedia_id": "9065794", "title": "King of the Dead (album)", "section": "Section::::Abstract.", "start_paragraph_id": 1, "start_character": 84, "end_paragraph_id": 1, "end_character": 161, "bleu_score": 1.0}]},
{"answer": "heavy metal", "provenance": [{"wikipedia_id": "9065986", "title": "One Foot in Hell", "section": "Section::::Abstract.", "start_paragraph_id": 1, "start_character": 153, "end_paragraph_id": 1, "end_character": 266, "bleu_score": 1.0}]},
{"answer": "Heavy Metal"}, {"answer": "metal"}, {"answer": "heavy metal music"}, {"answer": "Metal"}, {"answer": "Metal music"}, {"answer": "Heavy Metal Music"}], "meta": {"subj_aliases": [], "sub_surface": ["Cirith Ungol"], "obj_surface": ["Heavy metal", "Heavy Metal", "heavy metal music", "heavy metal", "heavy-metal", "heavy-metal music", "hard rock", "traditional heavy metal", "thrash/black metal", "heavy rock", "Heavy Metal Music"]}}

{"id": "a64179cb-01fd-42e9-9f40-bf3f0a1ad1e4",
"input": "USA-88 [SEP] time of spacecraft launch",
"output": [{"answer": "3 February 1993", "provenance": [{"wikipedia_id": "36387881", "title": "USA-88", "start_paragraph_id": 2, "start_character": 0, "end_paragraph_id": 2, "end_character": 145, "bleu_score": 1.0, "meta": {}, "section": "Section::::Abstract."}]}],
"meta": {"template_questions": ["What is the launch date of USA-88?", "What day was USA-88 launched?", "When was the launch date of USA-88?", "When was USA-88's launch date?", "On what date did USA-88 take off?", "What date was USA-88 launched?", "What was the launch date of USA-88?", "What was the date of USA-88's launch?", "On what date did USA-88 launch?", "On what date was USA-88 launched?"]}}
"""


class Options:
    def __init__(self):
        self.kilt_data_dir = ''
        self.kilt_passages = ''
        self.passage_ids = ''
        self.min_overlap = 0.75
        self.__required_args__ = ['kilt_data_dir', 'passage_ids', 'kilt_passages']


opts = Options()
fill_from_args(opts)

kilt_data_suffixes = ['-train-kilt.jsonl', '-dev-kilt.jsonl']

# first load passage_ids.txt, create map doc_id -> passage_ids
doc_id2pids = defaultdict(list)
for line in read_lines(opts.passage_ids, report_every=2000000):
    line = line.strip()
    doc_id_end = line.find('::')
    doc_id = line[:doc_id_end]
    range = line[doc_id_end+3:-1]
    range_open = line[doc_id_end+2]
    range_end = line[-1]
    start, end = [float(r) for r in range.split(',')]
    if range_open == '(':
        start += 0.5
    else:
        assert range_open == '['
    if range_end == ')' and end > start:
        end += 0.5
    else:
        assert range_end == ']' or end == start
        end += 1.0
    doc_id2pids[doc_id].append((line, start, end))


def matching_passage_ids(pids, start_para, end_para):
    def overlap(pas_start, pas_end):
        return (min(pas_end, end_para) - max(pas_start, start_para)) / (end_para - start_para)

    pid_with_overlap = [(pid, overlap(start, end)) for pid, start, end in pids if start < end_para and end > start_para]
    return pid_with_overlap


def get_overlap_and_positive(jobj):
    pid_with_overlap = []
    for output in jobj['output']:
        if 'provenance' in output:
            for prov in output['provenance']:
                doc_id = prov['wikipedia_id']
                start_para = prov['start_paragraph_id']
                end_para = prov['end_paragraph_id'] + 1  # adjust to exclusive end
                pid_with_overlap.extend(matching_passage_ids(doc_id2pids[doc_id], start_para, end_para))
    pid_with_overlap = list(set(pid_with_overlap))
    if len(pid_with_overlap) == 0:
        return pid_with_overlap, []
    pid_with_overlap.sort(key=lambda x: x[1], reverse=True)
    min_overlap = min(opts.min_overlap, pid_with_overlap[0][1])
    positives = list(set([pid for pid, o in pid_with_overlap if o >= min_overlap]))
    return pid_with_overlap, positives


pids = set()
for kilt_data_file in os.listdir(opts.kilt_data_dir):
    if not any([kilt_data_file.endswith(s) for s in kilt_data_suffixes]):
        continue
    for line in jsonl_lines(os.path.join(opts.kilt_data_dir, kilt_data_file)):
        jobj = json.loads(line)
        _, positives = get_overlap_and_positive(jobj)
        for p in positives:
            pids.add(p)
pid2passage = dict()
for line in jsonl_lines(opts.kilt_passages):
    jobj = json.loads(line)
    pid = jobj['pid']
    if pid in pids:
        pid2passage[pid] = jobj


for kilt_data_file in os.listdir(opts.kilt_data_dir):
    if not any([kilt_data_file.endswith(s) for s in kilt_data_suffixes]):
        continue
    total = 0
    no_provenace_count = 0
    output_file = os.path.join(opts.kilt_data_dir,
                               ('dev' if kilt_data_file.endswith('-dev-kilt.jsonl') else 'train')+
                               '_positive_pids.jsonl')
    passage_count_distribution = np.zeros(6, dtype=np.float)
    with write_open(output_file) as f:
        for line in jsonl_lines(os.path.join(opts.kilt_data_dir, kilt_data_file)):
            total += 1
            jobj = json.loads(line)
            inst_id = jobj['id']
            input = jobj['input']
            answers = [o['answer'] for o in jobj['output'] if 'answer' in o]
            if len(answers) == 0:
                print(f'WARNING: no answers: {line}')
            pid_with_overlap, positives = get_overlap_and_positive(jobj)
            pcount = len(positives) if len(positives) < len(passage_count_distribution) else len(passage_count_distribution) - 1
            passage_count_distribution[pcount] += 1
            if len(pid_with_overlap) == 0:
                no_provenace_count += 1
                continue
            f.write(json.dumps({'id': inst_id, 'query': input, 'answers': answers,
                                'positive_pids': positives, 'overlap_pids': pid_with_overlap,
                                'positive_passages': [pid2passage[pid] for pid in positives]})+'\n')
    logger.info(f'{kilt_data_file} instances with no passage {no_provenace_count} out of {total}')
    # display distribution of number of positive passages
    passage_count_distribution *= 1.0/passage_count_distribution.sum()
    percentages = [(f'{c}: ' if c < len(passage_count_distribution)-1 else f'>={c}: ') +
                   ('<1%' if 0.0 < p < 0.01 else f'{int(round(100*p))}%')
                   for c, p in enumerate(passage_count_distribution)]
    logger.info(f'Passage counts: {"; ".join(percentages)}')
