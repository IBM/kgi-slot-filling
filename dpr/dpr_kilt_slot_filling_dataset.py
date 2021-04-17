from util.line_corpus import read_lines, jsonl_lines, write_open
from collections import defaultdict
from util.args_help import fill_from_args
import ujson as json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

# convert KILT slot filling datasets to dataloader_biencoder format
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
        self.kilt_data = ''
        self.passage_ids = ''
        self.output_file = ''
        self.min_overlap = 0.75
        self.for_supporting_passage = False


opts = Options()
fill_from_args(opts)

# first load passage_ids.txt, create map doc_id -> passage_ids
doc_id2pids = defaultdict(list)
for line in read_lines(opts.passage_ids):
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


total = 0
no_provenace_count = 0
with write_open(opts.output_file) as f:
    for line in jsonl_lines(opts.kilt_data):
        total += 1
        jobj = json.loads(line)
        inst_id = jobj['id']
        input = jobj['input']
        answers = [o['answer'] for o in jobj['output']]
        pid_with_overlap = []
        for output in jobj['output']:
            if 'provenance' in output:
                for prov in output['provenance']:
                    doc_id = prov['wikipedia_id']
                    start_para = prov['start_paragraph_id']
                    end_para = prov['end_paragraph_id'] + 1  # adjust to exclusive end
                    pid_with_overlap.extend(matching_passage_ids(doc_id2pids[doc_id], start_para, end_para))
        if len(pid_with_overlap) == 0:
            no_provenace_count += 1
            continue
        pid_with_overlap.sort(key=lambda x: x[1], reverse=True)
        min_overlap = min(opts.min_overlap, pid_with_overlap[0][1])
        positives = [pid for pid, o in pid_with_overlap if o >= min_overlap]
        if opts.for_supporting_passage:
            query = input + ' [SEP] ' + answers[0]
        else:
            query = input
        f.write(json.dumps({'id': inst_id, 'query': query, 'answers': answers,
                            'positive_pids': positives, 'overlap_pids': pid_with_overlap})+'\n')
logger.info(f'instances with no passage {no_provenace_count} out of {total}')
