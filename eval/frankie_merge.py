from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, write_open
import ujson as json
from eval.convert_for_kilt_eval import convert_for_kilt_eval


class Options:
    def __init__(self):
        self.kilt_data = ''
        self.downstream = ''
        self.retrieval = ''
        self.output = ''
        self.__required_args__ = ['downstream', 'retrieval', 'output']


opts = Options()
fill_from_args(opts)

id2doc_ids = dict()
for line in jsonl_lines(opts.retrieval):
    jobj = json.loads(line)
    inst_id = jobj['id']
    id2doc_ids[inst_id] = jobj['doc_ids']

with write_open(opts.output) as outf:
    for line in jsonl_lines(opts.downstream):
        jobj = json.loads(line)
        inst_id = jobj['id']
        jobj['doc_ids'] = id2doc_ids[inst_id]
        outf.write(json.dumps(jobj)+'\n')

kilt_format_output = opts.output[:-6] + '_kilt_format.jsonl' if opts.output[-6:] == '.jsonl' \
    else opts.output + '_kilt_format.jsonl'
# chain into convert_for_kilt_eval
convert_for_kilt_eval(opts.output, kilt_format_output, opts.kilt_data)
