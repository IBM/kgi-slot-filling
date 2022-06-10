from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, write_open
import ujson as json
from collections import Counter
from eval.kilt.eval_downstream import evaluate
from eval.convert_for_kilt_eval import to_distinct_doc_ids

# when doing the training for a BM25 + DPR -> reranker from the retrieved from index
# we want to be able to take the output of two files like from dpr_apply, then combine into one


class Options:
    def __init__(self):
        self.initial_retrievals = ''
        self.top_k = 20
        self.output = ''
        self.weights = ''
        self.kilt_data = ''


opts = Options()
fill_from_args(opts)
initial_retrieval_files = [file for file in opts.initial_retrievals.split(',')]
id2info = dict()
id2passages = [dict() for _ in range(len(initial_retrieval_files))]
if not opts.weights:
    weights = [1.0 - 0.001 * ii for ii in range(len(initial_retrieval_files))]
else:
    weights = [float(w) for w in opts.weights.split(',')]
    assert len(weights) == len(initial_retrieval_files)

for file_ndx, file in enumerate(initial_retrieval_files):
    for line in jsonl_lines(file):
        jobj = json.loads(line)
        id = jobj['id']
        if file_ndx == 0:
            id2info[id] = jobj
        id2passages[file_ndx][id] = jobj['passages']

with write_open(opts.output) as f:
    for id, info in id2info.items():
        pid2inv_rank = Counter()
        pid2passage = dict()
        for weight, id2passage in zip(weights, id2passages):
            passages = id2passage[id]
            for ii, passage in enumerate(passages):
                pid = passage['pid']
                inv_rank = 1.0 / (1.0 + ii)
                pid2inv_rank[pid] += weight * inv_rank
                pid2passage[pid] = passage

        scored_pids = [(pid, inv_rank) for pid, inv_rank in pid2inv_rank.items()]
        scored_pids.sort(key=lambda x:x[1], reverse=True)
        all_passages = [pid2passage[pid] for pid, _ in scored_pids][:opts.top_k]
        info['passages'] = all_passages
        wids = to_distinct_doc_ids([passage['pid'] for passage in all_passages])
        info['output'] = [{'answer': '', 'provenance': [{'wikipedia_id': wid} for wid in wids]}]
        f.write(json.dumps(info)+'\n')

if opts.kilt_data:
    evaluate(opts.kilt_data, opts.output)
