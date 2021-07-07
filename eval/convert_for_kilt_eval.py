from util.line_corpus import read_lines, write_open
from util.args_help import fill_from_args
import ujson as json
import numpy as np

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
        self.__required_args__ = ['apply_file', 'eval_file']


opts = Options()
fill_from_args(opts)


def to_distinct_doc_ids(passage_ids):
    doc_ids = []
    for pid in passage_ids:
        doc_id = pid[:pid.find(':')]
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


doc_counts = np.zeros(5, dtype=np.int32)
with write_open(opts.eval_file) as f:
    for line in read_lines(opts.apply_file):
        jobj = json.loads(line)
        predictions = jobj['predictions']
        doc_ids = jobj['doc_ids']  # these are actually passage ids
        wids = to_distinct_doc_ids(doc_ids)  # convert to Wikipedia document ids
        doc_count = len(wids)
        doc_counts[-1 + (doc_count if doc_count <= 5 else 5)] += 1
        output = [{'answer': predictions[0],
                   'provenance': [{'wikipedia_id': wid} for wid in wids]}]
        f.write(json.dumps({'id': jobj['id'], 'output': output})+'\n')

print(f'documents retrieved (1-5+) = {doc_counts/doc_counts.sum()}')


