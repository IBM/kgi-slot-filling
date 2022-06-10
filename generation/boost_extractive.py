from util.args_help import fill_from_args
from util.line_corpus import jsonl_lines, write_open
import ujson as json
from generation.rag_util import _extractive_normalize
from eval.convert_for_kilt_eval import convert_for_kilt_eval


class Options:
    def __init__(self):
        self.prediction_file = ''
        self.kilt_data = ''
        self.output = ''
        self.extractive_boost = 1.0
        self.__required_args__ = ['prediction_file', 'output']


opts = Options()
fill_from_args(opts)

# add a bonus the the score of extractive predictions, then re-rank
with write_open(opts.output) as f:
    for line in jsonl_lines(opts.prediction_file):
        jobj = json.loads(line)
        inst_id = jobj['id']
        contexts = jobj['contexts']
        predictions = jobj['predictions']
        scores = jobj['predictions_scores']
        all_norm_context = ' ' + ' '.join([_extractive_normalize(context) for context in contexts]) + ' '
        is_extractive = [(' ' + _extractive_normalize(p) + ' ' in all_norm_context) for p in predictions]  # or (p in ['yes', 'no'])
        boosted = [s + opts.extractive_boost if e else s for e, s in zip(is_extractive, scores)]
        sort = [(p, s) for p, s in zip(predictions, boosted)]
        sort.sort(key=lambda x: x[1], reverse=True)
        jobj['predictions'] = [p for p, s in sort]
        jobj['predictions_scores'] = [float(s) for p, s in sort]
        f.write(json.dumps(jobj)+'\n')


kilt_format_output = opts.output[:-6] + '_kilt_format.jsonl'
convert_for_kilt_eval(opts.output, kilt_format_output, opts.kilt_data)
