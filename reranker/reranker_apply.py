import logging
from util.reporting import Reporting
from reranker.reranker_model import RerankerHypers, load
import torch
import ujson as json
from util.line_corpus import write_open, jsonl_lines, read_open
import torch.nn.functional as F
from eval.kilt.eval_downstream import evaluate
from eval.convert_for_kilt_eval import to_distinct_doc_ids

logger = logging.getLogger(__name__)


class RerankerApplyArgs(RerankerHypers):
    """
    Arguments and hypers
    """
    def __init__(self):
        super().__init__()
        self.output = ''
        self.kilt_data = ''
        self.max_batch_size = 64
        self.exclude_instances = ''
        self.include_passages = False
        self.__required_args__ = ['model_type', 'model_name_or_path',
                                  'output', 'initial_retrieval']


def one_instance(args, model, tokenizer, query, passages):
    texts_a = [query] * len(passages)
    texts_b = [p['title'] + '\n\n' + p['text'] for p in passages]
    all_probs = []
    for start_ndx in range(0, len(texts_a), args.max_batch_size):
        inputs = tokenizer(
            texts_a[start_ndx:start_ndx+args.max_batch_size],
            texts_b[start_ndx:start_ndx+args.max_batch_size],
            add_special_tokens=True,
            return_tensors='pt',
            max_length=args.max_seq_length,
            padding='longest',
            truncation=True)
        inputs = {n: t.to(model.device) for n, t in inputs.items()}
        probs = F.softmax(model(**inputs)[0].detach().cpu(), dim=-1)[:, 1].numpy().tolist()
        all_probs.extend(probs)
    return all_probs


def main():
    args = RerankerApplyArgs()
    args.fill_from_args()
    args.set_seed()
    assert args.world_size == 1 and args.n_gpu == 1  # TODO: support distributed

    # load model and tokenizer
    model, tokenizer = load(args)
    if args.exclude_instances:
        with read_open(args.exclude_instances) as f:
            exclude_instances = set(json.load(f))
    else:
        exclude_instances = None
    model.eval()
    report = Reporting()
    with torch.no_grad(), write_open(args.output) as output:
        for line in jsonl_lines(args.initial_retrieval):
            jobj = json.loads(line)
            inst_id = jobj['id']
            if exclude_instances and inst_id in exclude_instances:
                continue
            query = jobj['input']
            passages = jobj['passages']
            probs = one_instance(args, model, tokenizer, query, passages)
            scored_pids = [(p['pid'], prob) for p, prob in zip(passages, probs)]
            scored_pids.sort(key=lambda x: x[1], reverse=True)
            wids = to_distinct_doc_ids([pid for pid, prob in scored_pids])  # convert to Wikipedia document ids
            pred_record = {'id': inst_id, 'input': query, 'scored_pids': scored_pids,
                           'output': [{'answer': '', 'provenance': [{'wikipedia_id': wid} for wid in wids]}]}
            if args.include_passages:
                pred_record['passages'] = passages
            output.write(json.dumps(pred_record) + '\n')
            if report.is_time():
                print(f'Finished {report.check_count}; {report.check_count/report.elapsed_seconds()} per second.')
    if args.kilt_data:
        evaluate(args.kilt_data, args.output)


if __name__ == "__main__":
    main()
