from util.line_corpus import read_lines, write_open
import ujson as json
from util.reporting import Reporting
import logging
from dpr.retriever_bm25 import BM25Hypers, RetrieverBM25
from util.args_help import fill_from_args
from eval.convert_for_kilt_eval import to_distinct_doc_ids
from eval.kilt.eval_downstream import evaluate

logger = logging.getLogger(__name__)


class Options(BM25Hypers):
    def __init__(self):
        super().__init__()
        self.output = ''
        self.kilt_data = ''
        self.include_passages = False  # if set, we return the list of passages too
        self.instance_limit = -1
        self.__required_args__ = ['kilt_data', 'output', 'anserini_index', 'jar']


opts = Options()
fill_from_args(opts)
retriever = RetrieverBM25(opts)
report = Reporting()


def retrieve(queries):
    doc_scores, docs = retriever.retrieve_forward(queries)
    if 'id' in docs[0]:
        retrieved_doc_ids = [dd['id'] for dd in docs]
    elif 'pid' in docs[0]:
        retrieved_doc_ids = [dd['pid'] for dd in docs]
    else:
        retrieved_doc_ids = [['0:0'] * len(dd['text']) for dd in docs]  # dummy ids
    passages = None
    if opts.include_passages:
        passages = [{'titles': dd['title'], 'texts': dd['text']} for dd in docs]
    assert type(retrieved_doc_ids) == list
    assert all([type(doc_ids) == list for doc_ids in retrieved_doc_ids])
    if not all([type(doc_id) == str for doc_ids in retrieved_doc_ids for doc_id in doc_ids]):
        print(f'Error: {retrieved_doc_ids}')
        raise ValueError('not right type')
    return retrieved_doc_ids, passages


def record_one_instance(output, inst_id, input, doc_ids, passages):
    wids = to_distinct_doc_ids(doc_ids)
    pred_record = {'id': inst_id, 'input': input, 'output': [{'answer': '', 'provenance': [{'wikipedia_id': wid} for wid in wids]}]}
    if passages:
        pred_record['passages'] = [{'pid': pid, 'title': title, 'text':text} for pid, title, text in zip(doc_ids, passages['titles'], passages['texts'])]
    output.write(json.dumps(pred_record) + '\n')
    if report.is_time():
        print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second.')


def one_batch(id_batch, query_batch, output):
    """
    retrieve and record one batch of queries
    :param id_batch:
    :param query_batch:
    :param output:
    :return:
    """
    retrieved_doc_ids, passages = retrieve(query_batch)
    for bi in range(len(query_batch)):
        record_one_instance(output, id_batch[bi], query_batch[bi], retrieved_doc_ids[bi], passages[bi] if passages else None)


with write_open(opts.output) as output:
    id_batch, query_batch = [], []
    for line_ndx, line in enumerate(read_lines(opts.kilt_data)):
        if 0 < opts.instance_limit <= line_ndx:
            break
        inst = json.loads(line)
        id_batch.append(inst['id'])
        query_batch.append(inst['input'])
        if len(query_batch) == 2 * opts.num_processes:
            one_batch(id_batch, query_batch, output)
            id_batch, query_batch = [], []
    if len(query_batch) > 0:
        one_batch(id_batch, query_batch, output)
    print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second.')

evaluate(opts.kilt_data, opts.output)

