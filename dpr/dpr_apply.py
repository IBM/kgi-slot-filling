from transformers import (DPRQuestionEncoder, RagTokenizer, RagTokenForGeneration)
import torch
from util.line_corpus import read_lines, write_open
import ujson as json
from util.reporting import Reporting
import logging
from torch_util.hypers_base import HypersBase
from corpus.corpus_client import CorpusClient
from eval.convert_for_kilt_eval import to_distinct_doc_ids
from eval.kilt.eval_downstream import evaluate
import numpy as np

logger = logging.getLogger(__name__)


class Options(HypersBase):
    def __init__(self):
        super().__init__()
        self.output = ''
        self.kilt_data = ''
        self.corpus_endpoint = ''
        self.qry_encoder_path = ''
        self.rag_model_path = ''
        self.n_docs_for_provenance = 20  # we'll supply this many document ids for reporting provenance
        self.retrieve_batch_size = 32
        self.include_passages = False  # if set, we return the list of passages too
        self.__required_args__ = ['kilt_data', 'output', 'corpus_endpoint']

    def _post_init(self):
        super()._post_init()
        # launch the server with a fork if corpus_endpoint is a directory
        self._server_pid = CorpusClient.ensure_server(self)

    def cleanup_corpus_server(self):
        CorpusClient.cleanup_corpus_server(self)


opts = Options().fill_from_args()
torch.set_grad_enabled(False)

tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq')
# support loading from either a rag_model_path or qry_encoder_path
if opts.qry_encoder_path:
    assert not opts.rag_model_path
    qencoder = DPRQuestionEncoder.from_pretrained(opts.qry_encoder_path)
elif opts.rag_model_path:
    model = RagTokenForGeneration.from_pretrained(opts.rag_model_path)
    qencoder = model.question_encoder
else:
    raise ValueError('must supply either qry_encoder_path or rag_model_path')

qencoder = qencoder.to(opts.device)
qencoder.eval()
rest_retriever = CorpusClient(opts.corpus_endpoint, None, tokenizer)
report = Reporting()


def retrieve(queries):
    if opts.include_passages:
        doc_scores, docs = rest_retriever.string_retrieve(qencoder, queries, n_docs=opts.n_docs_for_provenance, return_scores=True)
    else:
        docs = rest_retriever.string_retrieve(qencoder, queries, n_docs=opts.n_docs_for_provenance)
        doc_scores = np.zeros(len(queries), opts.n_docs_for_provenance)
    if 'id' in docs[0]:
        retrieved_doc_ids = [dd['id'] for dd in docs]
    elif 'pid' in docs[0]:
        retrieved_doc_ids = [dd['pid'] for dd in docs]
    else:
        retrieved_doc_ids = [[0] * len(dd['text']) for dd in docs]  # dummy ids
    passages = None
    if opts.include_passages:
        passages = [{'titles': dd['title'], 'texts': dd['text'], 'scores': doc_scores[dndx].tolist()} for dndx, dd in enumerate(docs)]
    return retrieved_doc_ids, passages


def record_one_instance(output, inst_id, input, doc_ids, passages):
    wids = to_distinct_doc_ids(doc_ids)
    pred_record = {'id': inst_id, 'input': input, 'output': [{'answer': '', 'provenance': [{'wikipedia_id': wid} for wid in wids]}]}
    if passages:
        pred_record['passages'] = [{'pid': pid, 'title': title, 'text': text, 'score': float(score)}
                                   for pid, title, text, score in zip(doc_ids, passages['titles'], passages['texts'], passages['scores'])]
    output.write(json.dumps(pred_record) + '\n')
    if report.is_time():
        print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second. '
              f'{1000 * rest_retriever.retrieval_time/report.check_count} msecs per retrieval.')


def one_batch(id_batch, query_batch, output):
    retrieved_doc_ids, passages = retrieve(query_batch)
    for bi in range(len(query_batch)):
        record_one_instance(output, id_batch[bi], query_batch[bi], retrieved_doc_ids[bi], passages[bi] if passages else None)


if opts.world_size > 1:
    raise ValueError('Distributed not supported')
with write_open(opts.output) as output:
    id_batch, query_batch = [], []
    for line_ndx, line in enumerate(read_lines(opts.kilt_data)):
        inst = json.loads(line)
        id_batch.append(inst['id'])
        query_batch.append(inst['input'])
        if len(query_batch) == opts.retrieve_batch_size:
            one_batch(id_batch, query_batch, output)
            id_batch, query_batch = [], []
    if len(query_batch) > 0:
        one_batch(id_batch, query_batch, output)
    print(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second. '
          f'{1000 * rest_retriever.retrieval_time/report.check_count} msecs per retrieval.')

opts.cleanup_corpus_server()

evaluate(opts.kilt_data, opts.output)
