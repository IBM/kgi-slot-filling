#!/usr/bin/env python
# encoding: utf-8
from flask import Flask, request, jsonify
import base64
import numpy as np
from util.args_help import fill_from_args
import os
import logging
from dpr.simple_mmap_dataset import Corpus
from dpr.faiss_index import ANNIndex
import random

logger = logging.getLogger(__name__)


class Options():
    def __init__(self):
        self.port = 5001
        self.corpus_dir = ''
        self.model_name = 'facebook/rag-token-nq'
        self.rest_dtype = 16
        self.local_only = False  # only accessible on same machine
        self.debug = False
        self.log_info = False
        self.__required_args__ = ['corpus_dir']

    def get_rest_dtype(self):
        return np.float32 if self.rest_dtype == 32 else np.float16


def run(opts: Options):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if opts.log_info else logging.WARNING)
    app = Flask(__name__)
    if not opts.log_info:
        log = logging.getLogger('werkzeug')
        log.disabled = True
        app.logger.disabled = True
        app.logger.setLevel(logging.WARNING)
    # we either have a single index.faiss or we have an index for each offsets/passages
    if os.path.exists(os.path.join(opts.corpus_dir, "index.faiss")):
        passages = Corpus(os.path.join(opts.corpus_dir))
        index = ANNIndex(os.path.join(opts.corpus_dir, "index.faiss"))
        shards = None
        dim = index.dim()
    else:
        shards = []
        # loop over the different index*.faiss
        # so we have a list of (index, passages)
        # we search each index, then take the top-k results overall
        for filename in os.listdir(opts.corpus_dir):
            if filename.startswith('passages') and filename.endswith('.json.gz.records'):
                name = filename[len("passages"):-len(".json.gz.records")]
                shards.append((ANNIndex(os.path.join(opts.corpus_dir, f'index{name}.faiss')),
                               Corpus(os.path.join(opts.corpus_dir, f'passages{name}.json.gz.records'))))
        dim = shards[0][0].dim()
        assert all([dim == shard[0].dim() for shard in shards])
        print(f'Using sharded faiss with {len(shards)} shards.')
    print(dim)
    dummy_doc = {'pid': 'N/A', 'title': '', 'text': '', 'vector': np.zeros(dim, dtype=opts.get_rest_dtype())}

    @app.route('/config', methods=['GET'])
    def get_config():
        return jsonify({'dtype': opts.rest_dtype, 'dim': dim, 'corpus': opts.corpus_dir})

    def merge_results(query_vectors, k):
        # CONSIDER: consider ResultHeap (https://github.com/matsui528/faiss_tips)
        all_scores = np.zeros((query_vectors.shape[0], k * len(shards)), dtype=np.float32)
        all_indices = np.zeros((query_vectors.shape[0], k * len(shards)), dtype=np.int64)
        for si, shard in enumerate(shards):
            index_i, passages_i = shard
            scores, indexes = index_i.search(query_vectors, k)
            assert len(scores.shape) == 2
            assert scores.shape[1] == k
            assert scores.shape == indexes.shape
            assert scores.dtype == np.float32
            assert indexes.dtype == np.int64
            all_scores[:, si * k: (si + 1) * k] = scores
            all_indices[:, si * k: (si + 1) * k] = indexes
        kbest = all_scores.argsort()[:, -k:][:, ::-1]
        docs = [[shards[ndx // k][1][all_indices[bi, ndx]] for ndx in ndxs] for bi, ndxs in enumerate(kbest)]
        return docs

    def _random_docs(batch_size: int, k: int):
        num_passages = sum([len(s[1]) for s in shards]) if shards is not None else len(passages)

        def get_random():
            ndx = random.randint(0, num_passages-1)
            if shards is None:
                return passages[ndx]
            offset = 0
            for si in range(len(shards)):
                if ndx - offset < len(shards[si][1]):
                    return shards[si][1][ndx - offset]
                offset += len(shards[si][1])
            raise ValueError

        return [[get_random() for _ in range(k)] for _ in range(batch_size)]

    def _get_docs_by_pids(pids, *, dummy_if_missing=False):
        docs = []
        for pid in pids:
            doc = None
            if shards is None:
                doc = passages.get_by_pid(pid)
            else:
                for shard in shards:
                    doc = shard[1].get_by_pid(pid)
                    if doc is not None:
                        break
            if doc is None:
                if dummy_if_missing:
                    doc = dummy_doc
                else:
                    raise ValueError
            docs.append(doc)
        return docs

    @app.route('/fetch', methods=['POST'])
    def fetch_docs():
        rest_dtype = opts.get_rest_dtype()
        query = request.get_json()
        # input is 'pids': list of list of ids to get
        # and boolean for include vectors
        include_vectors = 'include_vectors' in query and query['include_vectors']
        dummy_if_missing = 'dummy_if_missing' in query and query['dummy_if_missing']
        batch_size = len(query['pids'])
        k = len(query['pids'][0])
        docs = [_get_docs_by_pids(pids, dummy_if_missing=dummy_if_missing) for pids in query['pids']]
        assert all([len(d) == k for d in docs])
        doc_dicts = [{'pid': [dqk['pid'] for dqk in dq],
                      'title': [dqk['title'] for dqk in dq],
                      'text': [dqk['text'] for dqk in dq]} for dq in docs]

        retval = {'docs': doc_dicts}
        if include_vectors:
            doc_vectors = np.zeros([batch_size, k, dim], dtype=rest_dtype)
            for qi, docs_qi in enumerate(docs):
                for ki, doc_qi_ki in enumerate(docs_qi):
                    doc_vectors[qi, ki] = doc_qi_ki['vector']
            retval['doc_vectors'] = base64.b64encode(doc_vectors).decode('ascii')

        return jsonify(retval)

    @app.route('/retrieve', methods=['POST'])
    def retrieve_docs():
        rest_dtype = opts.get_rest_dtype()
        query = request.get_json()
        # input is three parts:
        #  the base64 encoded fp16 numpy matrix
        #  k (the number of records per document)
        #  return-vectors flag
        k = query['k']
        include_vectors = 'include_vectors' in query and query['include_vectors']
        get_random = 'get_random' in query and query['get_random']
        if get_random:
            batch_size = query['batch_size']
            docs = _random_docs(batch_size, k)
        else:
            query_vectors = np.frombuffer(base64.decodebytes(query['query_vectors'].encode('ascii')), dtype=rest_dtype).reshape(-1, dim)
            query_vectors = query_vectors.astype(np.float32)
            batch_size = query_vectors.shape[0]
            assert query_vectors.shape[1] == dim
            if shards is None:
                scores, indexes = index.search(query_vectors, k)
                docs = [[passages[ndx] for ndx in ndxs] for ndxs in indexes]
            else:
                docs = merge_results(query_vectors, k)

        # add the gold_pids to the docs if requested
        if 'gold_pids' in query:
            gold_pids = query['gold_pids']
            assert len(gold_pids) == batch_size
            gdocs = []
            for qi in range(batch_size):
                gpids = gold_pids[qi][:k]
                assert isinstance(gpids, list)
                gold_docs = _get_docs_by_pids(gpids)
                gdocs.append(gold_docs + [dqk for dqk in docs[qi] if dqk['pid'] not in gpids][:k-len(gold_docs)])
            docs = gdocs
            assert all([len(d) == k for d in docs])

        if 'pid' in docs[0][0]:
            doc_dicts = [{'pid': [dqk['pid'] for dqk in dq],
                          'title': [dqk['title'] for dqk in dq],
                          'text': [dqk['text'] for dqk in dq]} for dq in docs]
        else:
            doc_dicts = [{'title': [dqk['title'] for dqk in dq],
                          'text': [dqk['text'] for dqk in dq]} for dq in docs]

        retval = {'docs': doc_dicts}
        if include_vectors:
            doc_vectors = np.zeros([batch_size, k, dim], dtype=rest_dtype)
            if not get_random:
                for qi, docs_qi in enumerate(docs):
                    if 'gold_pids' in query:
                        gpids = query['gold_pids'][qi]
                    else:
                        gpids = []
                    for ki, doc_qi_ki in enumerate(docs_qi):
                        # if we have gold_pids, set their vector to 100 * the query vector
                        if ki < len(gpids):
                            doc_vectors[qi, ki] = 100 * query_vectors[qi]
                        else:
                            doc_vectors[qi, ki] = doc_qi_ki['vector']
            retval['doc_vectors'] = base64.b64encode(doc_vectors).decode('ascii')

        # print(retval)
        # output
        #   list of docs: len(docs) == query_vectors.shape[0]; len(docs[i].title) == len(docs[i].text) == k
        #   doc_vectors: query_vectors.shape[0] x k x query_vectors.shape[1]
        return jsonify(retval)

    app.run(host='127.0.0.1' if opts.local_only else '0.0.0.0', debug=opts.debug, port=opts.port)


if __name__ == '__main__':
    opts = Options()
    fill_from_args(opts)
    run(opts)
