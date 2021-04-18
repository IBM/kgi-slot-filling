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
    passages = Corpus(os.path.join(opts.corpus_dir))
    index = ANNIndex(os.path.join(opts.corpus_dir, "index.faiss"))
    dim = index.dim()
    print(dim)

    @app.route('/config', methods=['GET'])
    def get_config():
        return jsonify({'dtype': opts.rest_dtype, 'dim': dim, 'corpus': opts.corpus_dir})

    @app.route('/retrieve', methods=['POST'])
    def retrieve_docs():
        rest_dtype = opts.get_rest_dtype()
        query = request.get_json()
        # input is three parts:
        #  the base64 encoded fp16 numpy matrix
        #  k (the number of records per document)
        #  return-vectors flag
        query_vectors = np.frombuffer(base64.decodebytes(query['query_vectors'].encode('ascii')), dtype=rest_dtype).reshape(-1, dim)
        k = query['k']
        include_vectors = 'include_vectors' in query and query['include_vectors']

        query_vectors = query_vectors.astype(np.float32)

        scores, indexes = index.search(query_vectors, k)

        docs = [[passages[ndx] for ndx in ndxs] for ndxs in indexes]

        if 'pid' in docs[0][0]:
            doc_dicts = [{'pid': [dqk['pid'] for dqk in dq],
                          'title': [dqk['title'] for dqk in dq],
                          'text': [dqk['text'] for dqk in dq]} for dq in docs]
        else:
            doc_dicts = [{'title': [dqk['title'] for dqk in dq],
                          'text': [dqk['text'] for dqk in dq]} for dq in docs]

        retval = {'docs': doc_dicts}
        if include_vectors:
            doc_vectors = np.zeros([query_vectors.shape[0], k, query_vectors.shape[1]], dtype=rest_dtype)
            for qi, docs_qi in enumerate(docs):
                for ki, doc_qi_ki in enumerate(docs_qi):
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
