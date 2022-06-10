import requests
import base64
import numpy as np
import json
import torch
import signal
from transformers import RagTokenForGeneration, RagTokenizer
from generation.rag_util import postprocess_docs
import logging
import time
import os
from torch_util.hypers_base import HypersBase
from generation.rag_util import prepare_seq2seq_batch
from util.reporting import Reporting


logger = logging.getLogger(__name__)


class CorpusClient:
    def __init__(self, endpoint: str, model: RagTokenForGeneration, tokenizer: RagTokenizer):
        self.endpoint = endpoint  # 'http://localhost:5001'
        self.model = model
        self.tokenizer = tokenizer
        self.retrieval_time = 0
        self.headers = {'Content-Type': 'application/json'}
        # get config info from server
        config = requests.get(self.endpoint+'/config', headers=self.headers).json()
        self.reporting = Reporting()
        self.rest_dtype = np.float32 if config['dtype'] == 32 else np.float16

    @staticmethod
    def ensure_server(hypers: HypersBase):
        if hypers.corpus_endpoint.startswith('http'):
            return -1  # -1, no one started a corpus server
        child_pid = 0
        port = hypers.port if hasattr(hypers, 'port') else 5001
        if not hasattr(hypers, 'global_rank') or hypers.global_rank == 0:
            child_pid = os.fork()
            if child_pid == 0:
                from corpus.corpus_server_direct import Options as FlaskOptions, run
                fopts = FlaskOptions()
                fopts.corpus_dir = hypers.corpus_endpoint
                fopts.port = port
                fopts.local_only = not hasattr(hypers, 'world_size') or hypers.world_size <= torch.cuda.device_count()
                run(fopts)
                exit(0)
        addr = os.environ['MASTER_ADDR'] if 'MASTER_ADDR' in os.environ else 'localhost'
        hypers.corpus_endpoint = f'http://{addr}:{port}'
        if not hasattr(hypers, 'global_rank') or hypers.global_rank == 0:
            # wait until server starts
            headers = {'Content-Type': 'application/json'}
            import requests
            while True:
                time.sleep(5)
                try:
                    test_config = requests.get(hypers.corpus_endpoint + '/config', headers=headers).json()
                    logger.warning(f'Started server: {test_config}')
                    break
                except:
                    logger.warning(f'Waiting on corpus server to start')
                    continue
        if hasattr(hypers, 'world_size') and hypers.world_size > 1:
            torch.distributed.barrier()
        return child_pid

    @staticmethod
    def cleanup_corpus_server(hypers: HypersBase, *, server_pid=None):
        if server_pid is None and hypers and hasattr(hypers, '_server_pid'):
            server_pid = hypers._server_pid
        if server_pid is None or server_pid < 0:
            # no corpus server was started
            return
        if hypers and hasattr(hypers, 'world_size') and hypers.world_size > 1:
            torch.distributed.barrier()  # wait for everyone to finish before killing the corpus server
        if server_pid > 0:
            os.kill(server_pid, signal.SIGKILL)

    def _rest_call(self, question_encoder_last_hidden_state: np.ndarray, n_docs: int):
        qstr = base64.b64encode(question_encoder_last_hidden_state.astype(self.rest_dtype)).decode('ascii')
        query = {'query_vectors': qstr, 'k': n_docs, 'include_vectors': True}
        start_time = time.time()
        response = requests.post(self.endpoint + '/retrieve', data=json.dumps(query), headers=self.headers)
        self.retrieval_time += time.time() - start_time
        rdocs = response.json()
        return rdocs

    def track_retrieval_metrics(self, positive_pids, docs):
        if positive_pids is None:
            return
        assert len(positive_pids) == len(docs)
        pids = [dd['pid'] for dd in docs]
        hit1 = 0
        inrecall = 0
        count = 0
        for positives, retrieved in zip(positive_pids, pids):
            if not positives:
                continue
            if retrieved[0] in positives:
                hit1 += 1
            if any([r in positives for r in retrieved]):
                inrecall += 1
            count += 1
        self.reporting.moving_averages(hit_1=hit1 / count, in_recall=inrecall / count)
        if self.reporting.is_time():
            self.reporting.display()

    def string_retrieve(self, question_encoder, query_strs, *, n_docs=5, return_scores=False):
        input_dict = prepare_seq2seq_batch(self.tokenizer, query_strs, return_tensors="pt")
        input_ids = input_dict['input_ids'].to(question_encoder.device)
        attention_mask = input_dict['attention_mask'].to(question_encoder.device)
        with torch.no_grad():
            question_enc_outputs = question_encoder(
                input_ids, attention_mask=attention_mask, return_dict=True
            )
            question_encoder_last_hidden_state = question_enc_outputs[0]  # hidden states of question encoder
        qstr = base64.b64encode(
            question_encoder_last_hidden_state.detach().cpu().numpy().astype(self.rest_dtype)).decode('ascii')
        query = {'query_vectors': qstr, 'k': n_docs, 'include_vectors': return_scores}
        start_time = time.time()
        response = requests.post(self.endpoint + '/retrieve', data=json.dumps(query), headers=self.headers)
        self.retrieval_time += time.time() - start_time
        rdocs = response.json()
        docs = rdocs['docs']

        if return_scores:
            doc_vectors = np.frombuffer(base64.decodebytes(rdocs['doc_vectors'].encode('ascii')), dtype=self.rest_dtype). \
                              reshape(-1, n_docs, question_encoder_last_hidden_state.shape[-1])[:, 0:n_docs, :]
            retrieved_doc_embeds = torch.Tensor(doc_vectors.copy()).to(question_encoder_last_hidden_state)
            with torch.no_grad():
                doc_scores = torch.bmm(
                    question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                ).squeeze(1).detach().cpu().numpy()
            return doc_scores, docs

        return docs

    def fetch(self, pids, *, include_vectors=True, dummy_if_missing=True):
        query = {'pids': pids, 'include_vectors': include_vectors, 'dummy_if_missing': dummy_if_missing}
        start_time = time.time()
        response = requests.post(self.endpoint + '/fetch', data=json.dumps(query), headers=self.headers)
        self.retrieval_time += time.time() - start_time
        try:
            rdocs = response.json()
        except json.decoder.JSONDecodeError:
            logger.error(f'Bad response: {response}\nRequest: {json.dumps(query)}')
            raise json.decoder.JSONDecodeError
        docs = rdocs['docs']
        if include_vectors:
            doc_vectors = np.frombuffer(base64.decodebytes(rdocs['doc_vectors'].encode('ascii')), dtype=self.rest_dtype).\
                reshape(len(pids), len(pids[0]), -1)
            retrieved_doc_embeds = torch.Tensor(doc_vectors.copy())
        else:
            retrieved_doc_embeds = None
        return docs, retrieved_doc_embeds

    def retrieve(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *,
                 n_docs=5, n_docs_for_provenance=-1, get_random=False, gold_pids=None):
        """
        :param input_ids:
        :param attention_mask:
        :param n_docs:
        :return: context_input_ids, Tensor (batch * n_docs x seq_len)
                 context_attention_mask, Tensor (batch * n_docs x seq_len)
                 doc_scores, Tensor (batch * n_docs)
                 docs, list of dict [{title: [t_1...t_n_docs_for_provenance], text: [...], pid: [...]}] * batch_size
        """
        if n_docs_for_provenance < n_docs:
            n_docs_for_provenance = n_docs
        question_enc_outputs = self.model.rag.question_encoder(
            input_ids, attention_mask=attention_mask, return_dict=True
        )
        question_encoder_last_hidden_state = question_enc_outputs[0]  # hidden states of question encoder
        if get_random:
            query = {'get_random': True, 'batch_size': input_ids.shape[0], 'k': n_docs_for_provenance, 'include_vectors': True}
        else:
            qstr = base64.b64encode(question_encoder_last_hidden_state.detach().cpu().numpy().astype(self.rest_dtype)).decode('ascii')
            query = {'query_vectors': qstr, 'k': n_docs_for_provenance, 'include_vectors': True}
        if gold_pids is not None:
            assert len(gold_pids) == input_ids.shape[0]
            assert all([isinstance(gp, list) for gp in gold_pids])
            query['gold_pids'] = [gps[:n_docs] for gps in gold_pids]
        start_time = time.time()
        response = requests.post(self.endpoint+'/retrieve', data=json.dumps(query), headers=self.headers)
        self.retrieval_time += time.time() - start_time
        try:
            rdocs = response.json()
        except json.decoder.JSONDecodeError:
            query['query_vectors'] = ''
            logger.error(f'Bad response: {response}\nRequest: {json.dumps(query)}')
            raise json.decoder.JSONDecodeError
        docs = rdocs['docs']
        doc_vectors = np.frombuffer(base64.decodebytes(rdocs['doc_vectors'].encode('ascii')), dtype=self.rest_dtype).\
            reshape(-1, n_docs_for_provenance, question_encoder_last_hidden_state.shape[-1])[:, 0:n_docs, :]
        retrieved_doc_embeds = torch.Tensor(doc_vectors.copy()).to(question_encoder_last_hidden_state)
        doc_scores = torch.bmm(
            question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
        ).squeeze(1)

        input_strings = self.tokenizer.question_encoder.batch_decode(input_ids, skip_special_tokens=True)
        context_input_ids, context_attention_mask = postprocess_docs(self.model.config, self.tokenizer,
                                                                     docs, input_strings, self.model.config.prefix,
                                                                     n_docs, return_tensors='pt')
        return context_input_ids.to(input_ids.device), context_attention_mask.to(input_ids.device), doc_scores, docs
