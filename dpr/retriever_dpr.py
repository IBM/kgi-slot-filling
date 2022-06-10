from torch_util.transformer_optimize import TransformerOptimize
from dpr.retriever_base import RetrieverBase, RetrieverHypers, InfoForBackward
from generation.rag_util import prepare_seq2seq_batch
import torch
from corpus.corpus_client import CorpusClient, postprocess_docs
import numpy as np
import base64
from transformers import RagTokenizer, RagTokenForGeneration, DPRQuestionEncoder, RagConfig
import logging
import torch.nn.functional as F
import os
import signal
from typing import List

logger = logging.getLogger(__name__)


class DPRHypers(RetrieverHypers):
    def __init__(self):
        super().__init__()
        self.rag_model_path = ''
        self.qry_encoder_path = ''
        self.n_docs = 20
        self.corpus_endpoint = ''
        self.debug = False
        self.max_seq_length = 512
        self.space_before_context = False
        self.__required_args__ = ['corpus_endpoint']


class DPRInfoForBackward(InfoForBackward):
    def __init__(self, retrieved_doc_embeds, question_encoder_last_hidden_state, input_ids, attention_mask, rng_state):
        super().__init__(rng_state)
        self.retrieved_doc_embeds = retrieved_doc_embeds
        self.question_encoder_last_hidden_state = question_encoder_last_hidden_state
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.rdocs = None  # NOTE: only populated for debug mode


class RetrieverDPR(RetrieverBase):
    def __init__(self, hypers: DPRHypers, *, apply_mode=False):
        """
        :param hypers:
        """
        super().__init__(hypers)
        self.hypers = hypers
        self.tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq')
        # support loading from either a rag_model_path or qry_encoder_path
        if hypers.qry_encoder_path:
            assert not hypers.rag_model_path
            qencoder = DPRQuestionEncoder.from_pretrained(hypers.qry_encoder_path)
        elif hypers.rag_model_path:
            model = RagTokenForGeneration.from_pretrained(hypers.rag_model_path)
            qencoder = model.question_encoder
        else:
            raise ValueError('must supply either qry_encoder_path or rag_model_path')

        qencoder = qencoder.to(hypers.device)
        if apply_mode:
            self.optimizer = None
            self.model = qencoder
            self.model.eval()
        else:
            qencoder.train()
            self.optimizer = TransformerOptimize(self.hypers,
                                                 self.hypers.num_train_epochs * self.hypers.train_instances,
                                                 qencoder)
            self.model = self.optimizer.model

        self._server_pid = CorpusClient.ensure_server(hypers)
        self.rest_retriever = CorpusClient(hypers.corpus_endpoint, None, None)
        self.config = RagConfig.from_pretrained('facebook/rag-token-nq')
        if self.hypers.space_before_context:
            if self.config.prefix is not None and self.config.prefix != ' ':
                logger.warning(f'Previous config.prefix was {self.config.prefix}, now space')
            self.config.prefix = ' '

    def cleanup_corpus_server(self):
        if not hasattr(self, '_server_pid') or self._server_pid < 0:
            # no corpus server was started
            return
        if self.hypers.world_size > 1:
            torch.distributed.barrier()  # wait for everyone to finish before killing the corpus server
        if self._server_pid > 0:
            os.kill(self._server_pid, signal.SIGKILL)

    def retrieve_forward(self, queries, *, positive_pids=None, rdocs=None):
        """

        :param queries: list of queries to retrieve documents for
        :param positive_pids: used for tracking and reporting on retrieval metrics
        :return: input for RAG: context_input_ids, context_attention_mask, doc_scores
          also docs and info-for-backward (when calling retrieve_backward)
        """
        n_docs = self.hypers.n_docs
        input_dict = prepare_seq2seq_batch(self.tokenizer, queries, max_length=self.hypers.max_seq_length, return_tensors="pt")
        input_ids = input_dict['input_ids'].to(self.model.device)
        attention_mask = input_dict['attention_mask'].to(self.model.device)
        if self.optimizer is not None:
            self.optimizer.model.train()
        else:
            self.model.eval()
        # CONSIDER: we may wish to break this up into multiple calls and multiprocess the REST calls and the GPU encoding
        # NOTE: for KD, we don't actually care about redoing the rng exactly, since the gradient does not depend on it anyway
        rng_state = InfoForBackward.get_rng_state()
        with torch.no_grad():
            question_enc_outputs = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
            question_encoder_last_hidden_state = question_enc_outputs[0].detach()
        if rdocs is None:
            rdocs = self.rest_retriever._rest_call(question_encoder_last_hidden_state.cpu().numpy(), n_docs)
        else:
            logger.info(f'Using provided retrieved docs for debug mode')
        docs = rdocs['docs']
        self.track_retrieval_metrics(positive_pids, docs, display_prefix='DPR:')

        doc_vectors = np.frombuffer(base64.decodebytes(rdocs['doc_vectors'].encode('ascii')),
                                    dtype=self.rest_retriever.rest_dtype). \
            reshape(-1, n_docs, question_encoder_last_hidden_state.shape[-1])
        retrieved_doc_embeds = torch.Tensor(doc_vectors.copy()).to(question_encoder_last_hidden_state)
        doc_scores = torch.bmm(
            question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
        ).squeeze(1).detach()

        input_strings = self.tokenizer.question_encoder.batch_decode(input_ids, skip_special_tokens=True)
        context_input_ids, context_attention_mask = postprocess_docs(self.config, self.tokenizer,
                                                                     docs, input_strings, self.config.prefix,
                                                                     n_docs, return_tensors='pt')
        # need to return the retrieved_doc_embeds and the rng state
        # also return the question_encoder_last_hidden_state so we can verify we get the same thing in retrieve_backward
        ifb = DPRInfoForBackward(retrieved_doc_embeds, question_encoder_last_hidden_state,
                                 input_ids, attention_mask, rng_state)
        if self.hypers.debug:
            ifb.rdocs = rdocs
        return context_input_ids, context_attention_mask, doc_scores, docs, ifb

    def add_vectors(self, ifb: DPRInfoForBackward, pids: List[List[str]]):
        _, doc_vectors = self.rest_retriever.fetch(pids)
        doc_vectors = doc_vectors.to(ifb.retrieved_doc_embeds)
        assert len(ifb.retrieved_doc_embeds.shape) == len(doc_vectors.shape) == 3
        assert ifb.retrieved_doc_embeds.shape[0] == doc_vectors.shape[0]
        ifb.retrieved_doc_embeds = torch.cat((ifb.retrieved_doc_embeds, doc_vectors), dim=1)

    def retrieve_backward(self, ifb: DPRInfoForBackward, *, doc_scores_grad=None, reranker_logits=None, target_mask=None):
        """
        At least one of doc_scores_grad, reranker_logits or target_mask should be provided
        :param ifb: the info-for-backward returned by retrieve_forward
        :param doc_scores_grad: Basic GCP gradients for the doc_scores returned by retrieve_forward
        :param reranker_logits: For KD training the query encoder
        :param target_mask: Ground truth for correct provenance
        :return:
        """
        if doc_scores_grad is not None:
            self.optimizer.reporting.report_interval_secs = 10000000  # this loss is artificial, so don't report it
        self.optimizer.model.train()
        # save current rng state and restore forward rng state
        with torch.random.fork_rng(devices=[self.optimizer.hypers.device]):
            ifb.restore_rng_state()
            question_enc_outputs = self.optimizer.model(
                ifb.input_ids, attention_mask=ifb.attention_mask, return_dict=True
            )
            question_encoder_last_hidden_state = question_enc_outputs[0]
        # check that question_encoder_last_hidden_state_forward == question_encoder_last_hidden_state
        if self.debug:
            difference = ((question_encoder_last_hidden_state - ifb.question_encoder_last_hidden_state) ** 2).sum()
            if difference > 0.001:
                logger.error(
                    f'the question encoder last hidden state is very different from that computed earlier: {difference}')
                raise ValueError

        retrieved_doc_embeds = ifb.retrieved_doc_embeds.to(question_encoder_last_hidden_state)
        # TODO: support batch negatives here
        doc_scores = torch.bmm(
            question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
        ).squeeze(1)
        if doc_scores_grad is not None:
            grad_loss = torch.dot(doc_scores.reshape(-1), doc_scores_grad.reshape(-1))
            grad_norm_val = torch.linalg.norm(doc_scores_grad.reshape(-1)).item()
        else:
            grad_loss = 0
            grad_norm_val = 0
        if target_mask is not None:
            hard_loss = -(F.log_softmax(doc_scores, dim=1).reshape(-1)[[m for mb in target_mask for m in mb]].sum())
        else:
            hard_loss = 0
        if reranker_logits is not None and self.hypers.kd_alpha > 0.0:
            combined_loss, hard_loss, kd_loss = self.add_kd_loss(doc_scores, reranker_logits, hard_loss + grad_loss)
        else:
            combined_loss = hard_loss + grad_loss
            kd_loss = 0
        self.optimizer.step_loss(combined_loss, grad_norm=grad_norm_val, hard_loss=hard_loss, kd_loss=kd_loss)

    def save(self):
        if self.hypers.global_rank != 0:
            return
        model_to_save = (
            self.optimizer.model.module if hasattr(self.optimizer.model, "module") else self.optimizer.model)
        model_to_save.save_pretrained(os.path.join(self.hypers.output_dir, 'qry_encoder'))
