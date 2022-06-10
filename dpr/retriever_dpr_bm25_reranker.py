from torch_util.transformer_optimize import TransformerOptimize, LossHistory
from dpr.retriever_base import RetrieverBase, InfoForBackward, RetrieverHypers
from generation.rag_util import postprocess_docs
from dpr.retriever_dpr import DPRHypers, RetrieverDPR
from dpr.retriever_bm25 import BM25Hypers, RetrieverBM25
import torch
import logging
import torch.nn.functional as F
import os
from reranker.reranker_model import load_pretrained, save_transformer
from transformers import (BertConfig, BertTokenizer, BertForSequenceClassification)
import random

logger = logging.getLogger(__name__)


class RerankerHypers(RetrieverHypers):
    def __init__(self):
        super().__init__()
        self.model_name_or_path = ''
        self.max_seq_length = 128
        self.n_docs = 5
        self.n_docs_for_provenance = 20
        self.freeze_dpr = False
        self.kd_bm25_vectors = False
        self.__required_args__ = ['model_name_or_path']


class RerankerInfoForBackward(InfoForBackward):
    def __init__(self, dpr_ifb, reranker_doc_scores, topk_reranker_doc_scores, inputs_list, rng_state):
        super().__init__(rng_state)
        self.dpr_ifb = dpr_ifb
        self.reranker_doc_scores = reranker_doc_scores
        self.topk_reranker_doc_scores = topk_reranker_doc_scores
        self.inputs_list = inputs_list


class RetrieverDPRReranker(RetrieverBase):
    def __init__(self, hypers: RerankerHypers, dpr_hypers: DPRHypers, bm25_hypers: BM25Hypers, *, apply_mode=False):
        super().__init__(hypers)
        self.hypers = hypers
        assert self.hypers.per_gpu_train_batch_size == 1
        if bm25_hypers.anserini_index:
            self.bm25_retriever = RetrieverBM25(bm25_hypers)
        else:
            self.bm25_retriever = None
            logger.info(f'No BM25')
        self.dpr_retriever = RetrieverDPR(dpr_hypers, apply_mode=apply_mode)
        self.max_length_count = 0
        # load model and tokenizer
        hypers.do_lower_case = True
        model, self.tokenizer = load_pretrained(hypers, BertConfig, BertForSequenceClassification, BertTokenizer,
                                                num_labels=2)
        if apply_mode:
            self.model = model
            self.optimizer = None
            self.loss_history = None
        else:
            instances_to_train_over = hypers.train_instances * hypers.num_train_epochs
            self.optimizer = TransformerOptimize(hypers, instances_to_train_over, model)
            self.model = self.optimizer.model
            self.optimizer.model.zero_grad()
            self.loss_history = LossHistory(hypers.train_instances //
                                            (hypers.full_train_batch_size // hypers.gradient_accumulation_steps))

    def _one_instance(self, query, docsi):
        texts_b = [title + '\n\n' + text for title, text in zip(docsi['title'], docsi['text'])]
        texts_a = [query] * len(texts_b)
        max_seq_length = self.retrieve_hypers.max_seq_length
        inputs = self.tokenizer(
            texts_a, texts_b,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=max_seq_length,
            padding='longest',
            truncation=True)
        # track how often we truncate to max_seq_length
        if inputs['input_ids'].shape[1] == max_seq_length:
            self.max_length_count += 1
        inputs = {n: t.to(self.model.device) for n, t in inputs.items()}
        logits = F.log_softmax(self.model(**inputs)[0], dim=-1)[:, 1]  # log_softmax over the binary classification
        return inputs, logits

    def retrieve_forward(self, queries, *, positive_pids=None, rdocs=None):
        """

        :param queries: list of queries to retrieve documents for
        :param positive_pids: used for tracking and reporting on retrieval metrics
        :return: input for RAG: context_input_ids, context_attention_mask, doc_scores
          also docs and info-for-backward (when calling retrieve_backward)
        """
        # fetch from DPR
        _, _, _, docs_dpr, dpr_ifb = \
            self.dpr_retriever.retrieve_forward(queries, positive_pids=positive_pids, rdocs=rdocs)

        # now fetch from BM25
        if self.bm25_retriever is not None:
            title_text_dpr = [set([title + '\n\n' + text for title, text in zip(doc_dpr['title'], doc_dpr['text'])])
                              for doc_dpr in docs_dpr]
            _, docs_bm25 = self.bm25_retriever.retrieve_forward(queries, exclude_by_content=title_text_dpr)

            # stack these context_input_ids, attention_mask and docs
            assert len(docs_dpr) == len(docs_bm25)
            docs = [{'pid': dpr['pid'] + bm25['pid'], 'title': dpr['title'] + bm25['title'], 'text': dpr['text'] + bm25['text']}
                    for dpr, bm25 in zip(docs_dpr, docs_bm25)]
            initial_retrieval_n_docs = self.dpr_retriever.hypers.n_docs + self.bm25_retriever.hypers.n_docs
            # optionally also retrieve the vectors for these pids
            if self.hypers.kd_bm25_vectors and self.optimizer is not None:
                bm25_pids = [bm25['pid'] for bm25 in docs_bm25]
                self.dpr_retriever.add_vectors(dpr_ifb, bm25_pids)
        else:
            docs = docs_dpr
            initial_retrieval_n_docs = self.dpr_retriever.hypers.n_docs

        assert initial_retrieval_n_docs >= self.hypers.n_docs_for_provenance
        context_input_ids, context_attention_mask = postprocess_docs(self.dpr_retriever.config, self.dpr_retriever.tokenizer,
                                                                     docs, queries, self.dpr_retriever.config.prefix,
                                                                     initial_retrieval_n_docs,
                                                                     return_tensors='pt')
        # we must reshape these to have order 3: batch x initial_retrieval_n_docs x seq_length
        context_input_ids = context_input_ids.reshape(len(queries), initial_retrieval_n_docs, -1)
        context_attention_mask = context_attention_mask.reshape(len(queries), initial_retrieval_n_docs, -1)

        if self.optimizer is not None:
            self.model.train()
        else:
            self.model.eval()
        rng_state = InfoForBackward.get_rng_state()
        inputs_list = []
        reranker_doc_scores = []
        reranked_docs = []
        topk_context_input_ids = []
        topk_context_attention_mask = []
        topk_reranker_doc_scores = []
        with torch.no_grad():
            for bi, docsi in enumerate(docs):
                inputs, logits = self._one_instance(queries[bi], docsi)
                # NOTE: these should only be the scores for the DPR
                if self.hypers.kd_bm25_vectors:
                    reranker_doc_scores.append(logits)
                else:
                    reranker_doc_scores.append(logits[:self.dpr_retriever.hypers.n_docs])
                top_k_ndx = torch.topk(logits, k=self.hypers.n_docs).indices
                top_k_provenance_ndx = torch.topk(logits, k=self.hypers.n_docs_for_provenance).indices
                if top_k_ndx.shape[0] != self.hypers.n_docs:
                    logger.error(f'{top_k_ndx.shape} does not match requested n_docs {self.hypers.n_docs}\n'
                                 f'over logits shaped {logits.shape}')
                assert top_k_ndx.shape[0] == self.hypers.n_docs

                # NOTE: redo forward pass on just the top-k, record those inputs so we only do the backward for those
                if self.optimizer is not None:
                    topk_docsi = {'title': [docsi['title'][k] for k in top_k_ndx],
                                  'text':  [docsi['text'][k] for k in top_k_ndx]}
                    inputs, top_k_logits = self._one_instance(queries[bi], topk_docsi)
                    inputs_list.append(inputs)
                else:
                    top_k_logits = logits[top_k_ndx]

                # now take the top-k (reranker.n_docs) according to our re-ranker
                topk_reranker_doc_scores.append(top_k_logits)  # CONSIDER: maybe we should do another log_softmax here?
                reranked_docs.append({'pid':   [docsi['pid'][k] for k in top_k_provenance_ndx],
                                      'title': [docsi['title'][k] for k in top_k_provenance_ndx],
                                      'text':  [docsi['text'][k] for k in top_k_provenance_ndx]})
                topk_context_input_ids.append(context_input_ids[bi][top_k_ndx])
                topk_context_attention_mask.append(context_attention_mask[bi][top_k_ndx])

        topk_reranker_doc_scores = torch.stack(topk_reranker_doc_scores)
        rr_ifb = RerankerInfoForBackward(dpr_ifb,
                                         torch.stack(reranker_doc_scores), topk_reranker_doc_scores,
                                         inputs_list, rng_state)

        self.track_retrieval_metrics(positive_pids, reranked_docs, display_prefix='Reranker:')

        return torch.stack(topk_context_input_ids), torch.stack(topk_context_attention_mask), \
               topk_reranker_doc_scores, reranked_docs, rr_ifb

    def retrieve_backward(self, ifb: RerankerInfoForBackward, *, doc_scores_grad=None, reranker_logits=None, target_mask=None):
        """
        At least one of doc_scores_grad, reranker_logits or target_mask should be provided
        :param ifb: the info-for-backward returned by retrieve_forward
        :param doc_scores_grad: These are the gradients for topk_reranker_doc_scores
        :param reranker_logits: For KD training the query encoder
        :param target_mask: Ground truth for correct provenance
        :return:
        """
        # TODO: target_mask should be recorded in ifb if positive_pids are passed, for retrieve_backward we should pass use_target_mask
        if doc_scores_grad is not None:
            self.optimizer.reporting.report_interval_secs = 10000000  # this loss is artificial, so don't report it

        dpr_doc_scores_grad = None
        if not self.hypers.freeze_dpr:
            self.dpr_retriever.retrieve_backward(ifb.dpr_ifb,
                                                 doc_scores_grad=dpr_doc_scores_grad,
                                                 reranker_logits=ifb.reranker_doc_scores,
                                                 target_mask=None)

        self.optimizer.model.train()
        start_global_step = self.optimizer.global_step
        # save current rng state and restore forward rng state
        with torch.random.fork_rng(devices=[self.optimizer.hypers.device]):
            ifb.restore_rng_state()
            # loop over the batch in ifb, apply the doc_scores_grad
            for bi, inputs in enumerate(ifb.inputs_list):
                topk_logits = F.log_softmax(self.optimizer.model(**inputs)[0], dim=-1)[:, 1]
                if self.debug:
                    # check that topk_logits is the same as we computed in forward pass
                    difference = ((topk_logits - ifb.topk_reranker_doc_scores[bi]) ** 2).sum()
                    if difference > 0.001:
                        logger.error( f'on {bi}, the topk reranker logits are very different from that computed earlier: '
                                      f'{topk_logits} vs. {ifb.topk_reranker_doc_scores[bi]}\ndifference= {difference}')
                        raise ValueError
                    if random.random() < 1/200:
                        logger.info(f'Sample doc_scores_grad: {doc_scores_grad[bi].tolist()}')
                grad_loss = torch.dot(topk_logits, doc_scores_grad[bi])
                self.optimizer.step_loss(grad_loss)
                # verify we never take an optimizer step except at the last iteration of this loop
                if self.optimizer.global_step > start_global_step and bi < len(ifb.inputs_list)-1:
                    logger.error(f'broken gradient accumulation steps vs batch size')
                    raise ValueError
        # CONSIDER: if the global step increases, try re-doing the forward and looking at how the doc scores change

    def cleanup_corpus_server(self):
        self.dpr_retriever.cleanup_corpus_server()

    def save(self):
        self.dpr_retriever.save()
        if self.hypers.global_rank != 0:
            return
        logger.info(f'loss_history = {self.loss_history.loss_history}')
        logger.info(f'truncated to max length ({self.hypers.max_seq_length}) {self.max_length_count} times')
        save_transformer(self.hypers, self.optimizer.model, self.tokenizer,
                         save_dir=os.path.join(self.hypers.output_dir, 'reranker'))
