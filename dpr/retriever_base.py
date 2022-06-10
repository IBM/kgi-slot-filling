from util.reporting import Reporting
from torch_util.hypers_base import HypersBase
import torch
import logging
import torch.nn.functional as F
import random
from tabulate import tabulate

logger = logging.getLogger(__name__)


class RetrieverHypers(HypersBase):
    def __init__(self):
        super().__init__()
        # for reasonable values see the various params.json under
        #    https://github.com/peterliht/knowledge-distillation-pytorch
        self.kd_alpha = 0.9
        self.kd_temperature = 10.0
        self._is_initialized = False

    def copy_from_base_hypers(self, hypers: HypersBase, train_instances: int, per_gpu_batch_size_scale=1):
        self.cache_dir, self.output_dir = hypers.cache_dir, hypers.output_dir
        self.device, self.n_gpu = hypers.device, hypers.n_gpu
        self.global_rank, self.world_size, self.local_rank = hypers.global_rank, hypers.world_size, hypers.local_rank
        self.full_train_batch_size = hypers.full_train_batch_size
        self.gradient_accumulation_steps = hypers.gradient_accumulation_steps // per_gpu_batch_size_scale
        self.per_gpu_train_batch_size = hypers.per_gpu_train_batch_size * per_gpu_batch_size_scale
        self.train_instances = train_instances
        self.num_train_epochs = hypers.num_train_epochs
        self.warmup_instances = hypers.warmup_instances
        if hasattr(hypers, 'warmup_fraction'):
            self.warmup_fraction = hypers.warmup_fraction
        if self.per_gpu_train_batch_size * self.world_size * self.n_gpu * self.gradient_accumulation_steps != self.full_train_batch_size:
            logger.error(f'{self.per_gpu_train_batch_size} * {self.world_size} * {self.n_gpu} * {self.gradient_accumulation_steps} != {self.full_train_batch_size}')
            raise ValueError
        self._is_initialized = True


class InfoForBackward:
    def __init__(self, rng_state):
        self.rng_state = rng_state

    @staticmethod
    def get_rng_state():
        fwd_cpu_state = torch.get_rng_state()
        if torch.cuda.is_initialized():
            fwd_gpu_state = torch.cuda.get_rng_state()
        else:
            fwd_gpu_state = None
        return fwd_cpu_state, fwd_gpu_state

    def restore_rng_state(self):
        fwd_cpu_state, fwd_gpu_state = self.rng_state
        if fwd_gpu_state is not None:
            torch.cuda.set_rng_state(fwd_gpu_state)
        torch.set_rng_state(fwd_cpu_state)


class RetrieverBase:
    def __init__(self, hypers: RetrieverHypers):
        assert hypers._is_initialized
        self.reporting = Reporting()
        self.retrieve_hypers = hypers
        self.debug = False

    def add_kd_loss(self, logits, teacher_labels, hard_loss):
        assert logits.shape == teacher_labels.shape
        assert len(logits.shape) == 2
        if self.debug and random.random() < 1/200:
            logger.info(tabulate([['dpr']+logits[0].tolist(),
                                  ['rer']+teacher_labels[0].tolist(),
                                  ['dpr p'] + F.softmax(logits[0], dim=0).tolist(),
                                  ['rer p'] + F.softmax(teacher_labels[0], dim=0).tolist(),
                                  ],
                                 floatfmt='.3f'))
        T = self.retrieve_hypers.kd_temperature
        kd_loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits / T, dim=1),
                                                            F.softmax(teacher_labels / T, dim=1)) * (T * T)
        combined_loss = self.retrieve_hypers.kd_alpha * kd_loss + (1.0 - self.retrieve_hypers.kd_alpha) * hard_loss
        return combined_loss, hard_loss, kd_loss

    def track_retrieval_metrics(self, positive_pids, docs, *, display_prefix=''):
        if positive_pids is None:
            return
        assert len(positive_pids) == len(docs)
        pids = [dd['pid'] for dd in docs]
        for positives, retrieved in zip(positive_pids, pids):
            if not positives:
                continue
            hit1 = 0
            hit5 = 0
            inrecall = 0
            if retrieved[0] in positives:
                hit1 = 1.0
            if any([r in positives for r in retrieved[:5]]):
                hit5 = 1.0
            if any([r in positives for r in retrieved]):
                inrecall = 1.0
            self.reporting.moving_averages(hit_1=hit1, hit_5=hit5, in_recall=inrecall)
        if self.reporting.is_time():
            self.reporting.display(prefix=display_prefix)

    def retrieve_forward(self, queries, *, positive_pids=None):
        """

        :param queries: list of queries to retrieve documents for
        :param positive_pids: used for tracking and reporting on retrieval metrics
        :return: input for RAG: context_input_ids, context_attention_mask, doc_scores
          also docs and info-for-backward (when calling retrieve_backward)
        """
        context_input_ids, context_attention_mask, doc_scores, docs, ifb = None, None, None, None, None
        return context_input_ids, context_attention_mask, doc_scores, docs, ifb

    def retrieve_backward(self, ifb: InfoForBackward, *, doc_scores_grad=None, reranker_logits=None, target_mask=None):
        """
        At least one of doc_scores_grad, reranker_logits or target_mask should be provided
        :param ifb: the info-for-backward returned by retrieve_forward
        :param doc_scores_grad: Basic GCP gradients for the doc_scores returned by retrieve_forward
        :param reranker_logits: For KD training the query encoder
        :param target_mask: Ground truth for correct provenance
        :return: nothing
        """
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def cleanup_corpus_server(self):
        raise NotImplementedError
