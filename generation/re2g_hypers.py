from dpr.retriever_dpr import DPRHypers
from dpr.retriever_dpr_bm25_reranker import RerankerHypers, BM25Hypers
from util.line_corpus import read_lines
from torch_util.hypers_base import HypersBase
import logging


logger = logging.getLogger(__name__)


class Re2gHypers(HypersBase):
    def __init__(self):
        super().__init__()
        self.dpr = DPRHypers()
        self.reranker = RerankerHypers()
        self.bm25 = BM25Hypers()
        self.kilt_data = ''
        self.max_target_length = 256
        # TODO: self.prefer_short_answer = False
        # control context length with dpr.max_seq_length
        self.model_name = 'facebook/rag-token-nq'
        self.model_path = ''
        self.no_leading_space = False
        self.positive_pids = ''
        self.per_gpu_train_batch_size = 1
        self.learning_rate = 3e-5
        self.reranker.learning_rate = 3e-5
        self.dpr.learning_rate = 3e-5
        self.gradient_accumulation_steps = -1
        self.num_instances = -1
        self.warmup_fraction = 0.1
        self.retrieve_batch_factor = 8
        self.debug = False
        self.prefer_extractive = False
        self.normalize_train_answer = False
        self.__required_args__ = ['dpr.corpus_endpoint', 'reranker.model_name_or_path',
                                  # 'bm25.jar', 'bm25.anserini_index',
                                  'output_dir', 'kilt_data']

    def _post_init(self):
        self._quiet_post_init = True
        super()._post_init()
        if self.num_instances <= 0:
            self.num_instances = sum(1 for _ in read_lines(self.kilt_data))
            logger.info(f'Counted num_instances = {self.num_instances}')
        self.per_gpu_train_batch_size = 1
        self.gradient_accumulation_steps = self.full_train_batch_size // self.world_size
        assert self.n_gpu == 1
        self.dpr.copy_from_base_hypers(self, self.num_instances, per_gpu_batch_size_scale=self.retrieve_batch_factor)
        self.reranker.copy_from_base_hypers(self, self.num_instances, per_gpu_batch_size_scale=1)
        logger.info(f'hypers:\n{self}')
        if self.debug:
            self.dpr.debug = True
            assert self.gradient_accumulation_steps == self.retrieve_batch_factor


class Re2gApplyHypers(Re2gHypers):
    """
    Include arguments to control generation for apply
    """
    def __init__(self):
        super().__init__()
        self.output = ''
        self.use_candidates = False
        self.num_return_sequences = 4
        self.num_beams = 4
        self.no_context = False
        self.limit = -1
        self.extractive_only = False
        self.num_instances = 1  # no training instances, just set to one
        self.__required_args__ = ['dpr.corpus_endpoint', 'reranker.model_name_or_path',
                                  # 'bm25.jar', 'bm25.anserini_index',
                                  'output', 'kilt_data']

