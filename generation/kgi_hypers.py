from torch_util.hypers_base import HypersBase
from corpus.corpus_client import CorpusClient
from transformers import RagTokenizer, RagSequenceForGeneration, RagTokenForGeneration
from util.line_corpus import read_lines
import logging


logger = logging.getLogger(__name__)


class KgiHypers(HypersBase):
    def __init__(self):
        super().__init__()
        self.kilt_data = ''
        self.model_name = 'facebook/rag-token-nq'
        self.model_path = ''
        self.no_leading_space = False
        self.corpus_endpoint = ''
        self.port = 5001  # for starting our own corpus server
        self.n_docs = 5
        self.max_context_length = 512
        self.max_target_length = 512
        self.fold = ''  # {1-n}of{n}
        # only used for train
        self.num_instances = -1
        self.prefer_extractive = False
        self.normalize_train_answer = False
        self.warmup_fraction = 0.1
        self.__required_args__ = ['kilt_data', 'output_dir', 'corpus_endpoint']

    def _post_init(self):
        super()._post_init()
        # launch the server with a fork if corpus_endpoint is a directory
        self._server_pid = CorpusClient.ensure_server(self)
        if hasattr(self, 'num_instances') and self.num_instances == -1:
            self.num_instances = sum(1 for _ in read_lines(self.kilt_data))
            logger.info(f'Counted num_instances = {self.num_instances}')

    def cleanup_corpus_server(self):
        CorpusClient.cleanup_corpus_server(self)

    def get_tokenizer_and_model(self):
        # initialize the model and index
        tokenizer = RagTokenizer.from_pretrained(self.model_name)
        # rag_conf = RagConfig.from_pretrained(opts.model_name)
        if 'rag-token' in self.model_name:
            model = RagTokenForGeneration.from_pretrained(self.model_path if self.model_path else self.model_name)
        elif 'rag-sequence' in self.model_name:
            model = RagSequenceForGeneration.from_pretrained(self.model_path if self.model_path else self.model_name)
        else:
            raise AssertionError
        return tokenizer, model
