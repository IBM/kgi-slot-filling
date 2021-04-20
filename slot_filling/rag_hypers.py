import signal
from torch_util.hypers_base import HypersBase
import torch
import os
from slot_filling.corpus_client import CorpusClient
from transformers import RagTokenizer, RagSequenceForGeneration, RagTokenForGeneration


class RagHypers(HypersBase):
    def __init__(self):
        super().__init__()
        self.kilt_data = ''
        self.model_name = 'facebook/rag-token-nq'
        self.model_path = ''
        self.no_leading_space = False
        self.corpus_endpoint = ''
        self.n_docs = 5
        self.fold = ''  # {1-n}of{n}
        # only used for train
        self.num_instances = -1
        self.warmup_fraction = 0.1
        self.__required_args__ = ['kilt_data', 'num_instances', 'output_dir', 'corpus_endpoint']

    def _post_init(self):
        super()._post_init()
        # launch the server with a fork if corpus_endpoint is a directory
        self._server_pid = CorpusClient.ensure_server(self)

    def cleanup_corpus_server(self):
        if not hasattr(self, '_server_pid') or self._server_pid < 0:
            # no corpus server was started
            return
        if self.world_size > 1:
            torch.distributed.barrier()  # wait for everyone to finish before killing the corpus server
        if self._server_pid > 0:
            os.kill(self._server_pid, signal.SIGKILL)

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
