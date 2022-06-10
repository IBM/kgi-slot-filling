from transformers import (DPRQuestionEncoder, RagTokenizer, RagTokenForGeneration)
import torch
import logging
from torch_util.hypers_base import HypersBase
from corpus.corpus_client import CorpusClient
import os
import signal

logger = logging.getLogger(__name__)


class DPROptions(HypersBase):
    def __init__(self):
        super().__init__()
        self.corpus_endpoint = ''
        self.qry_encoder_path = ''
        self.rag_model_path = ''
        self.__required_args__ = ['corpus_endpoint']

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


def load_model_and_retriever(opts: DPROptions):
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
    return qencoder, rest_retriever
