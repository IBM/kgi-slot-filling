from util.line_corpus import read_lines, write_open, jsonl_files
import ujson as json
import logging
import os
from util.args_help import fill_from_args
import torch
from typing import List
import numpy as np
from dpr.simple_mmap_dataset import gzip_str
from dpr.faiss_index import build_index, IndexOptions
import base64
from util.reporting import Reporting
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
)

logger = logging.getLogger(__name__)


class Options(IndexOptions):
    def __init__(self):
        super().__init__()
        self.rag_model_name = 'facebook/rag-token-nq'
        self.dpr_ctx_encoder_model_name = 'facebook/dpr-ctx_encoder-multiset-base'
        self.dpr_ctx_encoder_path = ''
        self.embed = '1of1'
        self.sharded_index = False
        self.corpus = ''
        self.output_dir = ''  # the output_dir will have the passages dataset and the hnsw_index.faiss
        self.batch_size = 16
        self.__required_args__ = ['output_dir']


opts = Options()
fill_from_args(opts)

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(opts.output_dir, exist_ok=True)


def embed(doc_batch: List, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> np.ndarray:
    documents = {"title": [doci['title'] for doci in doc_batch], 'text': [doci['text'] for doci in doc_batch]}
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return embeddings.detach().cpu().to(dtype=torch.float16).numpy()


def write(cur_offset, offsets, passage_file, doc_batch, embeddings):
    assert len(doc_batch) == embeddings.shape[0]
    assert len(embeddings.shape) == 2
    for di, doc in enumerate(doc_batch):
        doc['vector'] = base64.b64encode(embeddings[di].astype(np.float16)).decode('ascii')
        jstr_gz = gzip_str(json.dumps(doc))
        offsets.append(cur_offset)
        passage_file.write(jstr_gz)
        cur_offset += len(jstr_gz)
    return cur_offset


embed_num, embed_count = [int(n.strip()) for n in opts.embed.split('of')]
assert 1 <= embed_num <= embed_count

# And compute the embeddings
ctx_encoder = DPRContextEncoder.from_pretrained(opts.dpr_ctx_encoder_path if opts.dpr_ctx_encoder_path
                                                else opts.dpr_ctx_encoder_model_name).to(device=device)
ctx_encoder.eval()
ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(opts.dpr_ctx_encoder_model_name)

offsets = []
cur_offset = 0
passages = write_open(os.path.join(opts.output_dir, f'passages_{embed_num}_of_{embed_count}.json.gz.records'), binary=True)

report = Reporting()
all_data_files = jsonl_files(opts.corpus)
if len(all_data_files) < embed_count:
    raise ValueError(f'too few files ({len(all_data_files)}) to split into {embed_count} parts')
data_files = all_data_files[embed_num-1::embed_count]
doc_batch = []
for line in read_lines(data_files):
    if report.is_time():
        print(f'On instance {report.check_count}, {report.check_count/report.elapsed_seconds()} instances per second')
    jobj = json.loads(line)
    doc_batch.append(jobj)
    if len(doc_batch) == opts.batch_size:
        embeddings = embed(doc_batch, ctx_encoder, ctx_tokenizer)
        cur_offset = write(cur_offset, offsets, passages, doc_batch, embeddings)
        doc_batch = []
if len(doc_batch) > 0:
    embeddings = embed(doc_batch, ctx_encoder, ctx_tokenizer)
    cur_offset = write(cur_offset, offsets, passages, doc_batch, embeddings)

offsets.append(cur_offset)  # just the length of the file
passages.close()

with write_open(os.path.join(opts.output_dir, f'offsets_{embed_num}_of_{embed_count}.npy'), binary=True) as f:
    np.save(f, np.array(offsets, dtype=np.int64), allow_pickle=False)

print(f'Wrote passages_{embed_num}_of_{embed_count}.json.gz.records in {report.elapsed_time_str()}')

if opts.sharded_index:
    build_index(os.path.join(opts.output_dir, f'passages_{embed_num}_of_{embed_count}.json.gz.records'),
                os.path.join(opts.output_dir, f'index_{embed_num}_of_{embed_count}.faiss'), opts)
elif embed_count == 1:
    build_index(opts.output_dir, os.path.join(opts.output_dir, 'index.faiss'), opts)
