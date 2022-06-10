from transformers import (DPRQuestionEncoder, DPRContextEncoder,
                          DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast)
import torch
from torch_util.hypers_base import HypersBase
import os
import numpy as np
from typing import List
import ujson as json
from util.line_corpus import jsonl_lines, write_open
from eval.kilt.eval_downstream import evaluate
from eval.convert_for_kilt_eval import to_distinct_doc_ids
from util.reporting import Reporting


class Options(HypersBase):
    def __init__(self):
        super().__init__()
        self.initial_retrieval = ''
        self.kilt_data = ''
        self.dpr_path = ''
        # self.batch_size = 32
        self.output = ''
        self.__required_args__ = ['dpr_path', 'initial_retrieval', 'output']

    def get_tokenizer_and_model(self):
        qry_encoder = DPRQuestionEncoder.from_pretrained(os.path.join(self.dpr_path, 'qry_encoder')).to(self.device)
        qry_encoder.eval()
        ctx_encoder = DPRContextEncoder.from_pretrained(os.path.join(self.dpr_path, 'ctx_encoder')).to(self.device)
        ctx_encoder.eval()
        qry_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained('facebook/dpr-question_encoder-multiset-base')
        ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')
        return qry_tokenizer, qry_encoder, ctx_tokenizer, ctx_encoder


def ctx_embed(doc_batch: List[dict], ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> np.ndarray:
    documents = {"title": [doci['title'] for doci in doc_batch], 'text': [doci['text'] for doci in doc_batch]}
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    # FIXME: maybe attention mask here too
    with torch.no_grad():
        embeddings = ctx_encoder(input_ids.to(device=ctx_encoder.device), return_dict=True).pooler_output
    return embeddings.detach().cpu().to(dtype=torch.float16).numpy()


def qry_embed(qry_batch: List[str], qry_encoder: DPRQuestionEncoder, qry_tokenizer: DPRQuestionEncoderTokenizerFast) -> np.ndarray:
    inputs = qry_tokenizer(qry_batch, truncation=True, padding="longest", return_tensors="pt")  # max_length=self.hypers.seq_len_q,
    with torch.no_grad():
        embeddings = qry_encoder(inputs['input_ids'].to(device=qry_encoder.device),
                                 inputs['attention_mask'].to(device=qry_encoder.device), return_dict=True).pooler_output
    return embeddings.detach().cpu().to(dtype=torch.float16).numpy()


if __name__ == "__main__":
    opts = Options()
    opts.fill_from_args()
    report = Reporting()
    torch.set_grad_enabled(False)
    qry_tokenizer, qry_encoder, ctx_tokenizer, ctx_encoder = opts.get_tokenizer_and_model()
    missing_count = 0
    with write_open(opts.output) as f:
        for line in jsonl_lines(opts.initial_retrieval):
            jobj = json.loads(line)
            inst_id = jobj['id']
            query = jobj['input']
            passages = jobj['passages']
            pid2passage = {p['pid']: p for p in passages}
            # positive_pids = inst_id2pos_pids[inst_id]
            # target_mask = [p['pid'] in positive_pids for p in passages]
            # TODO: do some batching
            ctx_vecs = [ctx_embed([passage], ctx_encoder, ctx_tokenizer).reshape(-1) for passage in passages]
            qry_vec = qry_embed([query], qry_encoder, qry_tokenizer).reshape(-1)
            # now do dot products, create scored_pids
            scored_pids = [(p['pid'], np.dot(qry_vec, ctx_veci)) for p, ctx_veci in zip(passages, ctx_vecs)]
            # produce re-ranked output
            scored_pids.sort(key=lambda x: x[1], reverse=True)
            all_passages = [pid2passage[pid] for pid, _ in scored_pids]
            jobj['passages'] = all_passages
            wids = to_distinct_doc_ids([passage['pid'] for passage in all_passages])
            jobj['output'] = [{'answer': '', 'provenance': [{'wikipedia_id': wid} for wid in wids]}]
            f.write(json.dumps(jobj) + '\n')
            if report.is_time():
                print(f'{report.progress_str()}')
    print(f'Took {report.elapsed_time_str()} for {report.check_count}')
    if opts.kilt_data:
        evaluate(opts.kilt_data, opts.output)
