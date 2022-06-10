from dataloader.distloader_base import MultiFileLoader, DistBatchesBase
from dpr.biencoder_hypers import BiEncoderHypers
from util.line_corpus import jsonl_lines
import ujson as json
from typing import List
from transformers import PreTrainedTokenizerFast
import torch
import logging
import random

logger = logging.getLogger(__name__)


class BiEncoderInst:
    __slots__ = 'qry', 'pos_ctx', 'neg_ctx', 'pos_pids', 'ctx_pids'

    def __init__(self, qry, pos_ctx, neg_ctx, pos_pids, ctx_pids):
        self.qry = qry
        self.pos_ctx = pos_ctx
        self.neg_ctx = neg_ctx
        self.pos_pids = pos_pids
        self.ctx_pids = ctx_pids
        assert len(ctx_pids) == 2


class BiEncoderBatches(DistBatchesBase):
    def __init__(self, insts: List[BiEncoderInst], hypers: BiEncoderHypers,
                 qry_tokenizer: PreTrainedTokenizerFast, ctx_tokenizer: PreTrainedTokenizerFast):
        super().__init__(insts, hypers)
        self.qry_tokenizer = qry_tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.hypers = hypers
        self.batched_instances = []

    def create_conflict_free_batches(self, random):
        """
        Since we do batch negatives we want to ensure that the batch does not contain instances
        where the batch negatives contain positives.
        :return:
        """
        pushed_to_leftover = 0
        batch_neg_pids = set()  # the pids that our batch will call batch negatives (for any instance we might add to the batch)
        batch_pos_pids = set()  # the actual positives across all instances in our batch
        leftover_insts = []
        current_batch = []
        current_batch_leftover = []
        while len(self.insts) + len(leftover_insts) >= self.batch_size:
            # grab an instance
            if len(leftover_insts) > 0:
                inst = leftover_insts.pop()
            else:
                inst = self.insts.pop()
            # adding it to our batch should not violate our hard negative constraint:
            #  no positive or hard negative for one instance should be a positive for another instance
            if self.hypers.disable_confict_free_batches or \
                (all([pp not in batch_neg_pids for pp in inst.pos_pids]) and
                    all([np not in batch_pos_pids for np in inst.ctx_pids])):
                current_batch.append(inst)
                for cp in inst.ctx_pids:
                    batch_neg_pids.add(cp)
                for pp in inst.pos_pids:
                    batch_pos_pids.add(pp)
            else:
                current_batch_leftover.append(inst)  # this instance can't go in the current batch
                pushed_to_leftover += 1
            if len(current_batch) == self.batch_size:
                self.batched_instances.append(current_batch)
                leftover_insts.extend(current_batch_leftover)
                random.shuffle(leftover_insts)
                current_batch_leftover = []
                current_batch = []
                batch_neg_pids = set()
                batch_pos_pids = set()

        logger.warning(f'out of {len(self.batched_instances)} batches of size {self.batch_size}, '
                       f'pushed {pushed_to_leftover} out of batch due to conflict, '
                       f'{len(current_batch_leftover)} pushed out completely')
        if len(current_batch_leftover) > 2 * self.batch_size:
            logger.error(f'So many can not be batched! {len(current_batch_leftover)} unbatched!')
        self.insts = None  # no longer use insts, only batched_instances

    def post_init(self, *, batch_size, displayer=None, uneven_batches=False, random=None):
        self.batch_size = batch_size
        assert not uneven_batches
        assert random is not None
        random.shuffle(self.insts)
        self.create_conflict_free_batches(random) # this is why we override post_init and __getitem__
        self.num_batches = len(self.batched_instances)
        self.displayer = displayer
        self.uneven_batches = uneven_batches
        if self.hypers.world_size != 1:
            self._distributed_min()

    def __getitem__(self, index):
        if index >= self.num_batches:
            raise IndexError
        batch_insts = self.batched_instances[index]
        batch = self.make_batch(index, batch_insts)
        if index == 0 and self.displayer is not None:
            self.displayer(batch)
        return batch

    def make_batch(self, index, insts: List[BiEncoderInst]):
        ctx_titles = [title for i in insts for title in [i.pos_ctx[0], i.neg_ctx[0]]]
        ctx_texts = [text for i in insts for text in [i.pos_ctx[1], i.neg_ctx[1]]]
        # if index == 0:
        #     logger.info(f'titles = {ctx_titles}\ntexts = {ctx_texts}')
        qrys = [i.qry for i in insts]
        ctxs_tensors = self.ctx_tokenizer(ctx_titles, ctx_texts, max_length=self.hypers.seq_len_c,
                                          truncation=True, padding="longest", return_tensors="pt")
        qrys_tensors = self.qry_tokenizer(qrys, max_length=self.hypers.seq_len_q,
                                          truncation=True, padding="longest", return_tensors="pt")
        positive_indices = torch.arange(len(insts), dtype=torch.long) * 2
        assert qrys_tensors['input_ids'].shape[0] * 2 == ctxs_tensors['input_ids'].shape[0]
        return qrys_tensors['input_ids'], qrys_tensors['attention_mask'], \
               ctxs_tensors['input_ids'], ctxs_tensors['attention_mask'], \
               positive_indices


class BiEncoderLoader(MultiFileLoader):
    def __init__(self, hypers: BiEncoderHypers, per_gpu_batch_size: int, qry_tokenizer, ctx_tokenizer, data_dir,
                 positive_pid_file, *, files_per_dataloader=1, checkpoint_info=None):
        super().__init__(hypers, per_gpu_batch_size, data_dir,
                         checkpoint_info=checkpoint_info, files_per_dataloader=files_per_dataloader)
        self.qry_tokenizer = qry_tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.id2pos_pids = dict()
        for line in jsonl_lines(positive_pid_file):
            jobj = json.loads(line)
            self.id2pos_pids[jobj['id']] = jobj['positive_pids']

    def batch_dict(self, batch):
        """
        :param batch: input_ids_q, attention_mask_q, input_ids_c, attention_mask_c, positive_indices
        :return:
        """
        batch = tuple(t.to(self.hypers.device) for t in batch)
        return {'input_ids_q': batch[0], 'attention_mask_q': batch[1],
                'input_ids_c': batch[2], 'attention_mask_c': batch[3],
                'positive_indices': batch[4]}

    def display_batch(self, batch):
        input_ids_q = batch[0]
        input_ids_c = batch[2]
        positive_indices = batch[4]
        logger.info(f'{input_ids_q.shape} queries and {input_ids_c.shape} contexts\n{positive_indices}')
        qndx = random.randint(0, input_ids_q.shape[0]-1)
        logger.info(f'   query: {self.qry_tokenizer.decode(input_ids_q[qndx])}')
        logger.info(f' positve: {self.ctx_tokenizer.decode(input_ids_c[positive_indices[qndx]])}')
        logger.info(f'negative: {self.ctx_tokenizer.decode(input_ids_c[1+positive_indices[qndx]])}')

    def _one_load(self, lines):
        insts = []
        for line in lines:
            jobj = json.loads(line)
            qry = jobj['query']
            positive = jobj['positive']['title'], jobj['positive']['text']
            negs = jobj['negatives']
            if len(negs) == 0:
                logger.warning(f'bad instance! {len(negs)} negatives')
                continue
            if self.hypers.sample_negative_from_top_k > 0:
                neg_ndx = random.randint(0, min(len(negs), self.hypers.sample_negative_from_top_k)-1)
            else:
                neg_ndx = 0
            hard_neg = negs[neg_ndx]['title'], negs[neg_ndx]['text']
            ctx_pids = [jobj['positive']['pid'], negs[neg_ndx]['pid']]
            pos_pids = self.id2pos_pids[jobj['id']]
            assert len(positive) == 2
            assert len(hard_neg) == 2
            insts.append(BiEncoderInst(qry, positive, hard_neg, pos_pids, ctx_pids))
        return BiEncoderBatches(insts, self.hypers, self.qry_tokenizer, self.ctx_tokenizer)
