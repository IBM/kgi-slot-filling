from torch_util.transformer_optimize import TransformerOptimize
from dpr.biencoder_hypers import BiEncoderHypers
from dpr.biencoder_gcp import BiEncoder
from dpr.dataloader_biencoder import BiEncoderLoader
from transformers import (DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast)
import logging
import time
from util.line_corpus import jsonl_lines

logger = logging.getLogger(__name__)


class BiEncoderTrainArgs(BiEncoderHypers):
    def __init__(self):
        super().__init__()
        self.train_dir = ''
        self.positive_pids = ''
        self.num_instances = -1
        self.__required_args__ = ['train_dir', 'output_dir', 'positive_pids']

    def _post_init(self):
        super()._post_init()
        if self.num_instances <= 0:
            self.num_instances = sum(1 for _ in jsonl_lines(self.train_dir))
            logger.info(f'Counted num_instances = {self.num_instances}')


args = BiEncoderTrainArgs().fill_from_args()
if args.n_gpu != 1:
    logger.error('Multi-GPU training must be through torch.distributed')
    exit(1)
if args.world_size > 1 and 0 < args.encoder_gpu_train_limit:
    logger.error('Cannot support both distributed training and gradient checkpointing.  '
                 'Train with a single GPU or with --encoder_gpu_train_limit 0')
    exit(1)
qry_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained('facebook/dpr-question_encoder-multiset-base')
ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')
model = BiEncoder(args)
model.to(args.device)
model.train()
optimizer = TransformerOptimize(args, args.num_train_epochs * args.num_instances, model)
loader = BiEncoderLoader(args, args.per_gpu_train_batch_size, qry_tokenizer, ctx_tokenizer,
                         args.train_dir, args.positive_pids, files_per_dataloader=-1)
last_save_time = time.time()
args.set_seed()

while True:
    batches = loader.get_dataloader()
    if not optimizer.should_continue() or batches is None:
        break
    for batch in batches:
        loss, accuracy = optimizer.model(**loader.batch_dict(batch))
        optimizer.step_loss(loss, accuracy=accuracy)
        if not optimizer.should_continue():
            break
    if time.time()-last_save_time > 60*60:
        # save once an hour or after each file (whichever is less frequent)
        model_to_save = (optimizer.model.module if hasattr(optimizer.model, "module") else optimizer.model)
        logger.info(f'saving to {args.output_dir}')
        model_to_save.save(args.output_dir)
        last_save_time = time.time()

# save after running out of files or target num_instances
model_to_save = (optimizer.model.module if hasattr(optimizer.model, "module") else optimizer.model)
logger.info(f'saving to {args.output_dir}')
model_to_save.save(args.output_dir)
logger.info(f'Took {optimizer.reporting.elapsed_time_str()}')
