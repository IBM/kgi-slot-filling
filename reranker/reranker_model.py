import logging
from torch_util.hypers_base import HypersBase
import os
import torch
from transformers import (
    PreTrainedTokenizer,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}


class RerankerHypers(HypersBase):
    """
    Arguments and hypers
    """
    def __init__(self):
        super().__init__()
        self.initial_retrieval = ''
        self.max_num_seq_pairs_per_device = 32
        self.full_train_batch_size = 32
        self.gradient_accumulation_steps = self.full_train_batch_size
        self.per_gpu_train_batch_size = 1
        self.num_train_epochs = 2
        self.train_instances = -1
        self.learning_rate = 3e-5
        self.max_seq_length = 128
        self.num_labels = 2
        self.fold = ''  # IofN
        self.add_all_positives = False
        self.doc_match_weight = 0.0
        self.model_type = 'bert'
        self.model_name_or_path = 'nboost/pt-bert-base-uncased-msmarco'

    def _post_init(self):
        super()._post_init()
        self.gradient_accumulation_steps = self.full_train_batch_size
        self.per_gpu_train_batch_size = 1
        assert self.world_size == 1


def load_tokenizer(hypers: HypersBase, tokenizer_class):
    tokenizer: PreTrainedTokenizer = tokenizer_class.from_pretrained(
        hypers.tokenizer_name if hypers.tokenizer_name else hypers.model_name_or_path,
        do_lower_case=hypers.do_lower_case,
        cache_dir=hypers.cache_dir if hypers.cache_dir else None
    )
    return tokenizer


def load_pretrained(hypers: HypersBase, config_class, model_class, tokenizer_class, **extra_model_args):
    # Load pretrained model and tokenizer
    if hypers.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    config = config_class.from_pretrained(
        hypers.config_name if hypers.config_name else hypers.model_name_or_path,
        cache_dir=hypers.cache_dir if hypers.cache_dir else None,
        **extra_model_args
    )
    tokenizer = load_tokenizer(hypers, tokenizer_class)
    model = model_class.from_pretrained(
        hypers.model_name_or_path,
        from_tf=bool(".ckpt" in hypers.model_name_or_path),
        config=config,
        cache_dir=hypers.cache_dir if hypers.cache_dir else None,
    )
    if hypers.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(hypers.device)
    return model, tokenizer


def load(hypers: RerankerHypers):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[hypers.model_type.lower()]
    resume_from = hypers.resume_from if hasattr(hypers, 'resume_from') else ''
    if not resume_from:
        model, tokenizer = load_pretrained(hypers, config_class, model_class, tokenizer_class,
                                           num_labels=hypers.num_labels)
    else:
        tokenizer = load_tokenizer(hypers, tokenizer_class)
        model = model_class.from_pretrained(resume_from)
        model.to(hypers.device)

    return model, tokenizer


def save_transformer(hypers: HypersBase, model, tokenizer, *, save_dir=None):
    if hypers.global_rank == 0:
        if save_dir is None:
            save_dir = hypers.output_dir
        # Create output directory if needed
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", save_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        torch.save(hypers, os.path.join(save_dir, "training_args.bin"))
        model_to_save.save_pretrained(save_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)

