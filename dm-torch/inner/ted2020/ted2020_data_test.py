import sys
import pdb
import pprint
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from argparse import Namespace
from fairseq import utils
from fairseq.tasks.translation import TranslationConfig, TranslationTask
import matplotlib.pyplot as plt

src_lang = 'en'
tgt_lang = 'zh'

config = Namespace(
    datadir="/root/autodl-tmp/dataset/ted2020/DATA/data-bin/ted2020",
    savedir="./checkpoints/rnn",
    source_lang=src_lang,
    target_lang=tgt_lang,

    # cpu threads when fetching & processing data.
    num_workers=2,
    # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
    max_tokens=8192,
    accum_steps=2,

    # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
    lr_factor=2.,
    lr_warmup=4000,

    # clipping gradient norm helps alleviate gradient exploding
    clip_norm=1.0,

    # maximum epochs for training
    max_epoch=15,
    start_epoch=1,

    # beam size for beam search
    beam=5,
    # generate sequences of maximum length ax + b, where x is the source length
    max_len_a=1.2,
    max_len_b=10,
    # when decoding, post process sentence by removing sentencepiece symbols and jieba tokenization.
    post_process="sentencepiece",

    # checkpoints
    keep_last_epochs=5,
    resume=None,  # if resume from checkpoint name (under config.savedir)

    # logging
    use_wandb=False,
)


def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
        # Set this to False to speed up. However, if set to False, changing max_tokens beyond
        # first call of this method has no effect.
    )
    return batch_iterator


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO",  # "DEBUG" "WARNING" "ERROR"
        stream=sys.stdout,
    )
    proj = "hw5.seq2seq"
    logger = logging.getLogger(proj)
    if config.use_wandb:
        import wandb

        wandb.init(project=proj, name=Path(config.savedir).stem, config=config)
    cuda_env = utils.CudaEnvironment()
    utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## setup task
    task_cfg = TranslationConfig(
        data=config.datadir,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        train_subset="train",
        required_seq_len_multiple=8,
        dataset_impl="mmap",
        upsample_primary=1,
    )
    task = TranslationTask.setup_task(task_cfg)

    logger.info("loading data for epoch 1")
    task.load_dataset(split="train", epoch=1, combine=True)  # combine if you have back-translation data.
    task.load_dataset(split="valid", epoch=1)

    sample = task.dataset("valid")[1]
    pprint.pprint(sample)
    pprint.pprint(
        "Source: " + \
        task.source_dictionary.string(
            sample['source'],
            config.post_process,
        )
    )
    pprint.pprint(
        "Target: " + \
        task.target_dictionary.string(
            sample['target'],
            config.post_process,
        )
    )

    seed = 2023

    demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=20, num_workers=1, cached=False)
    demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
    sample = next(demo_iter)
    pprint.pprint(sample)
