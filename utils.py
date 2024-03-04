import os
import random
import string
import re
import warnings
from collections import Counter
from typing import List
import shutil
import sys
from datetime import datetime
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import spacy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from hydra.utils import to_absolute_path
import wandb
from transformers import AutoModelForCausalLM

warnings.filterwarnings("ignore", message="UserWarning: Passing `max_length` to BeamSearchScorer is "
                                          "deprecated and has no effect. `max_length` should be passed "
                                          "directly to `beam_search(...)")


num_layers = {
    'gpt2': 12,
    'gpt2-large': 36,
    'gpt2-xl': 48,
    'distilgpt2': 6,
}


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""

    def __init__(self, fn, cfg, ask=True, main_process=True, use_wandb=False, wandb_name=None,
                 log_path=None):
        self.main_process = main_process
        self.log_path = './logs/' if log_path is None else log_path
        self.logdir = None
        self.cfg = cfg
        self.use_wandb = use_wandb

        if self.main_process:
            logdir = self.log_path + fn

            os.makedirs(os.path.join(logdir, 'sample_weights'), exist_ok=True)
            os.makedirs(os.path.join(logdir, 'checkpoints'), exist_ok=True)

            self.set_dir(logdir)

            if self.use_wandb:
                wandb.login(key=cfg.wandb_key)
                wandb.config = OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                )
                wandb.init(project=cfg.wandb_project, name=wandb_name, dir=logdir,
                           entity=cfg.wandb_entity, settings=wandb.Settings(start_method='fork'))

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')
        with open(os.path.join(logdir, 'config.yaml'), 'w+') as fp:
            OmegaConf.save(config=self.cfg, f=fp.name)

    def close_writer(self):
        if self.main_process and self.use_wandb:
            wandb.finish()

    def log(self, string):
        if self.main_process:
            self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
            self.log_file.flush()

            print('[%s] %s' % (datetime.now(), string))
            sys.stdout.flush()

    def log_dirname(self, string):
        if self.main_process:
            self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
            self.log_file.flush()

            print('%s (%s)' % (string, self.logdir))
            sys.stdout.flush()

    def wandb_log(self, log_dict, commit=None):
        if self.main_process and self.use_wandb:
            wandb.log(log_dict, commit=commit)


def decode_to_clean_text(tokenizer, ids):
    gen_text = tokenizer.batch_decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return list(map(str.strip, gen_text))


def clean_up(text):
    text = text.replace('<pad>', '')
    text = text.replace('</s>', '')
    text = text.replace(".", '')
    text = text.replace(',', '')
    text = text.replace("'", '')
    text = text.replace('"', '')
    return text


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, ground_truth, match_length=False):
    norm_pred = normalize_answer(prediction)
    norm_truth = normalize_answer(ground_truth)
    if not match_length:
        norm_pred = norm_pred[:len(norm_truth)]
    return norm_pred == norm_truth


# taken from squad codebase
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()

    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def set_random_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def shuffle_groups(df, group_col):
    """
    Shuffles the order of groups in a Pandas DataFrame without shuffling the order of items within each group.

    Parameters:
    - df: the input DataFrame
    - group_col: the name of the column containing the groups to be shuffled

    Returns:
    - a shuffled copy of the input DataFrame
    """
    # Get a list of unique groups
    groups = df[group_col].unique()

    # Shuffle the list of groups
    np.random.shuffle(groups)

    # Define a sorting key that sorts by the shuffled order of groups
    def sort_key(row):
        return np.argwhere(groups == row[group_col])[0][0]

    df['temp'] = df.apply(sort_key, axis=1)
    shuffled_df = df.sort_values('temp', kind='stable').drop('temp', axis=1).reset_index(drop=True)
    return shuffled_df


def return_k_unique(df, k, column):
    if k >= len(df[column].unique()):
        return df
    else:
        values_to_keep = df[column].unique()[:k]
        return df[df.apply(lambda x: x[column] in values_to_keep, axis=1)]


def cycle(loader):
    while True:
        for x in loader:
            yield x


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def metric_synchronize_between_processes(metrics, accelerator=None):
    if accelerator is not None:
        for k, v in metrics.items():
            t = torch.tensor([v], dtype=torch.float64, device=accelerator.device)
            gathered_items = accelerator.gather_for_metrics(t)
            metrics[k] = gathered_items.mean().item()
        accelerator.wait_for_everyone()
    else:
        if is_dist_avail_and_initialized():
            for k, v in metrics.items():
                t = torch.tensor([v], dtype=torch.float64, device='cuda')
                dist.barrier()
                dist.all_reduce(t)
                t /= dist.get_world_size()
                t = t.tolist()
                metrics[k] = t[0]


def update_path(cfg):
    if 'data_dir' in cfg: cfg.data_dir = to_absolute_path(cfg.data_dir)
    if 'test_path' in cfg: cfg.test_path = to_absolute_path(cfg.test_path)


def logging_path_check(cfg):
    from train import setup as train_setup
    _, fname, _ = train_setup(cfg.mode, cfg)
    log_path = './logs/' if cfg.log_path is None else cfg.log_path
    os.makedirs(log_path, exist_ok=True)
    logdir = log_path + fname
    os.makedirs(logdir, exist_ok=True)
    if len(os.listdir(logdir)) != 0 and cfg.resume_path is None:
        if cfg.context_size is None:
            ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                        "Will you proceed [y/N]? ")
            if ans in ['y', 'Y']:
                shutil.rmtree(logdir)
            else:
                exit(1)
        else:
            shutil.rmtree(logdir)


# Function to create a tqdm progress bar for distributed training
def tqdm_distributed(main_process, iterator, *args, **kwargs):
    if main_process:
        return tqdm(iterator, *args, **kwargs)
    else:
        return iterator  # No progress bar for non-main processes
