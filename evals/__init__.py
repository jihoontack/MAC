import os
import functools
from collections import defaultdict
from omegaconf import OmegaConf

from hydra.utils import to_absolute_path

from evals.qa_utils import qa_eval


def setup(cfg):
    if isinstance(cfg.eval, str):
        cfg.eval = [cfg.eval]

    eval_fns = {}
    for eval_mode in cfg.eval:
        if eval_mode == 'em':
            top_k = 1 if eval_mode == 'em' else int(eval_mode[2:])
            num_beams = max(cfg.num_beams, top_k - top_k % cfg.num_beam_groups + cfg.num_beam_groups)
            if num_beams != cfg.num_beams:
                print(f'overwriting arguement number beam groups of {cfg.num_beams} to {num_beams}')
            eval_fns[eval_mode] = functools.partial(qa_eval, top_k=top_k, diversity_penalty=cfg.diversity_penalty,
                                                    num_beam_groups=cfg.num_beam_groups, num_beams=num_beams)
        else:
            print('unknown evaluation mode:', eval_mode)

    if cfg.load_path is not None:
        log_dir = os.path.dirname(to_absolute_path(cfg.load_path))   # this should be cur_path/checkpoint/best_model.pt
        log_dir = os.path.dirname(log_dir)   # this should be cur_path
        log_dir = os.path.join(log_dir, 'evals')
    else:
        log_dir = f'./logs/evals_uniform/{cfg.base_model}'

    os.makedirs(log_dir, exist_ok=True)

    set_eval_batch_params(cfg)
    set_qa_lt_batch_params(cfg)
    cfg.suffix = f'_{cfg.mode_eval}'
    cfg.log_dir = log_dir

    with open(os.path.join(log_dir, f'config_test{cfg.suffix}.yaml'), 'w+') as fp:
        OmegaConf.save(config=cfg, f=fp.name)

    if not cfg.use_pretrained: cfg.base_model_state_dict_path = None

    return eval_fns


def set_eval_batch_params(cfg):
    key = ''.join(cfg.base_model.split('/')[-1].split('-')[:-1])

    batch_defaults = defaultdict(lambda: 1, {'distilgpt2': 4, 'gpt2': 4, 'gpt2-medium': 2, 'gpt2-large': 1})
    grad_acc_defaults = defaultdict(lambda: 16, {'distilgpt2': 4, 'gpt2': 4, 'gpt2-medium': 8, 'gpt2-large': 16})

    if cfg.batch_size == -1: cfg.batch_size = batch_defaults[key]
    if cfg.grad_acc_steps == -1: cfg.grad_acc_steps = grad_acc_defaults[key]


def set_qa_lt_batch_params(cfg):
    key = cfg.base_model.split('qa_models/')[-1]
    if '-retrain' in key:
        key = key.split('-retrain')[0]

    batch_defaults = defaultdict(lambda: 2,
                                 {'distilgpt2': 32, 'gpt2': 16, 'gpt2-medium': 8, 'gpt2-large': 8, 'gpt2-xl': 2,
                                  'gpt-neo-1.3B': 2})

    if cfg.lt_batch_size == -1: cfg.lt_batch_size = batch_defaults[key]
    if cfg.lt_grad_acc_steps == -1: cfg.lt_grad_acc_steps = 64 // cfg.lt_batch_size
