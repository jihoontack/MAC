import os

import hydra
import random
import torch
from transformers import AutoTokenizer, LlamaTokenizer

from common.dataset import get_eval_dataloader
from evals import setup
from evals.qa_utils import context_summarization
from models import get_base_model, get_amortize_encdec_model
from utils import set_random_seed


def extract_eval_batch(dataloader, num_steps):
    eval_batch = []
    for i, batch in enumerate(dataloader):
        eval_batch.append(batch)
    # shuffle list
    random.shuffle(eval_batch)
    return eval_batch[:num_steps]


@hydra.main(config_path='conf', config_name='config_eval')
def main(cfg):
    eval_fns = setup(cfg)
    cfg.rank = 0
    set_random_seed(cfg.seed)

    """ load base llm """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_lm = get_base_model(cfg)
    if cfg.base_model not in ["Llama2_7b"]:
        base_lm.to(device)

    """ define dataset, data loader, and tokenizer """
    if cfg.base_model == 'Llama2_7b':
        tokenizer = LlamaTokenizer.from_pretrained(cfg.llama_cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, cache_dir=cfg.CACHE_DIR)
    tokenizer_amort = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name_amort, cache_dir=cfg.CACHE_DIR, model_max_length=1024,
    ) if 'amort' in cfg.mode_eval else None
    train_dataloader, test_dataloader = get_eval_dataloader(cfg, tokenizer, tokenizer_amort)

    """ get weight model for camels / prompt model for ours """
    kwargs = {}
    kwargs_eval = {}
    amort_model = get_amortize_encdec_model(cfg, base_lm, tokenizer=tokenizer).to(device)
    amort_model.eval()
    kwargs['amort_model'] = amort_model
    kwargs_eval['amort_model'] = amort_model
    adapt_func = context_summarization
    context_summary_bank = adapt_func(train_dataloader, **kwargs)

    print('evaluating final model')
    base_lm.eval()
    for mode, eval_fn in eval_fns.items():
        eval_fn(cfg, test_dataloader, os.path.join(cfg.log_dir, f'final_{mode}_{cfg.suffix}.csv'),
                model=base_lm, tokenizer=tokenizer, context_summary_bank=context_summary_bank, **kwargs_eval)


if __name__ == "__main__":
    main()
