import numpy as np
from transformers import get_scheduler

from common.optim import get_optimizer
from models import get_amortize_encdec_model


def trainer(cfg, train_func, base_lm, tokenizer, tokenizer_amort,
            train_loader, val_loader, val_gen_loader, logger, accelerator):
    kwargs = {}

    """ init weight model or prompt """
    cfg.best_val_loss = np.inf
    cfg.best_em = 0.
    cfg.best_f1 = 0.

    """ get weight model for camels """
    amort_model = get_amortize_encdec_model(cfg, base_lm, tokenizer=tokenizer, tokenizer_amort=tokenizer_amort)
    amort_optimizer = get_optimizer(cfg, amort_model, accelerator)

    logger.log(amort_model)  # log prompt model
    train_steps = len(train_loader) * cfg.n_epochs // cfg.grad_acc_steps
    scheduler = get_scheduler(cfg.lr_schedule, amort_optimizer, int(train_steps * cfg.warmup_ratio), train_steps)

    amort_model, amort_optimizer, scheduler, train_loader, val_loader, val_gen_loader = accelerator.prepare(
        amort_model, amort_optimizer, scheduler, train_loader, val_loader, val_gen_loader
    )
    kwargs['scheduler'] = scheduler
    kwargs['amort_model'] = amort_model
    kwargs['amort_optimizer'] = amort_optimizer
    kwargs['accelerator'] = accelerator

    """ training start """
    logger.log_dirname(f"Start training")

    for i_epoch in range(0, cfg.n_epochs):
        logger.log_dirname(f'Starting training on epoch {i_epoch}')
        train_func(cfg, i_epoch, base_lm, train_loader, val_loader, val_gen_loader, logger, **kwargs)
