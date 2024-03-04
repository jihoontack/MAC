import torch
from transformers.optimization import Adafactor
import bitsandbytes as bnb
import bitsandbytes.optim as bnb_optim

def get_optimizer(cfg, model, accelerator):
    learning_rate = cfg.outer_lr
    if cfg.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adam8bit':
        return bnb.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.995),
                              optim_bits=8, percentile_clipping=5)
    elif cfg.optimizer == 'pagedadamw':
        return bnb_optim.PagedAdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.995),
                                    weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adafactor':
        return Adafactor(model.parameters(), scale_parameter=False, relative_step=False,
                         warmup_init=False, lr=learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError()
