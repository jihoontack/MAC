from datetime import datetime


def setup(mode, cfg, date=''):
    if not cfg.no_date:
        date = f'{datetime.today().strftime("%y%m%d")}_'

    reset = 'noreset' if cfg.no_reset else ''
    fname = f"{cfg.dataset}/{cfg.base_model}/{date}{mode}{reset}" \
            f"_bs{int(cfg.update_batch_size*cfg.grad_acc_steps)}_{cfg.optimizer}_lr{cfg.lr_schedule}_outlr{cfg.outer_lr}"
    wandb_name = f"{date}{mode}{reset}_{cfg.dataset}_{cfg.base_model}" \
                 f"_bs{int(cfg.update_batch_size*cfg.grad_acc_steps)}_{cfg.optimizer}_lr{cfg.lr_schedule}_outlr{cfg.outer_lr}"

    if cfg.use_pretrained:
        fname += '_PRETRAIN'
        wandb_name += '_PRETRAIN'

    if mode == 'amortize_encdec':
        from train.train_func.amortize_encdec import train_epoch
        fname += f'_amort{cfg.pretrained_model_amort}_tokens{cfg.num_virtual_tokens}_Ltok{cfg.layer_num_virtual_tokens}'
        wandb_name += f'_amort{cfg.pretrained_model_amort}_tokens{cfg.num_virtual_tokens}_Ltok{cfg.layer_num_virtual_tokens}'
        if not cfg.no_aggregate:
            fname += f'_CrossAttn{cfg.num_cross_attention_blocks}'
            wandb_name += f'_CrossAttn{cfg.num_cross_attention_blocks}'
        if cfg.question_model_amort is not None:
            fname += f'_qmodel{cfg.question_model_amort}'
            wandb_name += f'_qmodel{cfg.question_model_amort}'
        if cfg.no_aggregate:
            fname += '_no_aggregate'
            wandb_name += '_no_aggregate'
        if cfg.dropout_p != 0.:
            fname += f'_Cattdrop{cfg.dropout_p}'
            wandb_name += f'_Cattdrop{cfg.dropout_p}'
        if cfg.gpt_drop:
            fname += '_gptdrop'
            wandb_name += '_gptdrop'
        if cfg.lift_ratio != 1.:  # typo: it is LITE not LIFT (will update later)
            fname += f'_lift{cfg.lift_ratio}'
            wandb_name += f'_lift{cfg.lift_ratio}'
        if cfg.mixed_precision is not None:
            fname += f'_mp{cfg.mixed_precision}'
            wandb_name += f'_mp{cfg.mixed_precision}'
        if cfg.quant_type is not None:
            fname += f'_quant{cfg.quant_type}'
            wandb_name += f'_quant{cfg.quant_type}'
    else:
        raise NotImplementedError()

    fname += f'_seed_{cfg.seed}'
    if cfg.suffix is not None:
        fname += f'_{cfg.suffix}'
        wandb_name += f'_{cfg.suffix}'

    return train_epoch, fname, wandb_name
