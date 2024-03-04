from omegaconf import OmegaConf
import hydra

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import AutoTokenizer, LlamaTokenizer

from common.dataset import get_dataloader
from train.trainer import trainer
from models import get_base_model
from utils import Logger, set_random_seed, update_path, logging_path_check


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def main(cfg):
    update_path(cfg)
    cfg.distributed = torch.cuda.device_count() > 1
    cfg.world_size = torch.cuda.device_count()

    """ Use huggingface accelerator (automatically use distributed) """
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_acc_steps,
        mixed_precision=cfg.mixed_precision,
        kwargs_handlers=[ddp_kwargs]
    )
    main_process = accelerator.is_main_process
    if main_process: logging_path_check(cfg)
    accelerator.wait_for_everyone()

    """ fixing randomness """
    set_random_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    """ define dataset, data loader, and tokenizer """
    if cfg.base_model == 'Llama2_7b':
        tokenizer = LlamaTokenizer.from_pretrained(cfg.llama_cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, cache_dir=cfg.CACHE_DIR)
    tokenizer_amort = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name_amort, cache_dir=cfg.CACHE_DIR, model_max_length=1024,
    ) if 'amort' in cfg.mode else None
    train_loader, val_loader, val_gen_loader = get_dataloader(
        cfg, tokenizer=tokenizer, tokenizer_amort=tokenizer_amort
    )

    """ Initialize model, optimizer """
    base_lm = get_base_model(cfg, accelerator)

    """ define train and test type """
    from train import setup as train_setup
    train_func, fname, wandb_name = train_setup(cfg.mode, cfg)

    """ define logger """
    logger = Logger(fname, cfg, ask=cfg.resume_path is None, main_process=main_process,
                    use_wandb=cfg.wandb_log and not cfg.no_eval, wandb_name=wandb_name, log_path=cfg.log_path)
    logger.log(OmegaConf.to_yaml(cfg))  # log config

    """ train """
    trainer(cfg, train_func, base_lm, tokenizer, tokenizer_amort,
            train_loader, val_loader, val_gen_loader, logger, accelerator)

    """ close tensorboard """
    logger.close_writer()


if __name__ == "__main__":
    main()
