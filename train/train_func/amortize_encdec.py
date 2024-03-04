from collections import defaultdict

import numpy as np
import torch

from utils import metric_synchronize_between_processes, tqdm_distributed


def train_epoch(cfg, i_epoch, base_lm, train_loader, val_loader, val_gen_loader,
                logger, amort_model, amort_optimizer, accelerator, scheduler=None):

    metrics_dic = defaultdict(lambda: [])
    accelerator.wait_for_everyone()
    main_process = accelerator.is_main_process

    # training
    for i_step, batch in tqdm_distributed(
            main_process, enumerate(train_loader), desc='training_epoch', position=0, total=len(train_loader)
    ):
        # compute loss
        with accelerator.accumulate(amort_model):
            amort_model.train()
            outer_loss, metrics = amort_model(batch)
            accelerator.backward(outer_loss)

            if accelerator.sync_gradients:
                # clip gradient when using sync gradients
                grad_norm = accelerator.clip_grad_norm_(amort_model.parameters(), cfg.grad_clip_thresh)

                # log metrics when using sync gradients (i.e., actual gradient update)
                logger.wandb_log({'grad_norm': grad_norm}, commit=False)
                logger.wandb_log({'outer_lr': amort_optimizer.param_groups[0]['lr']}, commit=False)
                logger.wandb_log({'train': {f'{k}': np.mean(v) for k, v in metrics_dic.items()}})
                metrics_dic.clear()
            amort_optimizer.step()
            amort_optimizer.zero_grad()
            scheduler.step()

            metric_synchronize_between_processes(metrics, accelerator)  # sync metrics across processes
            for k, v in metrics.items():
                metrics_dic[f'[AGG]{k}'].append(v)

            del batch  # free memory
            accelerator.wait_for_everyone()

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # validation
    if not cfg.no_eval:
        logger.log(f'VALIDATION, epoch {i_epoch} end')

        amort_model.eval()
        amort_model.zero_grad()
        amort_optimizer.zero_grad()
        unwrapped_model = accelerator.unwrap_model(amort_model)

        with torch.no_grad():
            val_metrics, memory_bank = unwrapped_model.validate(val_loader, main_process=main_process)
            memory_bank = accelerator.gather_for_metrics(memory_bank)
            val_qa_metrics = unwrapped_model.validate_qa(
                val_gen_loader, context_bank=memory_bank, main_process=main_process
            )
            val_metrics.update(val_qa_metrics)
            del memory_bank

        metric_synchronize_between_processes(val_metrics, accelerator)  # sync metrics across processes

        logger.wandb_log({'val': val_metrics}, commit=False)
        if val_metrics['[AGG]qa_loss'] < cfg.best_val_loss:
            logger.log('Saving best model')
            cfg.best_val_loss = float(val_metrics['[AGG]qa_loss'])
            if cfg.quant_type is None:
                logger.log('Saving best qa loss model')
                unwrapped_model.save(i_epoch, log_dir=logger.logdir, file_name=f'best_val_loss.pt',
                                     main_process=main_process)

        if val_metrics['[AGG]em'] > cfg.best_em:
            cfg.best_em = float(val_metrics['[AGG]em'])
            if cfg.quant_type is None:
                logger.log('Saving best em model')
                unwrapped_model.save(i_epoch, log_dir=logger.logdir, file_name=f'best_em.pt',
                                     main_process=main_process)

        if val_metrics['[AGG]f1'] > cfg.best_f1:
            logger.log('Saving best f1 model')
            cfg.best_f1 = float(val_metrics['[AGG]f1'])
            unwrapped_model.save(i_epoch, log_dir=logger.logdir, file_name=f'best_f1.pt',
                                 main_process=main_process)

        amort_model.zero_grad()
        amort_optimizer.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.log_dirname(f'End of validation, back to training')

        logger.log('Saving last epoch model')
        unwrapped_model.save(i_epoch, log_dir=logger.logdir, file_name=f'last_epoch.pt',
                             main_process=main_process)
        accelerator.wait_for_everyone()
