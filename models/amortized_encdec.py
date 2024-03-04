import os
import random
from typing import Optional
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from models.module.aggregate import Aggregator, MLP, PassPrompt
from models.module.self_attention import TokenSelfAttend
from utils import tqdm_distributed, decode_to_clean_text, exact_match, f1_score


def passprompt(input1, input2, prompts):
    return prompts


def log_perplexity(logits, coeffs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()


def _get_batch_size(input_ids: Optional[torch.Tensor], inputs_embeds: Optional[torch.Tensor]) -> int:
    """
    Get the batch size based on either input_ids or input_embeds
    Raises an ValueError if both are None.
    """
    if (input_ids is None) and (inputs_embeds is None):
        raise ValueError("You have to provide either input_ids or inputs_embeds")

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size


class AmortEncDecAggregateWrapper(nn.Module):
    def __init__(self, config, base_lm=None, enc_decoder=None, question_encoder=None):
        super().__init__()
        self.config = config
        self.base_lm = base_lm
        self._log_dir = config.log_dir if 'log_dir' in config else './logs'

        self.tau = config.tau if 'tau' in config else 1.

        self.base_model_torch_dtype = None
        self.token_dim = self.config.token_dim
        self.hidden_size = self.token_dim
        self.set_base_lm(base_lm)

        self.prompt_prob_max_training = 0.
        self.prompt_train_count = 0.
        self.prompt_prob_max_eval = 0.
        self.prompt_eval_count = 0.

        self.input_size = self.token_dim
        self.output_size = self.token_dim
        self.num_virtual_tokens = self.config.num_virtual_tokens
        self.num_actual_tokens = self.num_virtual_tokens
        self.layer_num_virtual_tokens = self.config.layer_num_virtual_tokens
        self.num_actual_tokens = self.layer_num_virtual_tokens * self.base_num_layers * 2
        self.token_SA = TokenSelfAttend(self.token_dim, self.num_actual_tokens, self.num_virtual_tokens)
        self.enc_decoder = enc_decoder
        self.train_iter = 0

        token_dim = self.token_dim
        mlp_hidden_size = self.hidden_size

        self.mlps = nn.ModuleList([
            MLP(self.enc_decoder.encoder.config.hidden_size, mlp_hidden_size)
            for _ in range(self.num_virtual_tokens)
        ])
        hidden_size = question_encoder.encoder.config.hidden_size

        if self.config.no_aggregate:
            self.aggregator = PassPrompt(config, question_encoder, token_dim,
                                         hidden_size, self.num_virtual_tokens, self.num_actual_tokens,
                                         dropout_p=config.dropout_p, )
        else:
            self.aggregator = Aggregator(
                config, question_encoder, token_dim,
                hidden_size, self.num_virtual_tokens, self.num_actual_tokens,
                dropout_p=config.dropout_p,
            )

    def set_base_lm(self, base_lm):
        self.base_model_torch_dtype = base_lm.base_lm.dtype
        self.base_num_layers = base_lm.config.num_hidden_layers
        self.base_num_attention_heads = base_lm.config.num_attention_heads

    def freeze_param(self):
        for param in self.base_lm.parameters():
            param.requires_grad = False

    def forward(self, update_batch, train=True):

        batch_size = _get_batch_size(update_batch['text_ids'], None)
        text_labels = update_batch['text_ids'].clone()
        text_labels[update_batch['text_attention'] != 1] = -100

        if self.config.lift_ratio != 1.0 and train:
            update_batch["gen_q_ids_amort"] = update_batch["gen_q_ids_amort"][:int(batch_size * self.config.lift_ratio)]
            update_batch["gen_q_attn_mask_amort"] = update_batch["gen_q_attn_mask_amort"][:int(batch_size * self.config.lift_ratio)]
            update_batch['text_ids'] = update_batch['text_ids'][:int(batch_size * self.config.lift_ratio)]
            update_batch['text_attention'] = update_batch['text_attention'][:int(batch_size * self.config.lift_ratio)]
            text_labels = text_labels[:int(batch_size * self.config.lift_ratio)]
            update_batch['qa_ids'] = update_batch['qa_ids'][:int(batch_size * self.config.lift_ratio)]
            update_batch['qa_attention'] = update_batch['qa_attention'][:int(batch_size * self.config.lift_ratio)]
            update_batch['qa_target_ids'] = update_batch['qa_target_ids'][:int(batch_size * self.config.lift_ratio)]

        prompts, prompt_latent, context_summary_bank = self.prompt(
            update_batch["text_ids_amort"],
            update_batch["text_attention_amort"],
            update_batch["gen_q_ids_amort"],
            update_batch["gen_q_attn_mask_amort"],
            train=train,
        )

        if self.config.lift_ratio != 1.0 and train and self.config.no_aggregate:
            context_summary_bank = context_summary_bank[:int(batch_size * self.config.lift_ratio)]

        with torch.no_grad():
            # initial text loss and qa outputs should be measured without prompts
            init_text_loss = self.base_lm(
                input_ids=update_batch['text_ids'],
                attention_mask=update_batch['text_attention'],
                labels=text_labels,
                prompts=None,
            ).loss
            init_qa_outputs = self.base_lm(
                input_ids=update_batch['qa_ids'],
                attention_mask=update_batch['qa_attention'],
                labels=update_batch['qa_target_ids'],
                prompts=None,
            )
            final_text_loss = self.base_lm(
                input_ids=update_batch['text_ids'],
                attention_mask=update_batch['text_attention'],
                labels=text_labels,
                prompts=prompts,
            ).loss

        if self.training and self.config.gpt_drop:
            self.base_lm.train()
        else:
            self.base_lm.eval()

        qa_output = self.base_lm(
            input_ids=update_batch['qa_ids'],
            attention_mask=update_batch['qa_attention'],
            labels=update_batch['qa_target_ids'],
            prompts=prompts
        )
        qa_loss = qa_output.loss

        metrics = {
            'text_loss': final_text_loss.item(),
            'text_gain_from_base': init_text_loss.item() - final_text_loss.item(),
            'qa_loss': qa_loss.item(),
            'qa_gain_from_base': init_qa_outputs.loss.item() - qa_loss.item(),
        }
        total_loss = qa_loss
        metrics['total_loss'] = total_loss.item()

        if not train:
            return total_loss, metrics, context_summary_bank

        return total_loss, metrics

    def compute_qa_metrics(self, batch, context_summary_bank, top_k=1, no_adapt=False):
        em_correct = 0

        avg_f1s = []
        total_cnt = len(batch['gen_q_ids'])
        use_cache = False

        with torch.no_grad():
            kwargs = {}
            if not no_adapt:
                continuous_prompt = self.predict_prompt_from_memory(
                    batch['gen_q_ids_amort'],
                    batch['gen_q_attn_mask_amort'],
                    context_summary_bank
                )

                if self.config.base_model in ['Llama2_7b']:
                    # continuous_prompt = DynamicCache.from_legacy_cache(continuous_prompt)
                    use_cache=True
                kwargs['prompts'] = continuous_prompt

            outs = self.base_lm.generate(
                input_ids=batch['gen_q_ids'],
                attention_mask=batch["gen_q_attn_mask"],
                use_cache=use_cache,
                max_length=batch['gen_q_ids'].shape[1] + 16,
                num_return_sequences=top_k,
                num_beams=1,
                peft_generation=not no_adapt,
                do_sample=False,
                early_stopping=False, # this is for beam search (not used for validation)
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

            dec = decode_to_clean_text(self.tokenizer, outs)
            texts = decode_to_clean_text(self.tokenizer, batch['gen_q_ids'])
            targets = decode_to_clean_text(self.tokenizer, batch['answer_ids'])
            for i in range(len(batch['gen_q_ids'])):
                answer = targets[i]

                predicted_answers = [dec[i * top_k + j][len(texts[i]):] for j in range(top_k)]

                answer_token = outs[i][len(batch['gen_q_ids'][i]):]
                if self.config.rank == 0:
                    print('------')
                    print(f"Question {self.tokenizer.decode(batch['gen_q_ids'][i], skip_special_tokens=True)}")
                    print(f"Answer GT: {answer}")
                    print(f"Answer Pred: {self.tokenizer.decode(answer_token, skip_special_tokens=True)}")
                    print(f"Answer Pred token: {answer_token}")
                    print('------')

                em = 0
                f_1s = []
                for pred_ans in predicted_answers:
                    if exact_match(pred_ans, answer, match_length=False):
                        em = 1
                    f_1s.append(f1_score(pred_ans, answer))
                em_correct += em
                avg_f1s.append(np.mean(f_1s))

        return {'em': em_correct / total_cnt, 'f1': np.mean(avg_f1s).item()}

    def load(self, epoch=None, checkpoint_step=None, target_path=None):
        """Loads a checkpoint.

        Args:
            either epoch and checkpoint step or an explicit path

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        if target_path is None:
            target_path = (
                f'{os.path.join(self._log_dir, "state")}'
                f'{epoch}-{checkpoint_step}.pt'
            )

        state = torch.load(target_path)['state_dict']

        self.load_state_dict(state, strict=False)
        print(f'Loaded checkpoint iteration {checkpoint_step}.')

        # if q encoder init is True, we copy enc_decoder weights to q encoder
        if self.config.qencoder_init:
            # use enc_decoder to initialize q encoder's encoder
            self.aggregator.question_encoder.encoder.load_state_dict(
                self.enc_decoder.state_dict()
            )

            # use self.learnable_prompts to initialize q encoder's learnable_prompts
            self.aggregator.question_encoder.learnable_prompts.data = self.learnable_prompts.data
            print(f'Loaded checkpoint iteration {checkpoint_step} to question encoder.')


    def save(self, epoch, checkpoint_step=None, log_dir=None, file_name=None, main_process=False):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            epoch (int)
            checkpoint_step (int): iteration to label checkpoint with
        """
        if main_process:
            if log_dir is None:
                log_dir = self._log_dir

            temp_base_lm = self.base_lm  # we don't want to save base_lm
            self.base_lm = None

            file_name = file_name or f'state{epoch}.pt'
            state_dict = self.state_dict()
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
            torch.save(
                dict(state_dict=state_dict), f'{os.path.join(log_dir, "checkpoints", file_name)}'
            )
            self.base_lm = temp_base_lm

        if self.config.distributed: dist.barrier()

    def validate(self, val_loader, main_process=False, no_adapt=False):
        metrics_dic = defaultdict(lambda: [])

        context_bank = []
        for i_step, batch in tqdm_distributed(
                main_process, enumerate(val_loader), desc='validation: Amortizing context',
                position=1, total=len(val_loader)
        ):
            _, metrics, context = self.forward(batch, train=False)
            context_bank.append(context)

            for k, v in metrics.items():
                metrics_dic[f'[AGG]{k}'].append(v)

            del batch  # free memory

        context_bank = torch.cat(context_bank, dim=0)
        return {k: np.mean(v) for k, v in metrics_dic.items()}, context_bank

    def validate_qa(self, val_gen_loader, context_bank=None, no_adapt=False, main_process=False):
        metrics_dic = defaultdict(lambda: [])

        for i_step, batch in tqdm_distributed(
                main_process, enumerate(val_gen_loader), desc='Validation: Aggregating context',
                position=1, total=len(val_gen_loader)
        ):

            if self.config.no_aggregate and context_bank is not None:
                context_bank = self.context_amortize(
                    batch["text_ids_amort"], batch["text_attention_amort"], False
                )

            qa_metrics = self.compute_qa_metrics(batch, context_bank, no_adapt=no_adapt)
            for k, v in qa_metrics.items():
                metrics_dic[f'[AGG]{k}'].append(v)

            if not no_adapt and not self.config.no_aggregate:
                for context_window in self.config.context_window_list:
                    context_hierarch = self.get_hierarchical_context(
                        batch, context_bank.clone().detach(), context_window
                    )
                    qa_metrics_hierarch = self.compute_qa_metrics(
                        batch, context_hierarch, no_adapt=no_adapt
                    )
                    for k, v in qa_metrics_hierarch.items():
                        metrics_dic[f'[AGG][Context{context_window}]{k}'].append(v)

                del context_hierarch
            del batch

        return {k: np.mean(v) for k, v in metrics_dic.items()}

    def get_prompt_from_logits(self, prompt_logit, return_pred=False):
        prompt_prob = F.softmax(prompt_logit / self.tau, dim=-1)
        pred = torch.argmax(prompt_prob, dim=-1)

        if self.training:
            self.prompt_prob_max_training += prompt_prob.max(dim=-1)[0].mean().item() * prompt_prob.shape[0]
            self.prompt_train_count += prompt_prob.shape[0]

        else:
            self.prompt_prob_max_eval += prompt_prob.max(dim=-1)[0].mean().item() * prompt_prob.shape[0]
            self.prompt_eval_count += prompt_prob.shape[0]

        word_embedding_weights = self.word_embedding_weights.to(prompt_prob.device)
        prompt = torch.matmul(prompt_prob, word_embedding_weights.detach())

        if return_pred:
            return prompt, pred
        return prompt

    def get_prob(self, train):
        if train:
            return self.prompt_prob_max_training / (self.prompt_train_count + 1e-8)
        else:
            return self.prompt_prob_max_eval / (self.prompt_eval_count + 1e-8)

    def context_amortize(self, indices, text_attention, train):
        if self.config.lift_ratio != 1.0 and train:
            indices_lift = indices[:int(indices.shape[0] * self.config.lift_ratio)]
            text_attention_lift = text_attention[:int(text_attention.shape[0] * self.config.lift_ratio)]
            indices_no_lift = indices[int(indices.shape[0] * self.config.lift_ratio):]
            text_attention_no_lift = text_attention[int(text_attention.shape[0] * self.config.lift_ratio):]
            with torch.no_grad():
                hidden_state_no_lift = self.enc_decoder(
                    input_ids=indices_no_lift,
                    attention_mask=text_attention_no_lift,
                )
                prompt_no_lift = []
                for i, mlp in enumerate(self.mlps):
                    prompt_no_lift.append(mlp(hidden_state_no_lift[:, i:i + 1, :]))

                prompt_no_lift = torch.cat(prompt_no_lift, dim=1)
            hidden_state_lift = self.enc_decoder(
                input_ids=indices_lift,
                attention_mask=text_attention_lift,
            )
            prompt_lift = []
            for i, mlp in enumerate(self.mlps):
                prompt_lift.append(mlp(hidden_state_lift[:, i:i + 1, :]))

            prompt_lift = torch.cat(prompt_lift, dim=1)
            prompt = torch.cat([prompt_lift, prompt_no_lift], dim=0)
        else:
            hidden_state = self.enc_decoder(
                input_ids=indices,
                attention_mask=text_attention,
            )

            prompt = []
            for i, mlp in enumerate(self.mlps):
                prompt.append(mlp(hidden_state[:, i:i + 1, :]))

            prompt = torch.cat(prompt, dim=1)
        return prompt  # continuous prompt  (save this to the memory bank)

    def get_hierarchical_context(self, batch, context_summary_bank, context_window):
        context_bank = context_summary_bank
        while len(context_bank) > context_window:
            context_bank = self.hierarchical_aggregate(
                batch['gen_q_ids_amort'],
                batch['gen_q_attn_mask_amort'],
                context_bank,
                context_window
            )
        return context_bank

    def hierarchical_aggregate(self, gen_q_ids_amort, gen_q_attn_mask_amort,
                               context_summary_bank, hierarchy_context_size):
        chunk_iter = range(0, gen_q_ids_amort.shape[1], hierarchy_context_size)
        prompt = []
        for k in chunk_iter:
            context_chunk = context_summary_bank[k:k + hierarchy_context_size]
            predict = self.aggregator(
                gen_q_ids_amort, gen_q_attn_mask_amort, context_chunk
            )
            prompt.append(predict)
        return torch.cat(prompt, dim=0)

    def predict_prompt_from_memory(self, gen_q_ids_amort, gen_q_attn_mask_amort, context_summary_bank):
        prompt_latent = self.aggregator(gen_q_ids_amort, gen_q_attn_mask_amort, context_summary_bank)
        prompt = self.prompt_latent_to_prompt(prompt_latent, train=False)
        return prompt

    def prob_init(self):
        self.prompt_prob_max_training = 0.
        self.prompt_train_count = 0.
        self.prompt_prob_max_eval = 0.
        self.prompt_eval_count = 0.

    def prompt_latent_to_prompt(self, prompt_latent, train=True):
        repeat_token = self.token_SA(prompt_latent)
        past_key_values = repeat_token.view(
            len(repeat_token),
            self.layer_num_virtual_tokens,
            self.base_num_layers * 2,
            self.base_num_attention_heads,
            self.token_dim // self.base_num_attention_heads,
        )
        prompt = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return prompt

    def prompt(self, indices, text_attention, gen_q_ids_amort, gen_q_attn_mask_amort, train):
        context_summary = self.context_amortize(indices, text_attention, train)
        prompt_latent = self.aggregator(gen_q_ids_amort, gen_q_attn_mask_amort, context_summary)

        if self.config.hierarchy_aware and train and self.config.hierarchy_aware_p > random.random():
            prompt_latent = self.aggregator(gen_q_ids_amort, gen_q_attn_mask_amort, prompt_latent)

        prompt = self.prompt_latent_to_prompt(prompt_latent, train)
        return prompt, prompt_latent, context_summary
