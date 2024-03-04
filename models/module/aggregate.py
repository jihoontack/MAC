import torch
import torch.nn as nn
from einops import rearrange

from models.module.cross_attention import CrossAttentionBlock


class MLP(nn.Module):
    def __init__(self, mlp_hidden_size, hidden_size):
        super().__init__()
        if mlp_hidden_size == hidden_size:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(mlp_hidden_size, hidden_size)

        self.mlp = nn.Sequential(
            nn.LayerNorm(mlp_hidden_size),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        return self.skip(x) + self.mlp(x)


class Aggregator(nn.Module):
    def __init__(self, config, question_encoder, baselm_token_dim, amort_enc_dim,
                 num_virtual_tokens, num_actual_tokens, dropout_p=0.):
        super().__init__()
        self.config = config
        self.num_virtual_tokens = num_virtual_tokens
        self.question_encoder = question_encoder
        self.mlps = nn.ModuleList([
            MLP(amort_enc_dim, baselm_token_dim)
            for _ in range(num_virtual_tokens)
        ])

        num_cross_attention_blocks = self.config.num_cross_attention_blocks
        self.cross_attention_list = nn.ModuleList([
            CrossAttentionBlock(dim=baselm_token_dim, context_dim=baselm_token_dim,
                                proj_drop=dropout_p, attn_drop=dropout_p,
                                num_heads=8, dim_head=96)
            for i in range(num_cross_attention_blocks)]
        )

    def forward(self, question_indices, qa_attention, prompt_set):
        # prompt set into batch of prompt set
        batch_prompt_set = prompt_set.unsqueeze(0).repeat(
            question_indices.shape[0], *([1] * len(prompt_set.shape))
        )
        batch_prompt_set = rearrange(batch_prompt_set, 'b s n d -> b (s n) d')

        hidden_state = self.question_encoder(
            input_ids=question_indices,
            attention_mask=qa_attention,
        )
        condition_init = []
        for i, mlp in enumerate(self.mlps):
            condition_init.append(mlp(hidden_state[:, i:i + 1, :]))
        condition_init = torch.cat(condition_init, dim=1)

        condition = condition_init
        for cross_attention  in self.cross_attention_list:
            condition = cross_attention(condition, batch_prompt_set)

        return condition  # since condition is the query of the cross attention


class PassPrompt(nn.Module):
    def __init__(self, config, question_encoder, hidden_size, mlp_hidden_size,
                 num_virtual_tokens, num_actual_tokens, dropout_p=0.):
        super().__init__()

    def forward(self, question_indices, qa_attention, prompt_set):
        return prompt_set
