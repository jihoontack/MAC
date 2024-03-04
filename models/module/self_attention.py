import warnings
import torch
from torch import nn

from einops import rearrange, repeat

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x) + x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., quant_type=None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.quant_type = quant_type
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TokenSelfAttend(nn.Module):
    def __init__(self, dim, num_actual_tokens, num_virtual_tokens, dropout = 0.):
        super().__init__()

        self.num_actual_tokens = num_actual_tokens
        self.num_virtual_tokens = num_virtual_tokens

        self.repeat_time = self.num_actual_tokens // self.num_virtual_tokens
        self.denom = self.num_actual_tokens % self.num_virtual_tokens

        if self.denom > 0:
            warnings.warn("num_actual_tokens is not divisible by num_virtual_tokens. "
                          "The last {} tokens will be repeated {} times".format(self.denom, self.repeat_time + 1))

        self.mlp_layers = nn.ModuleList([
            FeedForward(dim, 768, dropout=dropout)
            for _ in range(self.num_actual_tokens)
        ])  #  need to check this part again (slightly changed this part when refactoring)


    def forward(self, x):
        if self.repeat_time > 1:
            if self.denom > 0:
                x = torch.cat(
                    [repeat(x[:, :self.denom], 'b n d -> b (n r) d', r=self.repeat_time + 1),
                     repeat(x[:, self.denom:], 'b n d -> b (n r) d', r=self.repeat_time)],
                    dim=1)
            else:
                x = repeat(x, 'b n d -> b (n r) d', r=self.repeat_time)

        out = []
        for i, mlp in enumerate(self.mlp_layers):
            out.append(mlp(x[:, i:i + 1, :]))

        return torch.cat(out, dim=1)
