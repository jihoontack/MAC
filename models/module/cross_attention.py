import torch
from inspect import isfunction
from torch import nn, einsum
from einops import rearrange, repeat


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=96, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context, mask=None):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            dim,
            context_dim,
            num_heads=8,
            dim_head=96,
            proj_drop=0.,
            attn_drop=0.,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)
        self.attn = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=attn_drop,
        )

        hidden_dim = dim_head * num_heads if dim > dim_head * num_heads else dim
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(
            dim,
            hidden_dim,
            dropout=proj_drop,
        )

    def forward(self, x, context):  # x: question, context: prompts
        x = x + self.attn(self.norm1(x), self.norm_context(context))
        x = x + self.mlp(self.norm2(x))
        return x
