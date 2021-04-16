"""
Implementation taken and modified from:
https://github.com/lucidrains/vit-pytorch
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., just_values=False):
        super().__init__()
        self.just_values = just_values

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        qkv_dim = inner_dim if self.just_values else inner_dim * 3
        self.to_qkv = nn.Linear(dim, qkv_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads

        # Values, queries, keys
        if self.just_values:
            v = self.to_qkv(x)
            v = rearrange(v, 'b n (h d) -> b h n d', h = h)
            q = v; k = v
        else:
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class TransformerModule(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., just_values=False, no_ffn=False):
        super().__init__()
        self.no_ffn = no_ffn
        if self.no_ffn:
            self.no_ffn_layer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            )
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, just_values = just_values))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        out = {}
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x)
            if self.no_ffn:
                x = self.no_ffn_layer(x)
            else:
                x = ff(x)
            out[str(i)] = x
        
        out["output"] = x
        out.pop(str(i)) # This one is duplicate

        return out


class Transformer(nn.Module):
    def __init__(self, cfg, pos_embedding=None):
        super().__init__()
        self.cfg = cfg

        # TODO before it was possible to choose the input format as an image or a sequence
        #assert self.cfg["trf_image_size"] % self.cfg["trf_patch_size"] == 0, 'Image dimensions must be divisible by the patch size.'
        #num_patches = (self.cfg["trf_image_size"] // self.cfg["trf_patch_size"]) ** 2
        #entry_dim = self.cfg["channels"] * self.cfg["patch_size"] ** 2
        #self.rearange_tensor = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.cfg["patch_size"], p2 = self.cfg["patch_size"])

        if self.cfg["pool"] == "token_0":
            self.token_0 = nn.Parameter(torch.randn(1, 1, self.cfg["token_dim"]))

        self.dropout = nn.Dropout(self.cfg["emb_dropout"])

        self.transformer = TransformerModule(
            dim=self.cfg["token_dim"], 
            depth=self.cfg["depth"], 
            heads=self.cfg["heads"], 
            dim_head=self.cfg["dim_head"], 
            mlp_dim=self.cfg["mlp_dim"], 
            dropout=self.cfg["dropout"],
            just_values=self.cfg["just_values"],
            no_ffn= self.cfg["no_ffn"] if "no_ffn" in self.cfg else False,
        )

        self.pool = self.cfg["pool"]
        self.to_latent = nn.Identity()

        self.layer_norm = nn.LayerNorm(self.cfg["token_dim"])

    def forward(self, x, pos_embedding=None, return_intermediate=False):
        # TODO before it was possible to choose the input format as an image or a sequence
        # if self.cfg["image_input"]:
        #     x = self.rearange_tensor(x)
        #     if pos_embedding is not None:
        #         pos_embedding = self.rearange_tensor(pos_embedding)

        b, n, _ = x.shape

        tokens_0 = repeat(self.token_0, '() n d -> b n d', b = b)

        if pos_embedding is None:
            pass
        else:
            x = x + pos_embedding

        x = torch.cat((tokens_0, x), dim=1)

        x = self.dropout(x)

        x = self.transformer(x)
        
        if return_intermediate:
            output = {}
            output["intermediate"] = x
        x = x["output"]

        if self.cfg["pool"] == "token_0":
            x = x[:, 0]
        elif self.cfg["pool"] == "mean":
            x = x.mean(dim=1)
        elif self.cfg["pool"] == "none":
            pass
        else:
            raise NotImplementedError

        x = self.to_latent(x)
        x = self.layer_norm(x)

        if return_intermediate:
            output["output"] = x
            return output
        else:
            return x