
import torch, torch.nn as nn, torch.nn.functional as F

def rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1).flatten(-2)

class Block(nn.Module):
    def __init__(self, dim, heads, drop):
        super().__init__()
        self.h, self.drop = heads, drop
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3*dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(4*dim, dim), nn.Dropout(drop))
    def forward(self, x, cos, sin):
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, N, 3, self.h, D//self.h).permute(2,0,3,1,4)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(a.transpose(1,2).reshape(B, N, D))
        return x + self.mlp(self.ln2(x))

class TinyMyoLike(nn.Module):
    def __init__(self, n_ch=16, win=200, patch=20, dim=192, depth=8,
                 heads=3, n_cls=53, drop=0.1):
        super().__init__()
        self.n_ch, self.n_p, self.patch = n_ch, win//patch, patch
        self.embed = nn.Linear(patch, dim)
        self.ch_embed = nn.Embedding(n_ch, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(n_ch*dim, n_cls)
        hd = dim//heads
        pos = torch.arange(self.n_p).repeat(n_ch).float()
        ang = pos[:,None] * (1.0/10000 ** (torch.arange(0, hd, 2).float()/hd))[None]
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
    def forward(self, x):
        B = x.shape[0]
        t = self.embed(x.reshape(B, self.n_ch, self.n_p, self.patch))
        t = t + self.ch_embed.weight[None,:,None,:]
        t = t.flatten(1,2)
        for blk in self.blocks:
            t = blk(t, self.cos.to(t.dtype), self.sin.to(t.dtype))
        t = self.norm(t).reshape(B, self.n_ch, self.n_p, -1).mean(2)
        return self.head(t.flatten(1))
