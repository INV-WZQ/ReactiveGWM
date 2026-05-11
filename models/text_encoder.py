"""UMT5 text encoder used by Wan2.2 (custom diffsynth weight layout).

Architecture is a stripped UMT5: tied vocab, GELU-gated FFN, T5-style relative
position bias, RMS-style layer norm. Matches the keys in
`models_t5_umt5-xxl-enc-bf16.pth`.
"""
import html
import math
import re

import ftfy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


def _fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        c = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-c, max=c)
    return x


class _GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class _T5LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.type_as(self.weight)
        return self.weight * x


class _T5Attention(nn.Module):
    def __init__(self, dim, dim_attn, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)

    def forward(self, x, mask=None, pos_bias=None):
        b, n, c = x.size(0), self.num_heads, self.head_dim
        q = self.q(x).view(b, -1, n, c)
        k = self.k(x).view(b, -1, n, c)
        v = self.v(x).view(b, -1, n, c)
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias = attn_bias + pos_bias
        if mask is not None:
            mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)
        attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        out = torch.einsum('bnij,bjnc->binc', attn, v).reshape(b, -1, n * c)
        return self.o(out)


class _T5FeedForward(nn.Module):
    def __init__(self, dim, dim_ffn):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), _GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x) * self.gate(x))


class _T5SelfAttention(nn.Module):
    def __init__(self, dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos):
        super().__init__()
        self.norm1 = _T5LayerNorm(dim)
        self.attn = _T5Attention(dim, dim_attn, num_heads)
        self.norm2 = _T5LayerNorm(dim)
        self.ffn = _T5FeedForward(dim, dim_ffn)
        self.pos_embedding = None if shared_pos else _T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)

    def forward(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.pos_embedding is None else self.pos_embedding(x.size(1), x.size(1))
        x = _fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = _fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class _T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        rel = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        rel = self._bucket(rel)
        return self.embedding(rel).permute(2, 0, 1).unsqueeze(0).contiguous()

    def _bucket(self, rel):
        if self.bidirectional:
            n = self.num_buckets // 2
            buckets = (rel > 0).long() * n
            rel = rel.abs()
        else:
            n = self.num_buckets
            buckets = 0
            rel = -torch.min(rel, torch.zeros_like(rel))
        max_exact = n // 2
        large = max_exact + (torch.log(rel.float() / max_exact) /
                             math.log(self.max_dist / max_exact) * (n - max_exact)).long()
        large = torch.min(large, torch.full_like(large, n - 1))
        buckets += torch.where(rel < max_exact, rel, large)
        return buckets


class WanTextEncoder(nn.Module):
    """UMT5 encoder used by Wan2.2-TI2V-5B; loads `models_t5_umt5-xxl-enc-bf16.pth`."""
    def __init__(self, vocab=256384, dim=4096, dim_attn=4096, dim_ffn=10240,
                 num_heads=64, num_layers=24, num_buckets=32, shared_pos=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, dim)
        self.pos_embedding = _T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.blocks = nn.ModuleList([
            _T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos)
            for _ in range(num_layers)
        ])
        self.norm = _T5LayerNorm(dim)
        self.shared_pos = shared_pos

    def forward(self, ids, mask=None):
        x = self.token_embedding(ids)
        e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None
        for blk in self.blocks:
            x = blk(x, mask=mask, pos_bias=e)
        return self.norm(x)


def _whitespace_clean(text: str) -> str:
    """Bit-identical port of `diffsynth.models.wan_video_text_encoder.
    whitespace_clean(basic_clean(text))`. Run on every prompt before tokenizing
    — both the training-time pipeline and the cache precompute do this, so
    skipping it makes the negative prompt's T5 embedding diverge (e.g. the
    Chinese fullwidth comma `，` U+FF0C → ASCII `,` U+002C swap maps to a
    completely different UMT5 token id). Don't drop this."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class WanTokenizer:
    """Wraps HF AutoTokenizer with the (return_mask, max_length) interface used by Wan."""
    def __init__(self, path: str, seq_len: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.seq_len = seq_len

    def __call__(self, text):
        if isinstance(text, str):
            text = [text]
        text = [_whitespace_clean(t) for t in text]
        out = self.tokenizer(text, return_tensors='pt', padding='max_length',
                             truncation=True, max_length=self.seq_len,
                             add_special_tokens=True)
        return out.input_ids, out.attention_mask
