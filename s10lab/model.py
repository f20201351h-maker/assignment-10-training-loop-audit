"""
A compact GPT, faithful to Karpathy's nanoGPT (github.com/karpathy/nanoGPT, MIT licence).

Same module tree and parameter names as nanoGPT's model.py (transformer.wte, transformer.h.0.attn.c_attn, ...),
same init (N(0, 0.02), residual projections scaled by 1/sqrt(2L)), same tied lm_head.

Two additions for Session 10:
  * a tracer: inside `tracing(rec)` every interesting activation is reported with its symbolic shape and a
    one-line meaning, so a training step can describe its own geometry;
  * while tracing, attention runs the explicit softmax(QK^T/sqrt(D))V path so the [B,H,T,T] score tensor
    exists and can be inspected; outside tracing it uses F.scaled_dot_product_attention like nanoGPT.

The model returns logits only. The loss lives in train.py because Session 10's accumulation experiment
needs the per-token *sum* and the valid-token *count*, not a pre-averaged mean.
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

_TRACER = None


@contextmanager
def tracing(recorder):
    """recorder(name, tensor, symbolic_shape, meaning) is called for every traced activation."""
    global _TRACER
    prev, _TRACER = _TRACER, recorder
    try:
        yield recorder
    finally:
        _TRACER = prev


def _rec(name, t, sym, meaning):
    if _TRACER is not None:
        _TRACER(name, t, sym, meaning)


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (nanoGPT)."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer_idx = layer_idx

    def forward(self, x):
        B, T, C = x.size()
        H, D = self.n_head, C // self.n_head
        p = f"h.{self.layer_idx}.attn"
        qkv = self.c_attn(x)
        _rec(f"{p}.qkv", qkv, "[B,T,3C]", "one fused projection: Q, K and V for every head, side by side")
        q, k, v = qkv.split(self.n_embd, dim=2)
        _rec(f"{p}.q", q, "[B,T,C]", "queries before the head split (K and V have the same shape)")
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        _rec(f"{p}.q_heads", q, "[B,H,T,D]", "C split into H heads of D dims; heads moved next to batch")
        _rec(f"{p}.k_heads", k, "[B,H,T,D]", "keys, same layout as queries")
        _rec(f"{p}.v_heads", v, "[B,H,T,D]", "values, same layout as queries")
        if _TRACER is not None:
            # explicit path so the score matrix exists and can be checked
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
            _rec(f"{p}.scores", att, "[B,H,T,T]", "query position t (row) scoring key position s (column), per head")
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att.float(), dim=-1).to(q.dtype)
            _rec(f"{p}.probs", att, "[B,H,T,T]", "softmax over keys; causal, so the upper triangle is exactly 0")
            y = att @ v
            y_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            _rec(f"{p}.sdpa_max_abs_diff", (y.float() - y_ref.float()).abs().max(), "[]",
                 "explicit attention vs the fused SDPA kernel used in training")
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        _rec(f"{p}.y_heads", y, "[B,H,T,D]", "per-head weighted sum of values")
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        _rec(f"{p}.y_merged", y, "[B,T,C]", "heads concatenated back into one C-wide vector per token")
        y = self.resid_dropout(self.c_proj(y))
        _rec(f"{p}.out", y, "[B,T,C]", "attention output after c_proj, what gets added to the residual")
        return y


class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_idx = layer_idx

    def forward(self, x):
        p = f"h.{self.layer_idx}.mlp"
        x = self.c_fc(x)
        _rec(f"{p}.c_fc", x, "[B,T,4C]", "expanded hidden: each token widened 4x")
        x = self.gelu(x)
        _rec(f"{p}.gelu", x, "[B,T,4C]", "after the nonlinearity, still 4C wide")
        x = self.c_proj(x)
        _rec(f"{p}.out", x, "[B,T,C]", "projected back down to C")
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config, layer_idx)
        self.layer_idx = layer_idx

    def forward(self, x):
        p = f"h.{self.layer_idx}"
        _rec(f"{p}.in", x, "[B,T,C]", "residual stream entering the block")
        h = self.ln_1(x)
        _rec(f"{p}.ln_1", h, "[B,T,C]", "normalised over C, per token; shape unchanged")
        x = x + self.attn(h)
        _rec(f"{p}.resid_after_attn", x, "[B,T,C]", "residual + attention")
        h = self.ln_2(x)
        _rec(f"{p}.ln_2", h, "[B,T,C]", "normalised again before the MLP")
        x = x + self.mlp(h)
        _rec(f"{p}.out", x, "[B,T,C]", "residual + MLP, the block output")
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying, as in nanoGPT
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size
        _rec("idx", idx, "[B,T]", "token ids: B sequences, T positions each")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        _rec("tok_emb", tok_emb, "[B,T,C]", "row idx[b,t] of wte: one C-dim vector per token")
        pos_emb = self.transformer.wpe(pos)
        _rec("pos_emb", pos_emb, "[T,C]", "learned position vectors; no batch dim, broadcast over B")
        x = self.transformer.drop(tok_emb + pos_emb)
        _rec("x0", x, "[B,T,C]", "token + position, the residual stream at layer 0")
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        _rec("ln_f", x, "[B,T,C]", "final hidden state, normalised")
        logits = self.lm_head(x)
        _rec("logits", logits, "[B,T,V]", "unnormalised score for each of V next characters, at every position")
        return logits

    def num_params(self):
        # parameters() de-duplicates the tied wte/lm_head tensor
        return sum(p.numel() for p in self.parameters())

    def configure_optimizer(self, lr, weight_decay, betas, device_type):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
