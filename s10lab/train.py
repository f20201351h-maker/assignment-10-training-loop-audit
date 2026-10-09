"""
The training loop, written so that every number it prints can be checked.

Loss convention: the model gives logits; we compute the per-token loss SUM and the VALID-token COUNT.
How those are combined across micro-batches is the whole of experiment 3:

  token          : sum_m sum_tokens loss / sum_m n_m        (every valid token has weight 1/N)
  mean_of_means  : (1/M) sum_m [ sum_tokens loss / n_m ]    (every micro-batch has weight 1/M)
"""
import math
import time
from contextlib import nullcontext

import torch
from torch.nn import functional as F

from .data import IGNORE


def autocast_ctx(device_type, precision):
    if precision == "bf16":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return nullcontext()


def token_loss_sum(logits, targets):
    """Per-token cross-entropy summed over valid targets, and the number of valid targets."""
    V = logits.size(-1)
    if logits.dtype not in (torch.float32, torch.float64):
        logits = logits.float()  # upcast bf16 logits; never downcast the fp64 gradient check
    s = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=IGNORE, reduction="sum")
    return s, (targets != IGNORE).sum()


def accumulate(model, micro_batches, mode, ctx, device):
    """Forward/backward over a window of micro-batches without stepping. Gradients pile up in .grad.
    Returns the loss the loop would *report* plus the token-weighted loss of the same window."""
    counts = [int((y != IGNORE).sum()) for _, y in micro_batches]
    N, M = sum(counts), len(micro_batches)
    assert N > 0
    sums = []
    for (x, y), n in zip(micro_batches, counts):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with ctx:
            logits = model(x)
        s, n_dev = token_loss_sum(logits, y)
        if mode == "token":
            (s / N).backward()
        elif mode == "mean_of_means":
            ((s / n) / M).backward()
        else:
            raise ValueError(mode)
        sums.append(s.detach())
    sums = torch.stack(sums)
    true_loss = sums.sum() / N
    if mode == "token":
        reported = true_loss
    else:
        reported = (sums / torch.tensor(counts, device=sums.device, dtype=sums.dtype)).mean()
    return {"reported": reported, "true": true_loss, "counts": counts, "sums": sums}


def full_batch_backward(model, micro_batches, ctx, device):
    """Reference: the same examples as ONE batch, loss = mean over all valid tokens."""
    X = torch.cat([x for x, _ in micro_batches]).to(device)
    Y = torch.cat([y for _, y in micro_batches]).to(device)
    with ctx:
        logits = model(X)
    s, n = token_loss_sum(logits, Y)
    (s / n).backward()
    return (s / n).detach()


def flat_grad(model):
    return torch.cat([p.grad.detach().flatten().double() for p in model.parameters()])


def compare_grads(g, g_ref):
    rel = float((g - g_ref).norm() / g_ref.norm())
    cos = float(torch.dot(g, g_ref) / (g.norm() * g_ref.norm()))
    return {"rel_l2_err": rel, "cosine": cos, "norm_ratio": float(g.norm() / g_ref.norm())}


# ---- grad-norm bookkeeping ----------------------------------------------------------------------

def param_groups_for_norms(model):
    """Map each parameter to a coarse group: emb (tied wte/lm_head), wpe, h.0..h.L-1, ln_f."""
    names = []
    for name, _ in model.named_parameters():
        if name.startswith("transformer.wte"):
            names.append("wte/lm_head")
        elif name.startswith("transformer.wpe"):
            names.append("wpe")
        elif name.startswith("transformer.h."):
            names.append("h." + name.split(".")[2])
        else:
            names.append("ln_f")
    uniq = list(dict.fromkeys(names))
    return uniq, torch.tensor([uniq.index(n) for n in names])


def grad_norms(model, group_index, n_groups):
    grads = [p.grad for p in model.parameters()]
    per_param = torch.stack(torch._foreach_norm(grads)).float()
    sq = torch.zeros(n_groups, device=per_param.device).index_add_(0, group_index.to(per_param.device),
                                                                     per_param ** 2)
    return per_param.pow(2).sum().sqrt(), sq.sqrt()


def lr_at(step, cfg):
    """nanoGPT schedule: linear warm-up, cosine decay to min_lr. step is 1-based."""
    if step <= cfg["warmup_steps"]:
        return cfg["lr"] * step / cfg["warmup_steps"]
    if step > cfg["max_steps"]:
        return cfg["min_lr"]
    r = (step - cfg["warmup_steps"]) / (cfg["max_steps"] - cfg["warmup_steps"])
    return cfg["min_lr"] + 0.5 * (1 + math.cos(math.pi * r)) * (cfg["lr"] - cfg["min_lr"])


@torch.no_grad()
def eval_loss(model, batches, ctx, device):
    """Token-weighted loss over a fixed list of (x, y) batches."""
    tot, n = 0.0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        with ctx:
            logits = model(x)
        s, c = token_loss_sum(logits, y)
        tot += float(s)
        n += int(c)
    assert n > 0, "eval set has no valid targets"
    return tot / n


def train_loop(model, opt, cfg, get_window, ctx, device, probe=None, val_batches=None, mode="token",
               log_every_val=50, pre_step_hook=None):
    """Generic loop. get_window(step) -> list of (x, y) micro-batches. Logs EVERY optimizer step.
    Per-step record: loss (reported), true token-weighted loss, pre-clip grad norm (total + per group),
    post-clip norm, clip flag, lr, valid tokens, wall time of the step (CUDA-synchronised), probe loss."""
    params = list(model.parameters())
    gnames, gidx = param_groups_for_norms(model)
    clip = cfg["grad_clip"]
    log, val_log = [], []
    cuda = device == "cuda"
    for step in range(1, cfg["max_steps"] + 1):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        window = get_window(step)
        probe_loss = eval_loss(model, [probe], ctx, device) if probe is not None else float("nan")
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = accumulate(model, window, mode, ctx, device)
        hook_out = pre_step_hook(step, model, window) if pre_step_hook is not None else None
        norm_manual, group_norms = grad_norms(model, gidx, len(gnames))
        norm_torch = torch.nn.utils.clip_grad_norm_(params, clip)  # returns the PRE-clip total norm
        post = torch.stack(torch._foreach_norm([p.grad for p in params])).pow(2).sum().sqrt()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rec = {
            "step": step, "lr": lr,
            "loss": float(res["reported"]), "true_loss": float(res["true"]),
            "grad_norm": float(norm_torch), "grad_norm_manual": float(norm_manual),
            "grad_norm_post_clip": float(post), "clipped": bool(float(norm_torch) > clip),
            "valid_tokens": sum(res["counts"]), "dt_s": dt, "tok_per_s": sum(res["counts"]) / dt,
            "probe_loss": probe_loss,
        }
        rec.update({f"gn_{n}": float(v) for n, v in zip(gnames, group_norms)})
        if mode != "token":
            rec["counts"] = res["counts"]
        assert math.isfinite(rec["loss"]) and math.isfinite(rec["grad_norm"]), f"non-finite at step {step}"
        assert abs(rec["grad_norm"] - rec["grad_norm_manual"]) <= 1e-3 * rec["grad_norm"] + 1e-6, rec
        if hook_out:
            rec.update(hook_out)
        log.append(rec)
        if val_batches is not None and (step % log_every_val == 0 or step == 1 or step == cfg["max_steps"]):
            val_log.append({"step": step, **{k: eval_loss(model, v, ctx, device) for k, v in val_batches.items()}})
    return log, val_log


# ---- throughput -----------------------------------------------------------------------------------

def make_fast_step(model, opt, ctx, clip):
    """The same step as train_loop with no logging and no host syncs (what we time for MFU)."""
    params = list(model.parameters())

    def step(window):
        N = sum(int(y.numel()) for _, y in window)  # dense batches: every target is valid
        for x, y in window:
            with ctx:
                logits = model(x)
            s, _ = token_loss_sum(logits, y)
            (s / N).backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return N
    return step


def time_steps(step_fn, window_fn, n_warm, n_timed, device, min_seconds=6.0):
    """Warm up, calibrate, then time at least n_timed steps and at least min_seconds, synchronised at both ends."""
    for i in range(n_warm):
        step_fn(window_fn(i))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(5):
        step_fn(window_fn(n_warm + i))
    torch.cuda.synchronize()
    per_step = (time.perf_counter() - t0) / 5
    n = max(n_timed, math.ceil(min_seconds / per_step))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokens = 0
    for i in range(n):
        tokens += step_fn(window_fn(n_warm + 5 + i))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"tokens": tokens, "seconds": dt, "steps": n, "tok_per_s": tokens / dt, "ms_per_step": 1000 * dt / n}
