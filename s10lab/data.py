"""
Tiny Shakespeare, character level (the nanoGPT shakespeare_char setup).

Two views of the same text:
  * dense chunks  : random contiguous windows of T+1 characters, every target valid (main loop, MFU);
  * speeches      : the text split on blank lines into speeches of very different lengths, right-padded
                    to T with targets = IGNORE. This is what gives micro-batches unequal valid-token counts
                    for the accumulation experiment, without inventing artificial lengths.
"""
import hashlib
import os
import urllib.request

import numpy as np
import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
IGNORE = -100


def load_text(cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "input.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    raw = open(path, "rb").read()
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest(), path


class CharData:
    def __init__(self, text, train_frac=0.9):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        n = len(text)
        self.split_at = int(n * train_frac)
        self.text = {"train": text[: self.split_at], "val": text[self.split_at:]}
        self.ids = {k: torch.tensor(self.encode(v), dtype=torch.long) for k, v in self.text.items()}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    # ---- dense view ---------------------------------------------------------------------------
    def dense_chunk(self, split, B, T, gen):
        """[B, T+1] contiguous windows. Inputs are chunk[:, :-1], targets chunk[:, 1:]."""
        data = self.ids[split]
        ix = torch.randint(len(data) - T - 1, (B,), generator=gen)
        return torch.stack([data[i: i + T + 1] for i in ix.tolist()])

    # ---- speech view --------------------------------------------------------------------------
    def speeches(self, split, T):
        """Each speech -> (input ids, target ids) truncated to T predictions. Speeches with < 2 chars dropped."""
        out = []
        for sp in self.text[split].split("\n\n"):
            sp = sp.strip("\n") + "\n"
            ids = self.encode(sp)[: T + 1]
            if len(ids) >= 2:
                out.append(ids)
        return out


def split_chunk(chunk):
    """Shift by one: position t of the input predicts character t+1."""
    return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()


def pad_speeches(speech_list, T, pad_id=0):
    """Right-pad to T. Padded input positions get pad_id, padded targets get IGNORE.
    Causal attention means real positions never look at the padding to their right."""
    B = len(speech_list)
    x = torch.full((B, T), pad_id, dtype=torch.long)
    y = torch.full((B, T), IGNORE, dtype=torch.long)
    for b, ids in enumerate(speech_list):
        n = len(ids) - 1
        x[b, :n] = torch.tensor(ids[:-1])
        y[b, :n] = torch.tensor(ids[1:])
    return x, y


def length_buckets(speech_list, n_buckets):
    """Sort speeches by number of predictions and cut into equal-count buckets (length-grouped batching,
    a common way to cut padding, and exactly the setting where micro-batches end up with unequal tokens)."""
    order = sorted(range(len(speech_list)), key=lambda i: len(speech_list[i]))
    edges = np.linspace(0, len(order), n_buckets + 1).astype(int)
    return [[order[j] for j in range(edges[b], edges[b + 1])] for b in range(n_buckets)]


def make_windows(speech_list, buckets, n_steps, micro_bs, seed):
    """n_steps accumulation windows. Window = one micro-batch drawn from each length bucket, in shuffled order.
    Returns a list of windows; each window is a list of index lists (one per micro-batch)."""
    rng = np.random.default_rng(seed)
    windows = []
    for _ in range(n_steps):
        mbs = [rng.choice(b, size=micro_bs, replace=False).tolist() for b in buckets]
        rng.shuffle(mbs)
        windows.append(mbs)
    return windows
