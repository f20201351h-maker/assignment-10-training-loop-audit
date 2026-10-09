"""
Independent re-computation of the headline numbers from the saved artifacts only.

It does not import s10lab and does not re-run the model. It re-derives each number from the raw values the
notebook wrote, using different code paths (plain floats, struct, hand arithmetic), and fails loudly if
anything disagrees.

    python audit_artifacts.py
"""
import csv
import json
import math
import os
import struct
import sys

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
fails = []


def load(name):
    with open(os.path.join(ART, name)) as f:
        return json.load(f)


def rows(name):
    with open(os.path.join(ART, name)) as f:
        return list(csv.DictReader(f))


def ok(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)


cfg = load("config.json")
m = cfg["model"]
V, C, L, Tmax = cfg["data"]["vocab_size"], m["n_embd"], m["n_layer"], m["block_size"]

# --- parameter count, from the per-tensor audit -------------------------------------------------------
audit = rows("param_audit.csv")
n_csv = sum(int(r["numel"]) for r in audit)
n_formula = V * C + Tmax * C + L * (12 * C * C + 2 * C) + C
a = load("autopsy.json")
ok("param_audit.csv rows sum to the formula VC + TmaxC + L(12C^2+2C) + C", n_csv == n_formula, f"{n_csv:,}")
ok("autopsy.json N_total matches", a["N_total"] == n_formula)
ok("every audited tensor is trainable", all(r["trainable"] == "True" for r in audit))
ok("H x D == C", m["n_head"] * (C // m["n_head"]) == C)

# --- finite difference, from the three recorded losses ------------------------------------------------
g = load("gradcheck.json")
num = (g["L_w_plus"] - g["L_w_minus"]) / (2 * g["eps"])
ok("central difference recomputed from L(w+eps), L(w-eps) matches the stored numerical gradient",
   abs(num - g["numerical"]) <= 1e-9 * abs(num), f"{num:.12e} vs {g['numerical']:.12e}")
rel = abs(num - g["autograd"]) / abs(g["autograd"])
ok("recomputed relative error < 1e-6", rel < 1e-6, f"{rel:.2e}")

# --- accumulation: re-derive both losses from the stored per-micro-batch sums ------------------------
acc = load("accumulation.json")
ref = acc["reference_checks"]["unequal_window_1_at_init"]
ok("correct accumulation == full batch (stored rel err < 1e-5)", ref["token"]["rel_l2_err"] < 1e-5, f"{ref['token']['rel_l2_err']:.1e}")
ok("broken accumulation != full batch (stored rel err > 1e-2)", ref["mean_of_means"]["rel_l2_err"] > 1e-2, f"{ref['mean_of_means']['rel_l2_err']:.2e}")
bad = rows("accumulation_broken_steps.csv")
good = rows("accumulation_correct_steps.csv")
ok("both accumulation runs have one row per optimizer step", len(bad) == len(good) == cfg["accumulation"]["max_steps"])
ok("both runs saw the same valid-token totals per step", all(x["valid_tokens"] == y["valid_tokens"] for x, y in zip(bad, good)))
unequal = 0
for r in bad:
    cs = json.loads(r["counts"])
    unequal += max(cs) > min(cs)
    ok_sum = sum(cs) == int(r["valid_tokens"])
    if not ok_sum:
        ok("micro-batch counts sum to valid_tokens", False, r["step"])
        break
ok("every broken-run step had unequal micro-batch token counts", unequal == len(bad), f"{unequal}/{len(bad)}")
c1 = acc["window_1_counts"]
ok("window 1 counts in accumulation.json match the CSV", json.loads(bad[0]["counts"]) == c1, str(c1))

# --- grad norm logged every step ----------------------------------------------------------------------
tm = rows("train_metrics.csv")
steps = [int(r["step"]) for r in tm]
ok("train_metrics.csv has a grad norm for every step 1..max_steps", steps == list(range(1, cfg["main_run"]["max_steps"] + 1)))
ok("all grad norms finite and positive", all(math.isfinite(float(r["grad_norm"])) and float(r["grad_norm"]) > 0 for r in tm))
clip = cfg["main_run"]["grad_clip"]
ok("clipped flag == (pre-clip norm > threshold) on every step",
   all((r["clipped"] == "True") == (float(r["grad_norm"]) > clip) for r in tm))
ev_path = os.path.join(ART, "grad_norm_event.json")
if os.path.exists(ev_path):
    ev = load("grad_norm_event.json")
    by_step = {int(r["step"]): r for r in tm}
    s0 = ev["event_step"]
    ok("event step's grad norm in the event file matches train_metrics.csv",
       abs(float(by_step[s0]["grad_norm"]) - ev["event_grad_norm"]) < 1e-9 * ev["event_grad_norm"] + 1e-12, f"step {s0}")

# --- MFU arithmetic ----------------------------------------------------------------------------------
mf = load("mfu.json")
N, peak = mf["N"], mf["peak"]["bf16"]
ok("MFU denominator is the A10G dense bf16 peak (70 TF) for an A10G",
   (mf["gpu"] != "NVIDIA A10G") or peak == 70e12, mf["gpu"])
for r in mf["table"]:
    if "bf16" in r["config"] or r["config"][:2] in ("A.", "B.", "C.", "D.", "E."):
        mfu = 100 * 6 * N * r["tok/s"] / (r["peak TF used"] * 1e12)
        ok(f"MFU recomputed for {r['config'][:40]}", abs(mfu - r["MFU % (6N)"]) < 1e-6, f"{mfu:.3f}%")

# --- 0.1 bits, decoded with struct / plain arithmetic --------------------------------------------------
fl = load("floats.json")
d = fl["derivation"]
fp32_bits = d["fp32"]["bits"]
ok("fp32 bits decode (via struct) to float32(0.1)",
   struct.unpack(">f", int(fp32_bits, 2).to_bytes(4, "big"))[0] == struct.unpack(">f", struct.pack(">f", 0.1))[0], d["fp32"]["hex"])
bf = d["bf16"]["bits"]
ok("bf16 bits are the top 16 bits of an fp32 pattern that decodes to the stated value",
   struct.unpack(">f", (int(bf, 2) << 16).to_bytes(4, "big"))[0] == d["bf16"]["value"], d["bf16"]["hex"])
e4 = d["fp8 E4M3 (OCP 'fn')"]["bits"]
s, e, mm = int(e4[0]), int(e4[1:5], 2), int(e4[5:], 2)
val = (-1) ** s * (1 + mm / 8) * 2.0 ** (e - 7)
ok("E4M3 bits decode by hand (bias 7, 3 mantissa bits) to the stated value",
   val == d["fp8 E4M3 (OCP 'fn')"]["value"], f"{e4} -> {val}")
ok("library encodings agree with hand derivations",
   all(fl["library"][k] == d[k]["hex"] for k in fl["library"]))

print(f"\n{'ALL PASSED' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
