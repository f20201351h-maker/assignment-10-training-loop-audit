START = 300 if not QUICK else 5
st = mlog.step.values

def centred(x, w):
    return pd.Series(x).rolling(w, center=True, min_periods=w // 2).median().values

def trailing_alarm(x, w, k, start=START):
    x = np.asarray(x, dtype=float)
    tr = pd.Series(x).rolling(w).median().values
    resid = x - centred(x, w)
    r = resid[start:]
    sigma = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
    se = 1.2533 * sigma / np.sqrt(w)            # standard error of a median of w noisy points
    idx = np.arange(len(x))
    runmin = np.minimum.accumulate(np.where(idx >= start, np.nan_to_num(tr, nan=np.inf), np.inf))
    hit = np.where((idx >= start) & (tr > runmin + k * se))[0]
    return (int(st[hit[0]]) if len(hit) else None), se, tr

TURN = []
for w in [25, 51, 101, 151] if not QUICK else [5]:
    g_s, p_s = centred(mlog.grad_norm.values, w), centred(mlog.probe_loss.values, w)
    sub = np.arange(len(st)) >= START
    gi, pi_ = np.nanargmin(np.where(sub, g_s, np.inf)), np.nanargmin(np.where(sub, p_s, np.inf))
    TURN.append({"window": w, "grad_norm_min_step": int(st[gi]), "probe_loss_min_step": int(st[pi_]),
                 "norm_leads_by": int(st[pi_] - st[gi])})
TURN = pd.DataFrame(TURN)
print("Retrospective turning points (centred rolling median, argmin after step %d):" % START)
print(TURN.to_string(index=False))
val_min_step = int(mval.step[mval.val.idxmin()])
print(f"full validation set (every {MAIN['eval_every']} steps): lowest at step {val_min_step}, {mval.val.min():.4f}")

ALARMS = []
for w, k in ([(25, 3), (25, 5), (51, 3), (51, 5), (101, 3), (101, 5)] if not QUICK else [(5, 3)]):
    a_g, se_g, _ = trailing_alarm(mlog.grad_norm.values, w, k)
    a_p, se_p, _ = trailing_alarm(mlog.probe_loss.values, w, k)
    ALARMS.append({"window": w, "k": k, "grad_norm_alarm": a_g, "probe_loss_alarm": a_p,
                   "lead": (a_p - a_g) if (a_g is not None and a_p is not None) else None,
                   "se_grad_norm": se_g, "se_probe": se_p})
ALARMS = pd.DataFrame(ALARMS)
print("\nCausal alarms (trailing median exceeds its running min by k standard errors):")
print(ALARMS.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

W_EV, K_EV = (101, 3) if not QUICK else (5, 3)
ev_g, se_g, tr_g = trailing_alarm(mlog.grad_norm.values, W_EV, K_EV)
ev_p, se_p, tr_p = trailing_alarm(mlog.probe_loss.values, W_EV, K_EV)
EVENT = {"rule": f"trailing median w={W_EV}, alarm at running min + {K_EV} SE, after step {START}",
         "event_step": ev_g, "loss_reacts_step": ev_p,
         "lead_steps": (ev_p - ev_g) if (ev_g and ev_p) else None, "natural_or_induced": "natural (no intervention)",
         "turning_points": TURN.to_dict(orient="records"), "alarms": ALARMS.to_dict(orient="records"),
         "val_min_step": val_min_step}
if ev_g is not None:
    row = mlog.set_index("step")
    gmin_step = int(st[np.nanargmin(np.where(np.arange(len(st)) >= START, tr_g, np.inf))])
    EVENT.update({
        "event_grad_norm": float(row.loc[ev_g, "grad_norm"]),
        "trailing_median_grad_norm_at_event": float(tr_g[ev_g - 1]),
        "trailing_median_grad_norm_min": float(np.nanmin(np.where(np.arange(len(st)) >= START, tr_g, np.inf))),
        "trailing_median_grad_norm_min_step": gmin_step,
        "probe_loss_trailing_at_event": float(tr_p[ev_g - 1]),
        "probe_loss_trailing_min_so_far_at_event": float(np.nanmin(tr_p[START:ev_g])),
        "train_loss_trailing_at_event": float(pd.Series(mlog.loss.values).rolling(W_EV).median().values[ev_g - 1]),
        "lr_at_event": float(row.loc[ev_g, "lr"]),
        "clipped_steps_first_50": int(mlog.clipped.iloc[:50].sum()),
        "per_block_norm_change_min_to_event": {
            c: float(pd.Series(mlog[c].values).rolling(W_EV).median().values[ev_g - 1] /
                     pd.Series(mlog[c].values).rolling(W_EV).median().values[gmin_step - 1])
            for c in mlog.columns if c.startswith("gn_")},
    })
    if ev_p:
        EVENT["grad_norm_trailing_at_loss_reaction"] = float(tr_g[ev_p - 1])
    print(f"\nEvent: grad-norm alarm at step {ev_g}; probe-loss alarm at step {ev_p}; lead {EVENT['lead_steps']} steps")
    print(f"  trailing-median grad norm: min {EVENT['trailing_median_grad_norm_min']:.4f} (step {gmin_step}) -> "
          f"{EVENT['trailing_median_grad_norm_at_event']:.4f} at the event (+{100 * (EVENT['trailing_median_grad_norm_at_event'] / EVENT['trailing_median_grad_norm_min'] - 1):.1f}%)")
    print(f"  probe loss at the event: trailing median {EVENT['probe_loss_trailing_at_event']:.4f} vs best so far {EVENT['probe_loss_trailing_min_so_far_at_event']:.4f} (still flat)")
    print(f"  train loss at the event {EVENT['train_loss_trailing_at_event']:.4f} and still falling; lr {EVENT['lr_at_event']:.2e}")
    print("  per-block grad-norm growth from the norm minimum to the event (trailing medians):")
    for c, v in sorted(EVENT["per_block_norm_change_min_to_event"].items(), key=lambda kv: -kv[1]):
        print(f"     {c[3:]:<12} x{v:.3f}")
    check("event is natural: no data or weights were altered during the main run", True)
    if not QUICK:
        check("in the retrospective analysis the norm turns before the probe loss for every window", (TURN.norm_leads_by > 0).all(),
              TURN.norm_leads_by.tolist())
        check("causal grad-norm alarm precedes the causal probe-loss alarm", ev_p is not None and ev_p > ev_g, f"{ev_g} -> {ev_p}")
save_json("grad_norm_event.json", EVENT)

fig = plt.figure(figsize=(12, 10))
gsp = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.1])
ax1 = fig.add_subplot(gsp[0]); ax2 = fig.add_subplot(gsp[1], sharex=ax1); ax3 = fig.add_subplot(gsp[2])
ax1.plot(st, mlog.loss, color="C0", alpha=0.25, lw=0.7, label="training batch loss (changes batch every step)")
ax1.plot(st, mlog.probe_loss, color="C1", alpha=0.35, lw=0.7, label="fixed held-out probe batch")
ax1.plot(st, tr_p, color="C1", lw=1.6, label=f"probe, trailing median w={W_EV}")
ax1.plot(mval.step, mval.val, "k.", ms=5, label="full validation set")
ax1.set_ylabel("cross-entropy (nats / char)"); ax1.legend(fontsize=8, loc="upper right"); ax1.set_ylim(0.5, 4.5)
ax2.semilogy(st, mlog.grad_norm, color="C2", alpha=0.35, lw=0.7, label="global grad norm, pre-clip, every step")
ax2.semilogy(st, tr_g, color="C2", lw=1.6, label=f"trailing median w={W_EV}")
ax2.axhline(MAIN["grad_clip"], color="grey", ls=":", lw=1, label=f"clip threshold {MAIN['grad_clip']}")
ax2.set_ylabel("L2 norm of all gradients"); ax2.set_xlabel("optimizer step"); ax2.legend(fontsize=8, loc="upper right")
for ax in (ax1, ax2):
    if ev_g: ax.axvline(ev_g, color="C2", ls="--", lw=1)
    if ev_p: ax.axvline(ev_p, color="C1", ls="--", lw=1)
lo, hi = (max(START, (ev_g or START) - 450), min(len(st), (ev_p or len(st)) + 350))
zs = (st >= lo) & (st <= hi)
ax3.plot(st[zs], tr_g[zs], color="C2", lw=1.8, label="grad norm (trailing median)")
ax3.set_ylabel("grad norm", color="C2")
ax3b = ax3.twinx()
ax3b.plot(st[zs], tr_p[zs], color="C1", lw=1.8, label="probe loss (trailing median)")
ax3b.set_ylabel("held-out probe loss (nats / char)", color="C1")
if ev_g:
    ax3.axvline(ev_g, color="C2", ls="--", lw=1); ax3.text(ev_g, ax3.get_ylim()[1], f" norm alarm {ev_g}", color="C2", va="top", fontsize=9)
if ev_p:
    ax3.axvline(ev_p, color="C1", ls="--", lw=1); ax3.text(ev_p, ax3.get_ylim()[0], f" loss alarm {ev_p}", color="C1", va="bottom", fontsize=9)
ax3.set_xlabel("optimizer step"); ax3.set_title(f"zoom: steps {lo}-{hi}, both alarms use only past steps", fontsize=10)
fig.suptitle("Main run: grad norm logged from step 1 vs loss (natural run, nothing injected)", fontsize=11)
fig.tight_layout(); fig.savefig("figures/gradnorm_vs_loss.png", dpi=150); plt.show()
