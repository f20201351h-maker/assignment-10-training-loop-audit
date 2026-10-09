**What this shows.** I nudged `transformer.h.0.attn.c_attn.weight[99, 158]` (a query weight for head 1, dim 35, input channel 158), starting from w = 0.0255171787.
A +1e-4 nudge lowered the loss by 1.826e-7, and -1e-4 raised it by the same amount. That is `-1.826167128627e-03` numerically against `-1.826167129583e-03` from `backward()`,
a relative error of 5.2e-10. Eight random scalars from other layers agree just as well (worst absolute error 5.8e-12).

Things I did not expect:

- I started with eps = 1e-6 out of habit. The sweep shows that this is already on the round-off side of the valley in float64 (4.8e-8 instead of 5.2e-10), so I moved to 1e-4.
- In float32 the best I could get was about 1e-3 relative. At eps = 1e-5 and below the numerical gradient is exactly zero: `L(w+eps)` and `L(w-eps)` round to the same fp32 number.
- With TF32 matmuls switched on, the check falls apart: for eps <= 3e-3 the relative error is 1.9 to 378. TF32 rounds the matmul inputs to 10 mantissa bits, so the loss jitters by more than the nudge moves it.
  nanoGPT's `train.py` turns TF32 on. A gradient check run in that setting would "fail" for reasons that have nothing to do with backprop.

This verifies a handful of scalars at the initial weights on one batch. It does not prove every gradient in the model is right.
