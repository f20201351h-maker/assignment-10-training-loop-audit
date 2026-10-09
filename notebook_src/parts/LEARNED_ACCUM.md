**What this shows.**

- In the first window the four micro-batches held 286, 1799, 502 and 908 real targets. Under average-of-averages, each token in the 286-target micro-batch counted 3.06x as much as it should, and each token in the 1799 one counted 0.49x.
- Against the same 32 sequences run as one batch, token-normalised accumulation matches to 3.2e-7 (fp32 round-off). Average-of-averages is off by 18% in L2 (cosine 0.985).
  With equal token counts the two are identical (2.8e-7), which is exactly how this bug hid.
- Panel (a) is the part that matters. The loss each loop *prints* is almost the same: 1.552 vs 1.553 over the last 100 steps. The broken loop is minimising the number it prints, so that number falls as nicely as the correct one.
  Scored per token on the same windows (b), the broken run is at 1.750, not 1.552. On held-out speeches it ends 0.147 nats/char worse: +0.216 on long speeches and +0.031 on short ones.
- I expected the broken run to at least win on short speeches, since it up-weights them. It does not; it just loses by less there. I don't have a clean explanation for that.
- The broken gradient drifts further from the correct one as training goes on: cosine 0.985 at step 1, then a median of 0.82 (minimum 0.76) after step 50 (d).

Both runs used the same initial state dict (hash 9c95ba38b5b8faa7), the same 600 windows and the same optimiser. Only the scaling line before `backward()` differed.
