**My MFU and where the rest goes.**

- At the main config (32 x 256 x 2 = 16,384 tokens per step, bf16 autocast, eager), the loop does 351,964 tok/s. That is 6N = 64.5 MFLOP per token, so 22.7 TFLOP/s achieved, **32.4% of the A10G's 70 TF dense bf16 peak**.
  Counting attention FLOPs and dropping the `wpe` lookup gives 35.7%. Measured per step inside the real training run, with all its logging syncs, it is 345,259 tok/s.
- Had I used the A10 datasheet's 125 TF, I would have reported 18.2% for the same run.
- The GPU is never idle: kernel time per step (47.4 ms under the profiler) matches wall time (46.6 ms). Python, data loading (1%: A vs B) and launch overhead are *not* what's costing me at this batch size.
- The step splits roughly in half. GEMMs take 22.3 ms and run at about 47 TF (67% of peak). Their inner dimension is only 384, and the 65-wide `lm_head` manages 17%.
  The other ~24 ms goes to work that does no counted FLOPs: elementwise kernels (13.8 ms, including 2.5 ms of copies for the autocast casts), flash attention (4.3), LayerNorm (3.8) and fused AdamW (2.7).
  0.48 of the time x 0.67 efficiency ≈ 0.32, which is the measured MFU.
- Two levers, measured: `torch.compile` (which fuses elementwise chains and LayerNorm; I did not profile the compiled step) reaches 37.6%. A 128-sequence micro-batch reaches 36.0%. Going the other way, 8 sequences drops to 19.3%.
  So the distance to 40% here is mostly the memory-bound, non-matmul half of each step at C = 384, then small-K GEMMs. On a model this narrow I'd expect it to stay below 40% without fused kernels.
- Memory is nowhere near binding (1.31 GiB peak of 22 GiB). That is the opposite of the large-model case, where memory is the constraint that pushes MFU down.
