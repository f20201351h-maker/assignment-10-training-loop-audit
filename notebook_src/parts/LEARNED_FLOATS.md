**The bits of 0.1**, from 0.1 = 1.1001 1001 1001..._2 x 2^-4:

| format | sign | exponent (bias) | mantissa kept | rounding | bits | hex | value | rel. error |
|---|---|---|---|---|---|---|---|---|
| fp32 | 0 | 01111011 (127) | 23 bits | up | `0 01111011 10011001100110011001101` | 0x3DCCCCCD | 0.100000001490116 | 1.5e-8 |
| bf16 | 0 | 01111011 (127) | 7 bits | up | `0 01111011 1001101` | 0x3DCD | 0.10009765625 | 9.8e-4 |
| fp8 E4M3 (OCP fn) | 0 | 0011 (7) | 3 bits | up | `0 0011 101` | 0x1D | 0.1015625 | 1.6e-2 |

In all three the dropped bits start `1100...` (0.8 ULP), so all three round up. fp16 (shown as a contrast) cuts the pattern elsewhere and rounds *down*, to 0x2E66.

**Which I'd train in, here: bf16 autocast, with fp32 master weights and fp32 AdamW state.**

- On this A10G, bf16 ran 1.96x faster than fp32 (343k vs 175k tok/s). Over 300 identical steps the two loss curves stayed within 0.04 of each other and ended at 1.869 vs 1.867.
- fp8 isn't an option on this card. `torch._scaled_mm` refuses to run on compute capability 8.6, so fp8 here would only be a storage format.
- fp16 would need loss scaling: its smallest normal number is 6.1e-5, whereas bf16 has fp32's exponent and reaches 1.2e-38.
- The one place bf16 is clearly wrong is the weights themselves. 0.1 + 1e-4 is still 0.1 in bf16, and a thousand such updates leave 0.100098 instead of 0.2. So the matmuls run in bf16, but the weights and optimiser state that accumulate small updates stay fp32.
  E4M3 is worse: it flushes a 1e-4 gradient to exactly 0 unless something scales it first.

This is a choice for this GPU and this model, not a general ranking. On Hopper or Blackwell, an fp8 recipe with scaling is the thing to try.
