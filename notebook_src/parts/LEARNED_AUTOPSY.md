**What I found inside the model.**

- One micro-batch is `B x T = 32 x 256 = 8,192` positions. Every block keeps the residual stream at `[32, 256, 384]`; only two things ever change width:
  attention widens to `3C = 1152` for the fused Q/K/V projection and then splits C into `H x D = 6 x 64` heads (`[32, 6, 256, 64]`), and the MLP widens to `4C = 1536`.
- The `[32, 6, 256, 256]` score matrix exists only in my traced step. The training path uses fused SDPA (flash) and never builds it; the two paths agree to 7.8e-3 in bf16.
- 10,745,088 parameters, all trainable. MLP 65.9%, attention 32.9%, embeddings 1.1%. A naive sum over `named_parameters(remove_duplicate=False)` gives 10,770,048,
  because `lm_head.weight` *is* `transformer.wte.weight`. nanoGPT prints 10.65M because it leaves out `wpe`.
- Under bf16 autocast, the outputs of the Linear layers are bf16. LayerNorm outputs, the residual stream and the weights stay fp32, and I compute the loss in fp32.
- At this batch size the activations kept for backward (696 MB) are about 4x the 16-bytes-per-weight training state (164 MB).
- Initial loss is 4.296, a little above ln 65 = 4.174, because the initial logits are small but not zero.
