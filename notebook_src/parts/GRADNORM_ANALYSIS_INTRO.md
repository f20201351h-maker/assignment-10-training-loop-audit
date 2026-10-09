### Looking for a step where the norm moved first

What the trace offers, before any rule: the norm starts far above the clip threshold on step 1, falls below 1 within ~30 steps, and later
drifts *up* again while the training loss keeps falling. The held-out probe loss bottoms out and then rises: the model starts memorising the 1M-character
training set. So the candidate is the turn into overfitting: did the norm turn before the held-out loss did?

Single-step spikes are the other obvious candidates. I did not analyse them; the turn into overfitting is the larger, slower movement and the one a rule can be tested on.

The rule, chosen after looking at a pilot run of this exact configuration (same seeds, so not an out-of-sample test) and applied unchanged here:

1. **Retrospective turning point.** Centred rolling median of each series (several window sizes), argmin after step 300 (past warm-up and the early plateau).
2. **Causal alarm** (uses only past steps, so it is something a dashboard could actually do): trailing rolling median; alarm on the first step after 300 where it
   exceeds its own running minimum by k standard errors (noise from the MAD of the residual around the smoothed curve). Same rule for grad norm and probe loss.

The event is the grad-norm alarm at w=101, k=3; the other (w, k) rows are there so the choice can be judged.
