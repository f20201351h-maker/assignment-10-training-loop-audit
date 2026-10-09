**What this shows, and what it doesn't.**

- The norm was 14.87 at step 1. The first 26 steps were all clipped, 30 of the first 50, and 159 of 1500 overall. Clipping from step one is not a formality here.
- The event: the trailing-median grad norm bottomed at 0.582 (step 762), and the causal alarm fired at **step 830**, when it had risen 2.4%.
  At that step the held-out probe loss was at its best value so far (1.462) and the training loss was still falling (1.185). The probe-loss alarm fired at **step 904, 74 steps later**.
  The full validation set, scored every 50 steps, bottomed at step 850. Looking back with centred windows of 25 to 151 steps, the norm turned 60 to 135 steps before the probe loss.
- The rise sat in the transformer blocks (h.0 to h.5 up 3.8 to 6.4%), while the embedding and final LayerNorm gradients were flat or slightly down.
- This happened in a normal run; nothing was injected, so I did not need a stress test.

What I can't claim: the move is small and only shows up after heavy smoothing. With short windows the same rule fires at steps 316, 351 and 400 on bumps that never reached the loss.
The learning rate is also decaying (5.2e-4 at the event), so part of the rise could be the schedule rather than overfitting.
This is one natural instance of the norm leading the held-out loss into overfitting, not evidence that grad norm predicts loss in general.
The window and threshold were picked after looking at the pilot trace, and this run has the same seeds, so it is not an out-of-sample test of the rule.
