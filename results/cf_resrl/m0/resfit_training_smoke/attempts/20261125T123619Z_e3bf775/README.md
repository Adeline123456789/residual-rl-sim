# Headless ResFiT checkpoint evaluation regression

This local RTX 3090 smoke used the same pinned Square trainer, local final ACT
checkpoint, and ResFiT runtime as the failed formal attempt. It tested the
launcher change `MPLBACKEND=Agg` on base commit `e3bf775`, with two train
episodes, 32 fresh warm-up environment steps, and 608 main steps. Video and
Q-trajectory plotting were enabled.

Both fixed-step evaluations finished and wrote video, a plot, and structured
episode records. The run continued for 608 main steps after the first plot,
exceeding the 500 durably recorded main steps in the failed formal attempt.
The smoke exited 0. Its 640-step action ledger and checkpoint evaluations are
included here. Checkpoint weights and videos stay at their original local
paths. The formal run keeps its original 300,000-step budget and evaluation
configuration.
