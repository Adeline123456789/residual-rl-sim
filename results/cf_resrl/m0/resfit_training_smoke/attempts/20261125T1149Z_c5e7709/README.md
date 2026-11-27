# ResFiT Square runtime smoke

On the local RTX 3090, the revised overlay ran the real pinned Square trainer
against the fixed ACT 50k checkpoint. This isolated smoke used two registered
train episodes and 16 warm-up plus 16 main environment steps. It exercised
offline AV1 data, prioritized replay, critic and actor updates, and both
fixed-step checkpoint evaluations. The ledger contains 32 training steps.
The two evaluation JSON files contain four episode records tied to the exact
checkpoint hashes. The training budget for the formal run is 300,000 steps.

The runtime is an offline clone of `cf-resrl-m0` named `cf-resrl-resfit`, extended
with `configs/cf_resrl/m0_resfit_runtime_requirements.txt` from PyPI. A seeded
native TorchRL sampler and CUDA matrix multiply passed with Torch
`2.7.1+cu128`, TorchRL `0.9.2`, and TensorDict `0.9.1`. The ACT environment
was left intact. The smoke checkpoint binaries are retained locally at the
paths in the manifest and are not used as experiment inputs.
