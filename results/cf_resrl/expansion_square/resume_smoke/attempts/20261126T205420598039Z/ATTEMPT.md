# Real Square training-stop and restore witness

The first RTX 3090 process committed a durable full state at environment step 96 and was intentionally terminated with SIGTERM. A separate process restored at step 96 and added 64 interactions. The resulting ledger contains steps 1 through 160 once, and its first 96 records match the saved prefix byte for byte.

The five fixed evaluations at steps 32, 64, 96, 128 and 160 kept their recorded SHA-256 values. The latest three resume generations, steps 96, 128 and 160, remained on disk. Both isolated upstream worktrees were removed after the smoke. Full state, cached transitions, media, evaluation files, and model binaries remain in the local attempt. The shorter checkpoint interval and two training source episodes are specific to this witness. Formal X1 configurations retain one full local state every 100,000 steps with at most three generations and an unchanged 300,000-step budget.
