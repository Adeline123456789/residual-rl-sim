# Square ResFiT 300k interrupted at step 235,800

The user reported a host freeze and forced power-off. After the reboot, the launcher and trainer were gone. The original launcher status still says `RUNNING` because the process could not write an exit status. The last recorded training update and the end of the contiguous action ledger are both step 235,800. The precise cause of the host freeze is unknown.

The last fixed checkpoint is at 230,000 environment steps. It and all 23 evaluation JSON files were checked against the original manifest. The step-230,000 evaluation completed 49 of 50 episodes successfully. This checkpoint omits the online replay buffer at step 230,000 and the live simulation state. The on-disk online cache contains the warm-up data. A formal retry starts with a fresh replay buffer and retains the original seed, dataset split, local ACT base, and 10,000 warm-up plus 290,000 main environment steps.

The three registered overlays were reversed after verifying the old processes had exited. `overlay_recovery.txt` records the cleanup. The original models, replay caches, full action ledger, plots and videos remain in the local attempt. `interruption.json` records the local ledger checksum and the verified final available checkpoint checksum.
