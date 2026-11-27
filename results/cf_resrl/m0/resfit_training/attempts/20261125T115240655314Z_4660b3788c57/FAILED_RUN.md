# Square ResFiT 300k interrupted at step 10,500

The first full-budget attempt used launch commit `4660b3788c574eed7be9b55b90245b8ecbe22013`, the fixed local ACT weights, and seed 20261124. It exited with code 134 on 2026-11-25 at 20:18:50 Asia/Shanghai. The launcher reported successful removal of all three temporary overlays. See `status.json`, `resolved_launcher.json`, and the resolved training configuration.

The action ledger contains contiguous global steps 1–10,500, comprising 10,000 environment warm-up steps and 500 main training steps. Critic-only warm-up completed. The saved step-10,000 checkpoint produced 31 successes in 50 completed Square evaluation episodes (62%), with 12,808 evaluation environment steps in 87.19 seconds. The evaluation JSON binds the episode records to that checkpoint SHA256. The checkpoint is evaluation-loadable and omits replay buffers and live environment state, so it cannot resume the full-budget run.

After saving the evaluation video and Q-trajectory plot, the trainer logged updates through step 10,500. The first failure was in Tkinter destructors with `RuntimeError: main thread is not in main loop`, followed by `Tcl_AsyncDelete: async handler deleted by the wrong thread`. The Gymnasium worker EOF and broken-pipe errors appeared after the Tcl failure. See `stdout_stderr.log`.

`SHA256SUMS` records the settled evidence. The launcher’s recorded stdout SHA256 matches the first 467,804 bytes of the 468,065-byte log. The last 261 bytes are the multiprocessing resource-tracker warning appended after the launcher captured its checksum.

Local binary outputs remain in the original attempt.

| Output | Bytes | SHA256 |
| --- | ---: | --- |
| Step-10,000 ResFiT checkpoint | 401,512,038 | `1d14d496994b8a8d7e438c4d8963369310e83c83db1997c5862d645cd556b9da` |
| Step-10,000 evaluation MP4 | 7,125,046 | `27382e1349dd6dea96f9c20c415f9400fc250009b89397dab115b5aec62104fe` |
| Step-10,000 Q-trajectory PNG | 770,949 | `54b0bd157df76caeb88b50da126b71c5ca3e196c762f7c133f5b44d764d456da` |
