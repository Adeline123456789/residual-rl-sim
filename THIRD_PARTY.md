# Third-party sources

CF-ResRL keeps upstream projects as pinned Git submodules. Their original
licenses and notices remain authoritative; this repository does not relicense
their code.

| Project | Source | Revision | License | Use |
|---|---|---|---|---|
| ResFiT | <https://github.com/amazon-far/residual-offpolicy-rl> | `66262b06a0caf159fb001504949b7b48ccda5f64` | CC-BY-NC-4.0 | Official ACT and residual TD3 implementation used as the frozen baseline. |
| robosuite | <https://github.com/ARISE-Initiative/robosuite> | `77a4751233c29456a5381209e30dd0dbf39a6557` | MIT | Square simulator and controller implementation. |

License texts and upstream notices are stored in each submodule, including
`third_party/resfit/LICENSE`, `third_party/resfit/THIRD-PARTY`, and
`third_party/robosuite/LICENSE`.
