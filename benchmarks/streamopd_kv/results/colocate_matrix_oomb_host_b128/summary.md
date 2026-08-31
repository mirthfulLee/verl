# StreamOPD-colocate benchmark

| Total tokens | Mode | Microbatch | Step (s) | Tokens/s | Actor peak (GiB) | vs sync | vs colocate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | `streamopd-colocate` | 16 | 214.16 | 472.81 | 29.45 | 1.027x | 0.873x |
| 4096 | `streamopd-colocate` | 32 | 224.42 | 451.06 | 21.02 | 0.980x | 0.833x |
| 4096 | `verl-colocate-opd` | 32 | 186.97 | 539.24 | 20.34 | 1.176x | 1.000x |
| 4096 | `verl-sync-opd` | 32 | 219.97 | 459.83 | 23.64 | 1.000x | 0.850x |
| 8192 | `streamopd-colocate` | 16 | 546.38 | 394.04 | 36.32 | 0.809x | 0.618x |
| 8192 | `streamopd-colocate` | 32 | 561.36 | 380.15 | 27.89 | 0.788x | 0.601x |
| 8192 | `verl-colocate-opd` | 32 | 337.47 | 627.29 | 41.42 | 1.310x | 1.000x |
| 8192 | `verl-sync-opd` | 32 | 442.09 | 485.86 | 54.36 | 1.000x | 0.763x |
