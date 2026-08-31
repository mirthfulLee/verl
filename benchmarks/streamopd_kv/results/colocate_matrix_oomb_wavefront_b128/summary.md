# StreamOPD-colocate benchmark

| Total tokens | Mode | Microbatch | Step (s) | Tokens/s | Actor peak (GiB) | vs sync | vs colocate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | `streamopd-colocate` | 16 | 241.18 | 419.83 | 55.19 | 0.912x | 0.775x |
| 4096 | `streamopd-colocate` | 32 | 237.36 | 425.93 | 35.99 | 0.927x | 0.788x |
| 4096 | `verl-colocate-opd` | 32 | 186.97 | 539.24 | 20.34 | 1.176x | 1.000x |
| 4096 | `verl-sync-opd` | 32 | 219.97 | 459.83 | 23.64 | 1.000x | 0.850x |
| 8192 | `streamopd-colocate` | 16 | 572.42 | 373.69 | 37.55 | 0.772x | 0.590x |
| 8192 | `streamopd-colocate` | 32 | 550.11 | 383.10 | 37.98 | 0.804x | 0.613x |
| 8192 | `verl-colocate-opd` | 32 | 337.47 | 627.29 | 41.42 | 1.310x | 1.000x |
| 8192 | `verl-sync-opd` | 32 | 442.09 | 485.86 | 54.36 | 1.000x | 0.763x |
