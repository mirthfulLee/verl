# StreamOPD-colocate benchmark

| Total tokens | Mode | Microbatch | Step (s) | Tokens/s | Actor peak (GiB) | vs sync | vs colocate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | `streamopd-colocate` | 16 | 135.75 | 741.46 | 49.88 | 1.620x | 1.377x |
| 4096 | `streamopd-colocate` | 32 | 134.43 | 750.51 | 49.91 | 1.636x | 1.391x |
| 4096 | `verl-colocate-opd` | - | 186.97 | 539.24 | 20.34 | 1.176x | 1.000x |
| 4096 | `verl-sync-opd` | - | 219.97 | 459.83 | 23.64 | 1.000x | 0.850x |
| 8192 | `streamopd-colocate` | 16 | 296.86 | 711.17 | 35.31 | 1.489x | 1.137x |
| 8192 | `streamopd-colocate` | 32 | 295.60 | 702.18 | 35.34 | 1.496x | 1.142x |
| 8192 | `verl-colocate-opd` | - | 337.47 | 627.29 | 41.42 | 1.310x | 1.000x |
| 8192 | `verl-sync-opd` | - | 442.09 | 485.86 | 54.36 | 1.000x | 0.763x |
