# StreamOPD benchmark

| Total tokens | Mode | Microbatch | Step (s) | Tokens/s | Actor peak (GiB) | vs sync | vs colocate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | `streamopd` | 32 | 98.98 | 1023.43 | 53.80 | - | - |
| 4096 | `streamopd-posthoc` | 32 | 109.71 | 922.53 | 53.80 | - | - |
