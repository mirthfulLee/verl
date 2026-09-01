# StreamOPD benchmark

| Total tokens | Mode | Microbatch | Step (s) | Tokens/s | Actor peak (GiB) | vs sync | vs colocate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | `streamopd-posthoc-fixed` | 32 | 154.74 | 872.46 | 50.06 | - | - |
| 4096 | `streamopd-posthoc-fixed-wide` | 32 | 160.24 | 842.35 | 42.47 | - | - |
| 4096 | `streamopd-posthoc-legacy` | 32 | 158.36 | 850.20 | 53.81 | - | - |
