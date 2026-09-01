# StreamOPD benchmark

| Total tokens | Mode | Microbatch | Step (s) | Tokens/s | Actor peak (GiB) | vs sync | vs colocate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | `streamopd-adaptive` | 16 | 19.12 | 612.56 | 34.82 | - | - |
| 1024 | `streamopd-teacher-then-train` | 16 | 20.69 | 566.20 | 34.82 | - | - |
