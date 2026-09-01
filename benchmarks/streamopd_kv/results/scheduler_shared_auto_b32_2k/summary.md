# StreamOPD benchmark

| Total tokens | Mode | Microbatch | Step (s) | Tokens/s | Actor peak (GiB) | vs sync | vs colocate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | `streamopd-adaptive` | 16 | 26.63 | 824.34 | 34.75 | - | - |
| 2048 | `streamopd-teacher-then-train` | 16 | 33.14 | 662.43 | 34.75 | - | - |
