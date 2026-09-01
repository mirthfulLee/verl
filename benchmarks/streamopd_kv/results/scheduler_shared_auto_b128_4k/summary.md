# StreamOPD benchmark

| Total tokens | Mode | Microbatch | Step (s) | Tokens/s | Actor peak (GiB) | vs sync | vs colocate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | `streamopd-adaptive` | 16 | 113.77 | 993.55 | 50.06 | - | - |
| 4096 | `streamopd-teacher-then-train` | 16 | 122.51 | 922.71 | 50.06 | - | - |
