# On-Policy Distillation

This trainer jointly trains a student model with policy-gradient on-policy rollouts and a distillation loss against a frozen teacher model served by a separate Ray cluster. Compared to pure SFT from teacher generations, on-policy distillation typically closes more of the teacher/student gap at the same compute budget.

## Canonical Scripts

| Script                          | Teachers | Modality   | Infer | Train    | Platform |
|---------------------------------|----------|------------|-------|----------|----------|
| `run_qwen3_8b_fsdp.sh`          | single   | text       | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_megatron.sh`      | single   | text       | vLLM  | Megatron | NVIDIA   |
| `run_qwen3_vl_8b_fsdp.sh`       | single   | VL         | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_mopd_fsdp.sh`     | multi    | text + VL  | vLLM  | FSDP     | NVIDIA   |

Override `STUDENT_MODEL` and `TEACHER_MODEL` via env vars to swap model pairs in
the single-teacher scripts. The MOPD script exposes per-teacher overrides.

## Experimental StreamOPD-KV

`run_qwen3_streamopd_kv_fsdp.sh` runs strict direct `forward_kl_topk` distillation using the V1 StreamOPD-KV trainer.
It streams committed student tokens to one Teacher and reuses exported Rollout KV for reverse training.
Set `DATASET` to a training parquet file and use `STUDENT_MODEL` / `TEACHER_MODEL` for local paths or Hugging Face IDs.
See the [StreamOPD-KV guide](../../verl/experimental/streamopd_kv/README.md) for supported placements, dependencies,
configuration, and tests. The benchmark comparison controls live in `benchmarks/streamopd_kv`.

## Key Flags

- `distillation.enabled=True`
- `distillation.teacher_models.teacher_model.model_path=<HF path>` (single-teacher)
- `+distillation.teacher_models.<name>.{key,model_path,num_replicas,inference.*}` (multi-teacher)
- `distillation.distillation_loss.loss_mode={k1, k3, forward_kl_topk, ...}`
- `distillation.distillation_loss.use_policy_gradient=True|False`
- `distillation.distillation_loss.topk=64`


## Experimental StreamOPD-CF

StreamOPD-CF (chunked forward) uses independent Trainer,
Rollout and Teacher GPU pools. Run
[`run_qwen3_streamopd_cf_fsdp.sh`](run_qwen3_streamopd_cf_fsdp.sh). This strategy
streams committed rollout tokens into concurrent Teacher prefill and ascending
Trainer forwards. The final input chunk, loss and one backward per microbatch
wait for complete trajectories and Teacher supervision; each policy batch has
one optimizer update. See the
[StreamOPD-CF implementation guide](../../verl/experimental/streamopd_cf/README.md)
and [matched benchmark instructions](../../benchmarks/streamopd_cf/README.md).

The recommended trainer modes are `streamopd_cf` and `streamopd_kv`.
