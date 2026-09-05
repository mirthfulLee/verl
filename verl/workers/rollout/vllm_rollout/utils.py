# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ctypes
import dataclasses
import functools
import json
import logging
import os
import platform
import signal
import threading
from collections.abc import Mapping
from types import MethodType
from typing import Any, Literal, Optional, get_args

import torch
from packaging import version
from vllm.outputs import RequestOutput

from verl.utils.device import get_device_name, get_torch_device, is_npu_available
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, resolve_weight_name
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_quant_utils import apply_vllm_quant_patches, is_fp8_model, load_quanted_weights
from verl.workers.rollout.vllm_rollout.weight_update_utils import apply_buffer_updates, split_buffer_updates

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"
# The auto StreamOPD profile caps Teacher input fragments at 1024 tokens. Keep
# one fragment in one LM-head tile while retaining the runtime memory fallback
# below for explicit larger fragments and non-exclusive vLLM deployments.
_STREAMING_PROMPT_LOGPROBS_CHUNK_SIZE = 1024
_CUDA_ALLOCATOR_SEGMENT_BYTES = 2 * 1024**2
_VLLM_REDUNDANCY_BUFFER_BYTES = 150 * 1024**2
_EXCLUSIVE_GPU_MEMORY_KEY = "verl_exclusive_gpu_memory"
_STREAMING_TEACHER_MEMORY_KEY = "verl_streaming_teacher_logprobs"

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}


def _request_exclusive_gpu_memory(init_snapshot, cache_config) -> int:
    """Use all memory free after the worker initializes CUDA and NCCL."""

    free_memory = int(init_snapshot.free_memory)
    total_memory = int(init_snapshot.total_memory)
    if free_memory < 1 or total_memory < free_memory:
        raise RuntimeError("vLLM reported invalid device memory for exclusive-pool sizing")
    cache_config._verl_exclusive_initial_free_bytes = free_memory
    return free_memory


def _streamopd_unprofiled_workspace_bytes(vllm_config) -> int:
    """Return deterministic late allocations absent from vLLM's profile."""

    workspace_bytes = 0
    transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    additional_config = getattr(vllm_config, "additional_config", {}) or {}
    has_streamopd_connector = transfer_config is not None and transfer_config.kv_connector == "StreamOPDKVConnector"
    has_exclusive_pool = additional_config.get(_EXCLUSIVE_GPU_MEMORY_KEY, False)
    has_streaming_teacher = additional_config.get(_STREAMING_TEACHER_MEMORY_KEY, False)
    if not has_exclusive_pool and not has_streamopd_connector and not has_streaming_teacher:
        return 0
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    element_size = torch.empty((), dtype=model_config.dtype).element_size()

    def round_allocation(size: int) -> int:
        return (
            (size + _CUDA_ALLOCATOR_SEGMENT_BYTES - 1) // _CUDA_ALLOCATOR_SEGMENT_BYTES * _CUDA_ALLOCATOR_SEGMENT_BYTES
        )

    if has_exclusive_pool:
        max_rows = min(
            int(vllm_config.scheduler_config.max_num_seqs),
            int(vllm_config.scheduler_config.max_num_batched_tokens),
        )
        vocab_size = int(model_config.get_vocab_size())
        padded_vocab_size = (vocab_size + 63) // 64 * 64
        source_logits = 0
        if element_size != 4:
            source_logits = round_allocation(max_rows * padded_vocab_size * element_size)
        fp32_logits = round_allocation(max_rows * padded_vocab_size * 4)
        sort_indices = round_allocation(max_rows * padded_vocab_size * 8)
        sort_masks = 2 * round_allocation(max_rows * padded_vocab_size)
        # vLLM warms the native top-k/top-p sampler after KV allocation. Its
        # live peak contains five FP32 vocab matrices (input, processed
        # log-probabilities, sampling probabilities, exponential samples, and
        # their quotient), int64 sort indices, and two boolean sort masks.
        workspace_bytes += source_logits + 5 * fp32_logits + sort_indices + sort_masks + _CUDA_ALLOCATOR_SEGMENT_BYTES

    if has_streamopd_connector:
        extra = transfer_config.kv_connector_extra_config
        export_strategy = str(extra.get("streamopd_kv_export_strategy", "eos_host"))
        if export_strategy != "eos_host":
            writer_threads = int(extra.get("streamopd_writer_threads", 4))
            chunk_size = int(extra.get("streamopd_kv_chunk_size", 256))
            num_layers = int(model_config.get_num_layers(parallel_config))
            num_kv_heads = int(model_config.get_num_kv_heads(parallel_config))
            head_dim = int(model_config.get_head_size())
            output_bytes = num_layers * chunk_size * 2 * num_kv_heads * head_dim * element_size
            # Gather outputs plus small block-index and layer-order tensors.
            workspace_bytes += writer_threads * round_allocation(output_bytes) + _CUDA_ALLOCATOR_SEGMENT_BYTES

    if has_streaming_teacher:
        max_rows = min(
            int(vllm_config.scheduler_config.max_num_batched_tokens),
            int(model_config.max_model_len),
        )
        logit_rows = min(max_rows, _STREAMING_PROMPT_LOGPROBS_CHUNK_SIZE)
        tp_size = int(parallel_config.tensor_parallel_size)
        vocab_size = int(model_config.get_vocab_size())
        padded_vocab_size = (vocab_size + 63) // 64 * 64
        global_logits = round_allocation(logit_rows * padded_vocab_size * element_size)
        local_logits = round_allocation(logit_rows * (padded_vocab_size // tp_size) * element_size)
        normalized_logits = round_allocation(logit_rows * padded_vocab_size * 4)
        # Local LM-head output, TP-gathered logits, and normalization overlap on
        # asynchronous streams. Keep a second normalization-sized segment so
        # the next chunk never depends on a fragmented cached block.
        workspace_bytes += local_logits + global_logits + 2 * normalized_logits + _CUDA_ALLOCATOR_SEGMENT_BYTES

    return workspace_bytes


def _determine_exclusive_available_memory(worker, original_determine) -> int:
    """Deduct measured late-runtime state from vLLM's profiled KV budget."""

    available_bytes = int(original_determine(worker))
    graph_pool_count = 0
    if not worker.model_config.enforce_eager:
        dispatcher = getattr(worker.model_runner, "cudagraph_dispatcher", None)
        if dispatcher is None:
            raise RuntimeError("exclusive vLLM sizing cannot inspect CUDA graph capture modes")
        graph_pool_count = sum(bool(batch_descriptors) for _, batch_descriptors in dispatcher.get_capture_descs())
    activation_reserve_count = graph_pool_count + 1
    activation_reserve_bytes = activation_reserve_count * int(getattr(worker, "peak_activation_memory", 0))
    connector_reserve_bytes = _streamopd_unprofiled_workspace_bytes(worker.vllm_config)
    # Preserve the same profiling-error allowance used by vLLM when it reports
    # the maximum KV allocation that fits the device. Exclusive sizing starts
    # from post-NCCL free bytes, but is not exempt from that runtime variance.
    reserve_bytes = activation_reserve_bytes + connector_reserve_bytes + _VLLM_REDUNDANCY_BUFFER_BYTES
    if available_bytes <= reserve_bytes:
        raise RuntimeError(
            "exclusive vLLM pool cannot fit its measured graph and connector workspace: "
            f"available={available_bytes}, reserve={reserve_bytes}"
        )
    worker.cache_config._verl_exclusive_activation_reserve_bytes = activation_reserve_bytes
    worker.cache_config._verl_exclusive_activation_reserve_count = activation_reserve_count
    worker.cache_config._verl_exclusive_graph_pool_count = graph_pool_count
    worker.cache_config._verl_exclusive_connector_reserve_bytes = connector_reserve_bytes
    worker.cache_config._verl_exclusive_redundancy_reserve_bytes = _VLLM_REDUNDANCY_BUFFER_BYTES
    return available_bytes - reserve_bytes


def _configure_exclusive_gpu_memory(vllm_config) -> bool:
    additional_config = getattr(vllm_config, "additional_config", {}) or {}
    if not additional_config.get(_EXCLUSIVE_GPU_MEMORY_KEY, False):
        return False
    from vllm.v1.worker import gpu_worker

    gpu_worker.request_memory = _request_exclusive_gpu_memory
    original_determine = gpu_worker.Worker.determine_available_memory
    if not getattr(original_determine, "_verl_exclusive_gpu_memory", False):

        def determine_available_memory(worker) -> int:
            return _determine_exclusive_available_memory(worker, original_determine)

        determine_available_memory._verl_exclusive_gpu_memory = True
        gpu_worker.Worker.determine_available_memory = determine_available_memory
    return True


def _prompt_logprob_chunk_rows(*, free_bytes: int, vocab_size: int, requested_rows: int) -> int:
    """Size normalization from its FP32 output and transient rank mask."""

    if min(free_bytes, vocab_size, requested_rows) < 1:
        raise ValueError("prompt logprob workspace dimensions must be positive")
    bytes_per_row = vocab_size * (
        torch.empty((), dtype=torch.float32).element_size() + torch.empty((), dtype=torch.bool).element_size()
    )
    usable_bytes = max(0, free_bytes - 2 * _CUDA_ALLOCATOR_SEGMENT_BYTES)
    return max(1, min(requested_rows, usable_bytes // bytes_per_row))


def _gather_prompt_logprobs_in_chunks(sampler, logits, num_logprobs, token_ids, *, chunk_size: int):
    """Normalize and gather prompt scores in bounded row chunks."""

    if chunk_size < 1:
        raise ValueError("prompt logprob chunk size must be positive")
    device_module = get_torch_device()
    is_accelerator = logits.device.type == get_device_name()
    chunks = []
    start = 0
    while start < logits.shape[0]:
        rows = min(chunk_size, logits.shape[0] - start)
        if is_accelerator:
            free_bytes, _ = device_module.mem_get_info(logits.device)
            rows = _prompt_logprob_chunk_rows(
                free_bytes=free_bytes,
                vocab_size=logits.shape[-1],
                requested_rows=rows,
            )
        while True:
            end = start + rows
            try:
                chunk_logits = logits[start:end]
                chunk = sampler.gather_logprobs(
                    sampler.compute_logprobs(chunk_logits),
                    num_logprobs,
                    token_ids[start:end],
                )
                break
            except torch.OutOfMemoryError:
                if not is_accelerator or rows == 1:
                    raise
                rows = max(1, rows // 2)
                device_module.empty_cache()
        chunks.append(chunk)
        start = end
    if not chunks:
        raise ValueError("prompt logprob input must contain at least one row")
    output_type = type(chunks[0])
    return output_type(*(torch.cat([chunk[field] for chunk in chunks], dim=0) for field in range(3)))


def _compute_prompt_logprobs_in_chunks(model, sampler, hidden_states, num_logprobs, token_ids, *, chunk_size: int):
    """Bound LM-head, TP-gather, and normalization memory to one chunk."""

    if chunk_size < 1:
        raise ValueError("prompt logprob chunk size must be positive")
    chunks = []
    for start in range(0, hidden_states.shape[0], chunk_size):
        end = min(start + chunk_size, hidden_states.shape[0])
        logits = model.compute_logits(hidden_states[start:end])
        chunks.append(
            _gather_prompt_logprobs_in_chunks(
                sampler,
                logits,
                num_logprobs,
                token_ids[start:end],
                chunk_size=chunk_size,
            )
        )
        del logits
    if not chunks:
        raise ValueError("prompt logprob input must contain at least one row")
    output_type = type(chunks[0])
    return output_type(*(torch.cat([chunk[field] for chunk in chunks], dim=0) for field in range(3)))


def _resolve_vllm_weight_sync_local_rank(worker_local_rank: int, parallel_config: Any) -> int:
    worker_local_rank = int(worker_local_rank)
    if parallel_config is None:
        return worker_local_rank

    tp_size = max(int(getattr(parallel_config, "tensor_parallel_size", 1) or 1), 1)
    dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
    dp_local_size = int(getattr(parallel_config, "data_parallel_size_local", 1) or 1)
    if dp_size <= 1 and dp_local_size <= 1:
        return worker_local_rank

    dp_local_rank = getattr(parallel_config, "data_parallel_rank_local", None)
    if dp_local_rank is None:
        dp_rank = getattr(parallel_config, "data_parallel_rank", None)
        if dp_rank is None:
            dp_rank = getattr(parallel_config, "data_parallel_index", None)
        if dp_rank is not None and dp_local_size > 0:
            dp_local_rank = int(dp_rank) % dp_local_size

    if dp_local_rank is None:
        return worker_local_rank

    tp_rank = worker_local_rank % tp_size
    return int(dp_local_rank) * tp_size + tp_rank


def set_death_signal():
    """Kill the current process when the parent process exits."""
    if platform.system() != "Linux":
        return
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, automatically adjusts the `max_lora_rank` to the nearest allowed value.
    The allowed values are retrieved from vLLM's MaxLoRARanks type definition.
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0, get {lora_rank}"

    try:
        from vllm.config.lora import MaxLoRARanks
    except Exception:
        # FIXME: migrate vllm version https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py#L25
        MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]

    vllm_max_lora_ranks = sorted(get_args(MaxLoRARanks))
    if lora_rank > vllm_max_lora_ranks[-1]:
        raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")

    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank


# https://github.com/vllm-project/vllm/issues/13175
def monkey_patch_compute_logits(model, vocab_size: int, banned_token_ids: Optional[list[int]] = None):
    """Mask the tokens the sampler must never pick.

    Beyond the out-of-vocabulary tail, `banned_token_ids` covers tokens that live *inside* the
    vocabulary yet are still illegal to generate: the vision placeholders, which are meaningless
    unless a real image or video sits behind them. See `get_vision_placeholder_token_ids`.
    """
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        if banned_token_ids:
            logits[..., banned_token_ids] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. Online FP8 quantization
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        if os.environ.get("VERL_FULL_DETERMINISM", "0") == "1":
            from verl.workers.engine.utils import enable_full_determinism

            # VERL_SEED is set by vLLMHttpServer.__init__ only when the
            # rollout config has full_determinism=true.  Worker sub-processes
            # inherit their parent's env, so rollout workers will see it but
            # RM workers (whose parent vLLMHttpServer does not set it) won't.
            # If VERL_SEED is missing, skip — RM doesn't need the determinism
            # patch, only rollout does.
            verl_seed = os.environ.get("VERL_SEED")
            if verl_seed is not None:
                enable_full_determinism(seed=int(verl_seed))

        # 1. patch for Lora
        VLLMHijack.hijack()
        vllm_config = kwargs.get("vllm_config")
        if _configure_exclusive_gpu_memory(vllm_config):
            logger.info("vLLM will size its memory budget from post-NCCL free device memory")
        # 2. patch online fp8 quant. Some models, including DeepSeek-V4, get
        # fp8 from the HF config rather than an explicit rollout quantization arg.
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1" or is_fp8_model(vllm_config):
            apply_vllm_quant_patches()
        # 3. patch QAT (compressed-tensors NVFP4) for dynamic weight loading
        quant_config = getattr(vllm_config, "quant_config", None) if vllm_config else None
        _is_qat_model = getattr(quant_config, "quant_format", None) == "nvfp4-pack-quantized"
        _is_modelopt_qat = type(quant_config).__name__ == "ModelOptNvFp4Config"
        if _is_qat_model:
            from verl.utils.qat import apply_qat_patches

            apply_qat_patches()
            logger.info("Applied QAT (compressed-tensors) patches in vLLM worker subprocess")
        elif _is_modelopt_qat:
            from verl.utils.modelopt import apply_modelopt_nvfp4_patches

            apply_modelopt_nvfp4_patches()
            logger.info("Applied ModelOpt NVFP4 patches in vLLM worker subprocess")

        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        instance = super().__new__(cls)
        instance._is_qat_model = _is_qat_model
        instance._is_modelopt_qat = _is_modelopt_qat
        return instance

    def _get_drafter_model(self):
        """Return the drafter's model object, or None if unavailable."""
        drafter = getattr(self.model_runner, "drafter", None)
        return drafter.model if drafter is not None and hasattr(drafter, "model") else None

    def _get_draft_model_config(self):
        """Return the draft model config from speculative_config, or None."""
        spec = self.model_runner.vllm_config.speculative_config
        return spec.draft_model_config if spec is not None and spec.draft_model_config is not None else None

    def _use_mtp_drafter_weight_sync(self):
        """Return whether the vLLM MTP drafter should receive actor weights."""
        spec = self.model_runner.vllm_config.speculative_config
        return spec is not None and spec.method == "mtp" and self._get_drafter_model() is not None

    def _iter_all_models(self):
        """Yield models that need weight updates.

        Only vLLM MTP drafter sync is supported for now. Independent non-MTP
        draft models are not compatible with actor weight loading through this path.
        """
        yield self.model_runner.model
        if self._use_mtp_drafter_weight_sync():
            yield self._get_drafter_model()

    def _iter_all_models_with_config(self):
        """Yield (model, model_config) for models that need post-processing."""
        yield self.model_runner.model, self.model_runner.vllm_config.model_config
        if self._use_mtp_drafter_weight_sync():
            draft_cfg = self._get_draft_model_config()
            if draft_cfg is not None:
                yield self._get_drafter_model(), draft_cfg

    def monkey_patch_model(self, vocab_size: int, banned_token_ids: Optional[list[int]] = None):
        for model in self._iter_all_models():
            # patch compute_logits to avoid sampling OOV and other illegal tokens
            monkey_patch_compute_logits(model, vocab_size, banned_token_ids)
            # patch weight loader to support MoE model
            patch_vllm_moe_model_weight_loader(model)

    def enable_streaming_prompt_logprobs(self) -> bool:
        """Emit prompt logprobs for every vLLM 0.15.1 StreamingInput update.

        vLLM 0.15.1 keeps the KV cache for resumable requests, but its legacy
        GPU runner removes prompt-logprob bookkeeping after the first input.
        Re-arm that bookkeeping and return only the newly appended rows.  The
        patch is installed only on teacher workers and can be removed once the
        corresponding vLLM behavior is available upstream.
        """
        import vllm

        if version.parse(vllm.__version__) != version.parse("0.15.1"):
            return False

        model_runner = self.model_runner
        if getattr(model_runner, "_get_prompt_logprobs_dict", None) is None:
            logger.warning("vLLM 0.15.1 streaming prompt-logprob patch skipped: unsupported model runner")
            return False

        from vllm.v1.outputs import LogprobsTensors

        emitted_prompt_lens: dict[str, int] = {}

        def get_streaming_prompt_logprobs(runner, hidden_states, num_scheduled_tokens):
            active_request_ids = set(runner.requests)
            for request_id in tuple(emitted_prompt_lens):
                if request_id not in active_request_ids:
                    del emitted_prompt_lens[request_id]

            # A resumable request is scheduled as a cached request after its
            # first input, so the stock runner does not re-register it here.
            for request_id in num_scheduled_tokens:
                request = runner.requests.get(request_id)
                sampling_params = getattr(request, "sampling_params", None)
                num_logprobs = getattr(sampling_params, "prompt_logprobs", None)
                if num_logprobs is not None:
                    runner.num_prompt_logprobs.setdefault(request_id, num_logprobs)

            in_progress = runner.input_batch.in_progress_prompt_logprobs_cpu
            outputs = {}
            completed_requests = []
            for request_id, num_logprobs in runner.num_prompt_logprobs.items():
                num_tokens = num_scheduled_tokens.get(request_id)
                if num_tokens is None:
                    continue
                request = runner.requests[request_id]
                if request.prompt_token_ids is None:
                    continue

                prompt_len = len(request.prompt_token_ids)
                tensors = in_progress.get(request_id)
                if tensors is None:
                    tensors = LogprobsTensors.empty_cpu(prompt_len - 1, num_logprobs + 1)
                    in_progress[request_id] = tensors

                start = request.num_computed_tokens
                first_target = start + 1
                remaining = prompt_len - first_target
                if num_tokens <= remaining:
                    num_logits = num_tokens
                else:
                    num_logits = remaining
                    completed_requests.append(request_id)
                    outputs[request_id] = tensors
                if num_logits <= 0:
                    continue

                request_index = runner.input_batch.req_id_to_index[request_id]
                offset = runner.query_start_loc.np[request_index].item()
                request_hidden_states = hidden_states[offset : offset + num_logits]
                prompt_token_ids = torch.tensor(request.prompt_token_ids, device=runner.device)
                target_ids = prompt_token_ids[first_target : first_target + num_logits]
                token_ids, logprobs, ranks = _compute_prompt_logprobs_in_chunks(
                    runner.model,
                    runner.sampler,
                    request_hidden_states,
                    num_logprobs,
                    target_ids,
                    chunk_size=_STREAMING_PROMPT_LOGPROBS_CHUNK_SIZE,
                )
                output_slice = slice(start, start + num_logits)
                tensors.logprob_token_ids[output_slice].copy_(token_ids, non_blocking=True)
                tensors.logprobs[output_slice].copy_(logprobs, non_blocking=True)
                tensors.selected_token_ranks[output_slice].copy_(ranks, non_blocking=True)

            for request_id in completed_requests:
                del runner.num_prompt_logprobs[request_id]
                del in_progress[request_id]
            if outputs:
                runner._sync_device()

            for request_id, tensors in tuple(outputs.items()):
                if tensors is None:
                    continue
                request = runner.requests[request_id]
                prompt_len = len(request.prompt_token_ids or ())
                previous_prompt_len = emitted_prompt_lens.get(request_id, 0)
                if previous_prompt_len:
                    # Absolute tensor row i scores prompt token i + 1.  The
                    # first token in an appended fragment is already covered
                    # by the prior chunk's sampled-logprob row.
                    start = min(previous_prompt_len, tensors.logprobs.shape[0])
                    outputs[request_id] = type(tensors)(
                        logprob_token_ids=tensors.logprob_token_ids[start:],
                        logprobs=tensors.logprobs[start:],
                        selected_token_ranks=tensors.selected_token_ranks[start:],
                    )
                emitted_prompt_lens[request_id] = prompt_len
            return outputs

        model_runner._get_prompt_logprobs_dict = MethodType(get_streaming_prompt_logprobs, model_runner)
        logger.info("Enabled vLLM 0.15.1 StreamingInput prompt-logprob compatibility patch")
        return True

    def reset_device_memory_stats(self) -> None:
        """Reset per-step allocator peaks inside the vLLM worker process."""

        device_module = get_torch_device()
        device_module.reset_peak_memory_stats(self.device)

    def reset_streamopd_kv_transfer_stats(self) -> bool:
        """Reset StreamOPD connector counters when this worker owns one."""

        from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

        if not has_kv_transfer_group():
            return False
        connector = get_kv_transfer_group()
        reset = getattr(connector, "reset_transfer_stats", None)
        if reset is None:
            return False
        reset()
        return True

    def get_streamopd_kv_transfer_stats(self) -> dict[str, float]:
        """Return StreamOPD connector counters when this worker owns one."""

        from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

        if not has_kv_transfer_group():
            return {}
        connector = get_kv_transfer_group()
        collect = getattr(connector, "get_transfer_stats", None)
        return {} if collect is None else collect()

    def wait_for_streamopd_kv_transfers(self) -> float:
        """Drain StreamOPD exports before a phase-shared vLLM sleeps."""

        from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

        if not has_kv_transfer_group():
            return 0.0
        connector = get_kv_transfer_group()
        wait = getattr(connector, "wait_for_all_exports", None)
        return 0.0 if wait is None else float(wait())

    def get_device_memory_stats(self) -> dict[str, int]:
        """Return current and peak allocator bytes for this vLLM worker."""

        device_module = get_torch_device()
        free_bytes, total_bytes = device_module.mem_get_info(self.device)
        return {
            "allocated_bytes": int(device_module.memory_allocated(self.device)),
            "reserved_bytes": int(device_module.memory_reserved(self.device)),
            "max_allocated_bytes": int(device_module.max_memory_allocated(self.device)),
            "max_reserved_bytes": int(device_module.max_memory_reserved(self.device)),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

    def get_kv_cache_capacity(self) -> dict[str, int]:
        """Return the profiled GPU KV capacity in scheduler-visible tokens."""

        num_blocks = int(self.cache_config.num_gpu_blocks or 0)
        block_size = int(self.cache_config.block_size or 0)
        if num_blocks < 1 or block_size < 1:
            raise RuntimeError("vLLM KV cache capacity is unavailable after engine initialization")
        return {
            "num_gpu_blocks": num_blocks,
            "block_size": block_size,
            "capacity_tokens": num_blocks * block_size,
        }

    def trim_device_memory(self, minimum_free_bytes: int = 0) -> dict[str, int]:
        """Release inactive allocator cache at a scheduler-owned role switch."""

        device_module = get_torch_device()
        free_before, _ = device_module.mem_get_info(self.device)
        if int(free_before) < int(minimum_free_bytes):
            device_module.empty_cache()
        free_after, _ = device_module.mem_get_info(self.device)
        return {
            "freed_bytes": max(0, int(free_after) - int(free_before)),
            "free_before_bytes": int(free_before),
            "free_after_bytes": int(free_after),
        }

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if self.device is None:
            # vLLM workers may leave self.device unset on non-CUDA platforms (e.g. NPU);
            # fall back to the worker's local rank on the current accelerator.
            self.device = torch.device(f"{get_device_name()}:{self.local_rank}")

        # =========================== step 1: prepare for weight loading ===========================
        quant_reload_states = None

        # The engine came up on dummy weights, whose init zeroes integer buffers on
        # ROCm -- including the expert-parallel routing maps, which no weight stream
        # restores. Repair them before the reload so the rollout routes correctly.
        if torch.version.hip is not None:
            from verl.utils.vllm.rocm_vllm_moe_expert_map import restore_moe_expert_maps

            for model in self._iter_all_models():
                restore_moe_expert_maps(model)

        if self._is_qat_model:
            # QAT (compressed-tensors): Prepare for weight loading BEFORE receiving any buckets
            from verl.utils.qat import prepare_qat_for_load_weights

            for model in self._iter_all_models():
                prepare_qat_for_load_weights(model, device=self.device)
            logger.info("QAT: prepare_qat_for_load_weights completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import prepare_modelopt_for_weight_reload

            prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
            logger.info("ModelOpt: prepare_modelopt_for_weight_reload completed")
        elif peft_config and base_sync_done:
            # Remove the old LoRA before the new one arrives (applied after is_last below).
            self.remove_lora(VLLM_LORA_INT_ID)
            logger.info("LoRA adapter sync: remove old lora and prepare new lora")
        elif is_fp8_model(self.model_runner.vllm_config):
            from verl.utils.vllm.vllm_quant_utils import prepare_quanted_weights_for_loading

            quant_reload_states = [
                (model, prepare_quanted_weights_for_loading(model)) for model in self._iter_all_models()
            ]
        else:
            # TODO(wuxibin): not need anymore for newer vllm version.
            for model in self._iter_all_models():
                patch_vllm_moe_model_weight_loader(model)

        # =========================== step 2: receive weights and update ===========================
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        # LoRA adapters need a single complete tensor dict per ``add_lora``, but
        # the bucketed transport may split one across buckets. Accumulate and
        # apply only after ``is_last``; standard base weights load per bucket.
        lora_weights: dict[str, torch.Tensor] | None = {} if (peft_config and base_sync_done) else None

        def on_bucket_received(weights: list[tuple[str, torch.Tensor]], is_last: bool) -> None:
            if lora_weights is not None:
                # Clone: add_lora keeps these past the callback (reused IPC buffer, #6454).
                lora_weights.update((name, tensor.clone()) for name, tensor in weights)
                if not is_last:
                    return
                self._update_weights(
                    list(lora_weights.items()),
                    peft_config=peft_config,
                    base_sync_done=base_sync_done,
                )
                lora_weights.clear()
                return
            self._update_weights(
                weights,
                peft_config=peft_config,
                base_sync_done=base_sync_done,
            )

        receiver.receive_weights(on_bucket_received=on_bucket_received)

        # =========================== step 3: process weights after loading ===========================
        if self._is_qat_model:
            # QAT (compressed-tensors): call process_weights_after_loading AFTER all buckets are received
            from verl.utils.qat import manual_process_weights_after_loading

            for model in self._iter_all_models():
                manual_process_weights_after_loading(model)
            logger.info("QAT: process_weights_after_loading completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import modelopt_process_weights_after_loading

            modelopt_process_weights_after_loading(self.model_runner.model)
            logger.info("ModelOpt QAT: process_weights_after_loading completed")
        elif peft_config and base_sync_done:
            logger.info("LoRA adapter sync, no post-process needed")
        elif is_fp8_model(self.model_runner.vllm_config):
            from verl.utils.vllm.vllm_quant_utils import process_quanted_weights_after_loading

            for model, reload_state in quant_reload_states:
                process_quanted_weights_after_loading(model, reload_state)
        else:
            # Some post-load transforms are non-idempotent; run once after all buckets.
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            for model, model_config in self._iter_all_models_with_config():
                process_weights_after_loading(model, model_config, self.device)

    def _apply_buffer_updates_all_models(self, buffer_updates, main_named_buffers):
        """Apply buffer updates to the main model and any synced MTP drafter.

        The main model (yielded first) reuses the prebuilt ``named_buffers`` map;
        the drafter builds its own. Returns buffers applied to the main model.
        """
        models = list(self._iter_all_models())
        loaded = apply_buffer_updates(models[0], buffer_updates, named_buffers=main_named_buffers)
        for model in models[1:]:
            apply_buffer_updates(model, buffer_updates)
        return loaded

    def _update_weights(
        self,
        weights: list[tuple[str, torch.Tensor]],
        peft_config: dict,
        base_sync_done: bool,
    ):
        if peft_config and base_sync_done:
            # Clone out of the receiver's reused IPC bucket buffer: add_lora keeps these tensors
            # past this callback, so views into the freed/overwritten buffer crash later (#6454).
            weights = {name: tensor.clone() for name, tensor in weights}
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            param_updates, buffer_updates, named_buffers = split_buffer_updates(self.model_runner.model, weights)
            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(self.model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {self.model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(param_updates, self.model_runner) if param_updates else []
                # Keep the draft model in sync when present.
                if self._use_mtp_drafter_weight_sync() and param_updates:
                    load_quanted_weights(param_updates, self.model_runner, is_drafter=True)
                loaded_buffers = self._apply_buffer_updates_all_models(buffer_updates, named_buffers)
                logger.info(
                    f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}, loaded_buffers: {loaded_buffers}"
                )
            else:
                if param_updates:
                    for model in self._iter_all_models():
                        if peft_config is None:
                            model.load_weights(param_updates)
                        else:
                            names = {n for n, _ in model.named_parameters(remove_duplicate=False)}
                            names.update(n for n, _ in model.named_buffers())
                            model.load_weights((resolve_weight_name(model, n, names), t) for n, t in param_updates)
                loaded_buffers = self._apply_buffer_updates_all_models(buffer_updates, named_buffers)
                logger.info(
                    f"Loading standard weights (non-FP8, async), "
                    f"loaded_params: {len(param_updates)}, loaded_buffers: {loaded_buffers}"
                )

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication.

        Uses Ray job id + replica_rank + rollout-local rank to match the sender
        side and avoid cross-job collisions on shared hosts.
        In PD mode, each engine actor's local ranks start at 0; the optional
        VERL_ZMQ_BASE_TRAINER_RANK offset maps them back to trainer ranks.
        """
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        vllm_config = getattr(self.model_runner, "vllm_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        local_rank = _resolve_vllm_weight_sync_local_rank(self.local_rank, parallel_config)
        trainer_rank_base = os.environ.get("VERL_ZMQ_BASE_TRAINER_RANK")
        trainer_rank = int(trainer_rank_base) + local_rank if trainer_rank_base is not None else local_rank
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{trainer_rank}.sock"


class SuppressSignalInThread:
    def __enter__(self):
        self.original_signal = signal.signal

        def no_op_signal(sig, action):
            if threading.current_thread() is not threading.main_thread():
                print(f"Ignored signal {sig} in thread {threading.current_thread().name}")
                return
            return self.original_signal(sig, action)

        signal.signal = no_op_signal
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal = self.original_signal


@functools.lru_cache(maxsize=1)
def _optional_bool_vllm_args() -> set[str]:
    """Return the names of vLLM `AsyncEngineArgs` fields typed exactly `bool | None`.

    For such fields an omitted flag leaves the None default, which vLLM can
    resolve to True at engine-config time (e.g. `enable_prefix_caching`), so
    an explicit False must be serialized as `--no-<flag>` instead of being
    dropped.
    """
    from vllm.engine.arg_utils import AsyncEngineArgs

    return {f.name for f in dataclasses.fields(AsyncEngineArgs) if set(get_args(f.type)) == {bool, type(None)}}


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    Convert a config dictionary to CLI arguments for vLLM server.

    Handles different value types appropriately:
    - None: skipped
    - bool True: adds '--key'
    - bool False: adds '--no-key' for Optional[bool] engine args (whose None
      default resolves to True), otherwise skipped
    - list: expands to '--key item1 item2 ...'
    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
    - dict: JSON serialized
    - other: string converted

    Args:
        config: Dictionary of configuration key-value pairs

    Returns:
        List of CLI argument strings
    """
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
            elif k.replace("-", "_") in _optional_bool_vllm_args():
                # Absent flag resolves to True at engine-config time.
                cli_args.append(f"--no-{k}")
        elif isinstance(v, list):
            if not v:
                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
                continue
            # Lists need to be expanded as multiple separate arguments
            # e.g., --cuda-graph-sizes 1 2 4 8 becomes ['--cuda-graph-sizes', '1', '2', '4', '8']
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            # Use json.dumps for dict to ensure valid JSON format
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args


def build_mtp_speculative_config(
    method: str, num_speculative_tokens: int, engine_speculative_config: Any = None
) -> dict[str, Any]:
    """Build vLLM's MTP speculative config, applying rollout engine overrides."""
    if engine_speculative_config is None:
        engine_speculative_config = {}
    if isinstance(engine_speculative_config, str):
        engine_speculative_config = json.loads(engine_speculative_config)
    if not isinstance(engine_speculative_config, Mapping):
        raise TypeError("rollout.engine_kwargs.vllm.speculative_config must be a mapping when MTP rollout is enabled")

    return {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
        **{key: val for key, val in engine_speculative_config.items() if val is not None},
    }


def extract_prompt_logprobs(
    output: RequestOutput,
    num_prompt_logprobs: Optional[int],
    result_dict: dict[str, Any],
    start: int = 0,
    as_tensors: bool = False,
):
    """Extract prompt log probabilities, optionally starting at an artifact row."""
    if num_prompt_logprobs is None:
        return
    prompt_length = len(output.prompt_logprobs)
    if not 0 <= start < prompt_length:
        raise ValueError(f"prompt-logprob start {start} is outside [0, {prompt_length})")

    prompt_logprobs_ls, prompt_ids_ls = [], []
    # NOTE: logprob of first prompt token is None.
    for logprobs_dict in output.prompt_logprobs[start + 1 :]:
        if num_prompt_logprobs == 0:
            token_id_str = list(logprobs_dict.keys())[0]
            logprob = logprobs_dict[token_id_str].logprob
            prompt_logprobs_ls.append([logprob])
            prompt_ids_ls.append([int(token_id_str)])
        else:
            prompt_ids = [None] * num_prompt_logprobs
            prompt_logprobs = [None] * num_prompt_logprobs
            # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
            assert len(logprobs_dict) in [num_prompt_logprobs, num_prompt_logprobs + 1], len(logprobs_dict)
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                logprob = token_logprob.logprob
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)

    # NOTE: pad a dummy prompt logprob for last prompt token.
    prompt_logprobs_ls.append([0.0] * max(num_prompt_logprobs, 1))
    prompt_ids_ls.append([0] * max(num_prompt_logprobs, 1))

    if as_tensors:
        result_dict["prompt_ids"] = torch.tensor(prompt_ids_ls, dtype=torch.int32)
        result_dict["prompt_logprobs"] = torch.tensor(prompt_logprobs_ls, dtype=torch.float32)
    else:
        result_dict["prompt_ids"] = prompt_ids_ls
        result_dict["prompt_logprobs"] = prompt_logprobs_ls
