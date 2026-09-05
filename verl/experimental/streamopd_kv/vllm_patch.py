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
"""Opt-in vLLM memory planning and StreamingInput compatibility for StreamOPD.

Keep version-specific vLLM internals here; ordinary verl workers do not install
these patches. Remove compatibility patches when upstream supports the same
streamed teacher artifacts.
"""

import logging
from types import MethodType

import torch
from packaging import version

from verl.utils.device import get_device_name, get_torch_device

logger = logging.getLogger(__name__)

# The auto StreamOPD profile caps Teacher input fragments at 1024 tokens. Keep
# one fragment in one LM-head tile while retaining the runtime memory fallback
# below for explicit larger fragments and non-exclusive vLLM deployments.
_STREAMING_PROMPT_LOGPROBS_CHUNK_SIZE = 1024
_CUDA_ALLOCATOR_SEGMENT_BYTES = 2 * 1024**2
_VLLM_REDUNDANCY_BUFFER_BYTES = 150 * 1024**2
_EXCLUSIVE_GPU_MEMORY_KEY = "verl_exclusive_gpu_memory"
_STREAMING_TEACHER_MEMORY_KEY = "verl_streaming_teacher_logprobs"


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
    budget = available_bytes - reserve_bytes
    trajectory_count = int(
        (getattr(worker.vllm_config, "additional_config", {}) or {}).get("verl_streamopd_max_live_trajectories", 0)
    )
    if trajectory_count:
        # A strict policy cannot have more live trajectories than its entire
        # global batch, even if every request routes to this replica. Reserving
        # KV beyond that worst case adds no concurrency and can leave too
        # little room for lazily initialized training/library state at wake.
        model = worker.vllm_config.model_config
        parallel = worker.vllm_config.parallel_config
        page_size = int(worker.cache_config.block_size)
        slot_tokens = (int(model.max_model_len) + page_size - 1) // page_size * page_size
        # Teacher admission can use a larger page quantum than the engine.
        # Its reservation, including that rounding, must still fit in full.
        slot_tokens = max(
            slot_tokens, int(worker.vllm_config.additional_config.get("verl_streamopd_reserved_trajectory_tokens", 0))
        )
        bytes_per_token = (
            int(model.get_num_layers(parallel))
            * int(model.get_num_kv_heads(parallel))
            * int(model.get_head_size())
            * 2
            * torch.empty((), dtype=model.dtype).element_size()
        )
        budget = min(budget, trajectory_count * slot_tokens * bytes_per_token)
    return budget


def _configure_exclusive_gpu_memory(vllm_config) -> bool:
    additional_config = getattr(vllm_config, "additional_config", {}) or {}
    if not additional_config.get(_EXCLUSIVE_GPU_MEMORY_KEY, False):
        return False
    from vllm.v1.worker import gpu_worker

    vllm_config.cache_config._verl_exclusive_gpu_memory = True
    original_request = gpu_worker.request_memory
    if not getattr(original_request, "_verl_exclusive_gpu_memory", False):

        def request_memory(init_snapshot, cache_config):
            if getattr(cache_config, "_verl_exclusive_gpu_memory", False):
                return _request_exclusive_gpu_memory(init_snapshot, cache_config)
            return original_request(init_snapshot, cache_config)

        request_memory._verl_exclusive_gpu_memory = True
        gpu_worker.request_memory = request_memory
    original_determine = gpu_worker.Worker.determine_available_memory
    if not getattr(original_determine, "_verl_exclusive_gpu_memory", False):

        def determine_available_memory(worker) -> int:
            if not (getattr(worker.vllm_config, "additional_config", {}) or {}).get(_EXCLUSIVE_GPU_MEMORY_KEY):
                return original_determine(worker)
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


def enable_streaming_prompt_logprobs(worker) -> bool:
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

    model_runner = worker.model_runner
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
