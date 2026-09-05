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

import json
import os
from types import SimpleNamespace

import pytest
import torch

# vllm is not part of the `cpu` extra (it conflicts with the cpu torch world), so
# cpu_unit_tests skips this module; vllm.yml runs it in the vllm venv.
pytest.importorskip("vllm")

from verl.workers.rollout.vllm_rollout.utils import (
    _configure_exclusive_gpu_memory,
    _determine_exclusive_available_memory,
    _prompt_logprob_chunk_rows,
    _request_exclusive_gpu_memory,
    _resolve_vllm_weight_sync_local_rank,
    _streamopd_unprofiled_workspace_bytes,
    build_cli_args_from_config,
    vLLMColocateWorkerExtension,
)


def test_exclusive_gpu_memory_uses_post_nccl_free_bytes_without_deriving_fraction() -> None:
    cache_config = SimpleNamespace(gpu_memory_utilization=0.25)
    requested = _request_exclusive_gpu_memory(
        SimpleNamespace(free_memory=75 * 1024**3, total_memory=80 * 1024**3),
        cache_config,
    )
    assert requested == 75 * 1024**3
    assert cache_config.gpu_memory_utilization == 0.25


def test_exclusive_gpu_memory_marker_installs_worker_request(monkeypatch) -> None:
    from vllm.v1.worker import gpu_worker

    original = gpu_worker.request_memory
    original_determine = gpu_worker.Worker.determine_available_memory
    monkeypatch.setattr(gpu_worker, "request_memory", original)
    monkeypatch.setattr(gpu_worker.Worker, "determine_available_memory", original_determine)
    assert not _configure_exclusive_gpu_memory(SimpleNamespace(additional_config={}))
    assert gpu_worker.request_memory is original
    assert _configure_exclusive_gpu_memory(SimpleNamespace(additional_config={"verl_exclusive_gpu_memory": True}))
    assert gpu_worker.request_memory is _request_exclusive_gpu_memory
    assert gpu_worker.Worker.determine_available_memory._verl_exclusive_gpu_memory


def test_exclusive_gpu_memory_reserves_measured_runtime_and_graph_peaks() -> None:
    mib = 1024**2
    worker = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        model_runner=SimpleNamespace(
            cudagraph_dispatcher=SimpleNamespace(get_capture_descs=lambda: [("piecewise", [1, 2]), ("full", [1])])
        ),
        vllm_config=SimpleNamespace(kv_transfer_config=None),
        cache_config=SimpleNamespace(),
        peak_activation_memory=12 * mib,
    )

    assert _determine_exclusive_available_memory(worker, lambda _worker: 256 * mib) == 70 * mib
    assert worker.cache_config._verl_exclusive_activation_reserve_bytes == 36 * mib
    assert worker.cache_config._verl_exclusive_activation_reserve_count == 3
    assert worker.cache_config._verl_exclusive_graph_pool_count == 2
    assert worker.cache_config._verl_exclusive_connector_reserve_bytes == 0
    assert worker.cache_config._verl_exclusive_redundancy_reserve_bytes == 150 * mib


def test_exclusive_gpu_memory_accounts_for_native_sampler_workspace() -> None:
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        get_vocab_size=lambda: 151936,
    )
    vllm_config = SimpleNamespace(
        additional_config={"verl_exclusive_gpu_memory": True},
        kv_transfer_config=None,
        model_config=model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=64, max_num_batched_tokens=8192),
    )
    segment = 2 * 1024**2

    def rounded(size: int) -> int:
        return (size + segment - 1) // segment * segment

    rows = 64
    vocab = 151936
    expected = (
        rounded(rows * vocab * 2)
        + 5 * rounded(rows * vocab * 4)
        + rounded(rows * vocab * 8)
        + 2 * rounded(rows * vocab)
        + segment
    )
    assert _streamopd_unprofiled_workspace_bytes(vllm_config) == expected


def test_streaming_teacher_workspace_accounts_for_chunked_tp_logits_and_normalization() -> None:
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        max_model_len=4096,
        get_vocab_size=lambda: 151936,
    )
    vllm_config = SimpleNamespace(
        additional_config={"verl_streaming_teacher_logprobs": True},
        kv_transfer_config=None,
        model_config=model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
    )
    segment = 2 * 1024**2

    def rounded(size: int) -> int:
        return (size + segment - 1) // segment * segment

    expected = rounded(1024 * 151936 * 2) + rounded(1024 * (151936 // 2) * 2) + 2 * rounded(1024 * 151936 * 4) + segment
    assert _streamopd_unprofiled_workspace_bytes(vllm_config) == expected


def test_streaming_teacher_normalization_rows_follow_runtime_free_memory() -> None:
    vocab_size = 151936
    bytes_per_row = vocab_size * 5
    free_bytes = 2 * 2 * 1024**2 + 300 * bytes_per_row

    assert _prompt_logprob_chunk_rows(free_bytes=free_bytes, vocab_size=vocab_size, requested_rows=512) == 300
    assert _prompt_logprob_chunk_rows(free_bytes=free_bytes * 4, vocab_size=vocab_size, requested_rows=512) == 512


@pytest.mark.parametrize("export_strategy", ["eos_triton", "incremental_triton"])
def test_streamopd_gpu_gather_workspace_follows_export_strategy(export_strategy: str) -> None:
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        get_num_layers=lambda _parallel: 12,
        get_num_kv_heads=lambda _parallel: 4,
        get_head_size=lambda: 64,
    )
    extra = {
        "streamopd_kv_export_strategy": export_strategy,
        "streamopd_writer_threads": 3,
        "streamopd_kv_chunk_size": 256,
    }
    vllm_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=SimpleNamespace(
            kv_connector="StreamOPDKVConnector",
            kv_connector_extra_config=extra,
        ),
        model_config=model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    segment = 2 * 1024**2
    output_bytes = 12 * 256 * 2 * 4 * 64 * 2
    rounded_output_bytes = (output_bytes + segment - 1) // segment * segment

    assert _streamopd_unprofiled_workspace_bytes(vllm_config) == 3 * rounded_output_bytes + segment


def test_streamopd_eos_host_reserves_no_gpu_gather_workspace() -> None:
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    vllm_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=SimpleNamespace(
            kv_connector="StreamOPDKVConnector",
            kv_connector_extra_config={
                "streamopd_kv_export_strategy": "eos_host",
                "streamopd_writer_threads": 64,
                "streamopd_kv_chunk_size": 8192,
            },
        ),
        model_config=model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )

    assert _streamopd_unprofiled_workspace_bytes(vllm_config) == 0


def test_vllm_server_uses_replica_local_rendezvous_and_releases_ports(monkeypatch):
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    class ReservedSocket:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    server = vLLMHttpServer.__new__(vLLMHttpServer)
    server._master_address = "10.0.0.7"
    server._master_port = 31234
    server._master_sock = ReservedSocket()
    server._dp_rpc_sock = ReservedSocket()
    server._dp_master_sock = ReservedSocket()
    reserved_sockets = (server._master_sock, server._dp_rpc_sock, server._dp_master_sock)
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "8099")
    monkeypatch.setenv("VLLM_PORT", "8099")

    server._prepare_engine_core_environment()

    assert os.environ["MASTER_ADDR"] == "10.0.0.7"
    assert os.environ["MASTER_PORT"] == "31234"
    assert os.environ["VLLM_PORT"] == "31234"
    assert all(sock.closed for sock in reserved_sockets)
    assert server._master_sock is None
    assert server._dp_rpc_sock is None
    assert server._dp_master_sock is None


class TestBuildCliArgsFromConfig:
    """Tests for CLI argument serialization from config dictionaries."""

    def test_string_value(self):
        """String values become '--key value'."""
        config = {"model": "gpt2"}
        result = build_cli_args_from_config(config)
        assert result == ["--model", "gpt2"]

    def test_integer_value(self):
        """Integer values are converted to strings."""
        config = {"tensor-parallel-size": 4}
        result = build_cli_args_from_config(config)
        assert result == ["--tensor-parallel-size", "4"]

    def test_float_value(self):
        """Float values are converted to strings."""
        config = {"temperature": 0.7}
        result = build_cli_args_from_config(config)
        assert result == ["--temperature", "0.7"]

    def test_bool_true(self):
        """Bool True adds flag without value."""
        config = {"enable-prefix-caching": True}
        result = build_cli_args_from_config(config)
        assert result == ["--enable-prefix-caching"]

    def test_bool_false(self):
        """Optional[bool] args emit '--no-key' for an explicit False."""
        config = {"enable-prefix-caching": False}
        result = build_cli_args_from_config(config)
        assert result == ["--no-enable-prefix-caching"]

    def test_bool_false_plain_bool_omitted(self):
        """Bool False on a plain-bool arg is skipped (parser default is False)."""
        config = {"enforce_eager": False}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_bool_false_union_str_arg_omitted(self):
        """Bool False on a `bool | str | None` arg is skipped (string flag, no --no- form)."""
        config = {"hf_token": False}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_bool_false_underscore_key(self):
        """Underscore keys emit the negative flag in the key's own spelling."""
        config = {"enable_prefix_caching": False}
        result = build_cli_args_from_config(config)
        assert result == ["--no-enable_prefix_caching"]

    def test_bool_false_non_engine_arg_omitted(self):
        """Bool False on args unknown to AsyncEngineArgs is omitted."""
        config = {"disable-log-requests": False}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_none_value(self):
        """None values are skipped."""
        config = {"lora-path": None}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_list_values(self):
        """List values are expanded into multiple arguments."""
        config = {"cudagraph-capture-sizes": [1, 2, 4, 8]}
        result = build_cli_args_from_config(config)
        assert result == ["--cudagraph-capture-sizes", "1", "2", "4", "8"]

    def test_empty_list(self):
        """Empty lists are skipped (vLLM nargs='+' requires at least one value)."""
        config = {"cudagraph-capture-sizes": []}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_list_with_strings(self):
        """List of strings is properly expanded."""
        config = {"allowed-origins": ["http://localhost", "http://example.com"]}
        result = build_cli_args_from_config(config)
        assert result == ["--allowed-origins", "http://localhost", "http://example.com"]

    def test_dict_value(self):
        """Dict values are JSON serialized."""
        config = {"extra-config": {"key": "value", "nested": True}}
        result = build_cli_args_from_config(config)
        assert result[0] == "--extra-config"
        # JSON output may have different key ordering, so parse and compare
        assert json.loads(result[1]) == {"key": "value", "nested": True}

    def test_mixed_config(self):
        """Test a realistic mixed configuration."""
        config = {
            "tensor-parallel-size": 4,
            "enable-prefix-caching": True,
            "disable-log-requests": False,
            "lora-path": None,
            "cudagraph-capture-sizes": [1, 2, 4, 8],
            "max-model-len": 2048,
        }
        result = build_cli_args_from_config(config)

        # Check expected args are present
        assert "--tensor-parallel-size" in result
        assert "4" in result
        assert "--enable-prefix-caching" in result
        assert "--cudagraph-capture-sizes" in result
        assert "1" in result
        assert "8" in result
        assert "--max-model-len" in result
        assert "2048" in result

        # Check skipped values are not present
        assert "--disable-log-requests" not in result
        assert "--lora-path" not in result

    def test_preserves_order(self):
        """Arguments should preserve dictionary order (Python 3.7+)."""
        config = {"first": "a", "second": "b", "third": "c"}
        result = build_cli_args_from_config(config)
        assert result == ["--first", "a", "--second", "b", "--third", "c"]

    def test_empty_config(self):
        """Empty config returns empty list."""
        config = {}
        result = build_cli_args_from_config(config)
        assert result == []

    def test_single_element_list(self):
        """Single element list works correctly."""
        config = {"sizes": [42]}
        result = build_cli_args_from_config(config)
        assert result == ["--sizes", "42"]


class TestCliArgsVllmParserRoundTrip:
    """Serialized args must round-trip through vLLM's serve CLI parser."""

    @staticmethod
    def _build_parser():
        import vllm.entrypoints.cli.serve as serve_mod
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser(description="test")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        for cmd in serve_mod.cmd_init():
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
        return parser

    def test_explicit_false_survives_parsing(self):
        """Explicit False survives parse_args and AsyncEngineArgs.from_cli_args."""
        from vllm.engine.arg_utils import AsyncEngineArgs

        parser = self._build_parser()
        config = {
            "skip_tokenizer_init": False,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "enable_sleep_mode": True,
            "enforce_eager": False,
            "disable_log_stats": False,
        }
        argv = ["serve", "dummy-model"] + build_cli_args_from_config(config)
        namespace = parser.parse_args(args=argv)
        engine_args = AsyncEngineArgs.from_cli_args(namespace)
        assert engine_args.enable_prefix_caching is False
        assert engine_args.enable_chunked_prefill is True
        assert engine_args.enable_sleep_mode is True
        assert engine_args.skip_tokenizer_init is False
        assert engine_args.enforce_eager is False
        assert engine_args.disable_log_stats is False


class TestVllmColocateZmqHandle:
    def test_dp_local_rank_offsets_tensor_parallel_rank(self):
        """DP workers on the same node must not reuse the same TP-local socket."""
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=1,
        )

        assert _resolve_vllm_weight_sync_local_rank(1, parallel_config) == 3
        assert _resolve_vllm_weight_sync_local_rank(3, parallel_config) == 3

    def test_single_dp_keeps_local_rank(self):
        """The old single-DP handle layout remains unchanged."""
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank_local=0,
        )

        assert _resolve_vllm_weight_sync_local_rank(1, parallel_config) == 1

    def test_uses_global_dp_rank_when_local_rank_is_unset(self):
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=None,
            data_parallel_rank=3,
        )

        assert _resolve_vllm_weight_sync_local_rank(0, parallel_config) == 2

    def test_zmq_handle_uses_resolved_dp_rank(self, monkeypatch):
        parallel_config = SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_size=4,
            data_parallel_size_local=2,
            data_parallel_rank_local=1,
        )
        worker = SimpleNamespace(
            local_rank=1,
            model_runner=SimpleNamespace(
                vllm_config=SimpleNamespace(parallel_config=parallel_config),
            ),
        )
        monkeypatch.setenv("VERL_REPLICA_RANK", "2")
        monkeypatch.setenv("VERL_RAY_JOB_ID", "job-123")

        handle = vLLMColocateWorkerExtension._get_zmq_handle(worker)

        assert handle == "ipc:///tmp/rl-colocate-zmq-job-123-replica-2-rank-3.sock"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
