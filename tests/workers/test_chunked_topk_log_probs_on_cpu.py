# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Numerical and gradient tests for memory-bounded top-k log probabilities.

The helper avoids materializing the [B, T, V] log_softmax tensor (which can
exceed 28 GB at long context with V=152064 in bf16) by recomputing fused
log-softmax one chunk at a time during backward.

These tests verify:
  1. Forward output matches torch.log_softmax(...).gather(...).
  2. Result is independent of chunk_size (only memory/perf knob, never numerics).
  3. Backward gradients match the reference within autograd precision (~2.4e-7).
  4. The custom autograd function retains only the original logits and ids, not
     FP32 vocabulary-sized intermediates.

All tests run on CPU and complete in seconds.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
import torch.nn.functional as F

from verl.trainer.distillation.fsdp.losses import _chunked_topk_log_probs
from verl.workers.config.distillation import DistillationLossConfig


def _reference_topk_log_probs(logits: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    """Reference: standard log_softmax + gather (the pre-patch code path)."""
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=topk_ids)


@pytest.mark.parametrize(
    "B,T,V,K,dtype,atol",
    [
        # Small case: tight tolerance to verify correctness, not numerics.
        (2, 16, 128, 4, torch.float32, 5e-6),
        # Realistic vocab (Qwen-class V=152064) at modest seq.
        (2, 32, 152064, 8, torch.float32, 5e-6),
        # Larger top-K (OPD often uses K=64).
        (2, 32, 152064, 64, torch.float32, 5e-6),
    ],
)
def test_numerical_equivalence(B, T, V, K, dtype, atol):
    """chunked output matches torch.log_softmax(...).gather(...) within tolerance.

    Tests fp32 only on CPU. bf16/fp16 are tested separately on GPU
    (see test_numerical_equivalence_low_precision_on_gpu) because PyTorch's
    CPU log_softmax does not upcast bf16/fp16 to fp32, accumulating ~8e-3
    error that has nothing to do with our chunked implementation - it is
    the precision floor of bf16/fp16 mantissa on CPU.
    """
    torch.manual_seed(42)
    logits = torch.randn(B, T, V, dtype=dtype)
    topk_ids = torch.randint(0, V, (B, T, K))

    ref = _reference_topk_log_probs(logits, topk_ids)
    out = _chunked_topk_log_probs(logits, topk_ids, chunk_size=4096)

    max_diff = (ref - out).abs().max().item()
    assert max_diff <= atol, (
        f"Numerical mismatch for B={B}, T={T}, V={V}, K={K}, dtype={dtype}: "
        f"max |ref - out| = {max_diff:.2e} > atol = {atol:.0e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only test for low-precision dtypes")
@pytest.mark.parametrize(
    "dtype,atol",
    [
        # On GPU, fused log_softmax kernel internally upcasts to fp32, so
        # bf16/fp16 outputs are nearly identical to our chunked implementation.
        (torch.bfloat16, 5e-3),
        (torch.float16, 1e-3),
    ],
)
def test_numerical_equivalence_low_precision_on_gpu(dtype, atol):
    """Verify bf16/fp16 equivalence on GPU (where fused log_softmax upcasts)."""
    torch.manual_seed(42)
    B, T, V, K = 2, 64, 152064, 64
    logits = torch.randn(B, T, V, dtype=dtype, device="cuda")
    topk_ids = torch.randint(0, V, (B, T, K), device="cuda")

    ref = _reference_topk_log_probs(logits, topk_ids)
    out = _chunked_topk_log_probs(logits, topk_ids, chunk_size=4096)

    max_diff = (ref - out).abs().max().item()
    assert max_diff <= atol, (
        f"Numerical mismatch on GPU dtype={dtype}: max |ref - out| = {max_diff:.2e} > atol = {atol:.0e}"
    )


@pytest.mark.parametrize("chunk_size", [1, 64, 256, 1024, 4096, 16384, 99999999])
def test_chunk_size_invariance(chunk_size):
    """Result is independent of chunk_size (only changes memory/speed, not numerics)."""
    torch.manual_seed(123)
    B, T, V, K = 2, 64, 152064, 32
    logits = torch.randn(B, T, V, dtype=torch.float32)
    topk_ids = torch.randint(0, V, (B, T, K))

    # Single-chunk reference (chunk_size larger than N).
    out_single = _chunked_topk_log_probs(logits, topk_ids, chunk_size=99999999)
    out = _chunked_topk_log_probs(logits, topk_ids, chunk_size=chunk_size)

    max_diff = (out - out_single).abs().max().item()
    assert max_diff <= 5e-6, f"Result depends on chunk_size={chunk_size}: max diff vs single-chunk = {max_diff:.2e}"


def test_gradient_correctness():
    """Backward gradients match the reference implementation."""
    torch.manual_seed(7)
    B, T, V, K = 2, 16, 1024, 8

    # Build two identical input copies for autograd.
    logits1 = torch.randn(B, T, V, dtype=torch.float32, requires_grad=True)
    logits2 = logits1.detach().clone().requires_grad_(True)
    topk_ids = torch.randint(0, V, (B, T, K))

    ref = _reference_topk_log_probs(logits1, topk_ids)
    out = _chunked_topk_log_probs(logits2, topk_ids, chunk_size=4)

    # Forward equivalence (sanity).
    assert (ref - out).abs().max().item() <= 5e-6

    # Backward equivalence: scalar loss = sum.
    ref.sum().backward()
    out.sum().backward()

    grad_diff = (logits1.grad - logits2.grad).abs().max().item()
    assert grad_diff <= 1e-5, f"Gradient mismatch: max diff = {grad_diff:.2e}"


def test_handles_small_chunk_size():
    """Edge case: chunk_size smaller than N still produces correct output."""
    torch.manual_seed(0)
    B, T, V, K = 1, 1, 100, 5
    logits = torch.randn(B, T, V, dtype=torch.float32)
    topk_ids = torch.randint(0, V, (B, T, K))

    ref = _reference_topk_log_probs(logits, topk_ids)
    out = _chunked_topk_log_probs(logits, topk_ids, chunk_size=1)

    assert (ref - out).abs().max().item() <= 5e-6


def test_empty_input():
    """Edge case: N=0 (fully-padded micro-batch) returns empty tensor without error."""
    logits = torch.empty((0, 0, 1024), dtype=torch.float32)
    topk_ids = torch.empty((0, 0, 8), dtype=torch.long)
    out = _chunked_topk_log_probs(logits, topk_ids, chunk_size=4096)
    assert out.shape == (0, 0, 8)
    assert out.dtype == torch.float32


def test_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError, match="must be positive"):
        _chunked_topk_log_probs(
            torch.randn(1, 2, 8),
            torch.randint(0, 8, (1, 2, 2)),
            chunk_size=0,
        )


def test_memory_bound_chunk_config_has_safe_default_and_validates():
    assert DistillationLossConfig().chunked_topk_chunk_size == 512
    with pytest.raises(ValueError, match="must be positive"):
        DistillationLossConfig(chunked_topk_chunk_size=0)


def test_gradient_flows_through_memory_bound_function():
    """The memory-bounded autograd function preserves the logits gradient."""
    torch.manual_seed(0)
    B, T, V, K = 2, 8, 256, 4
    logits = torch.randn(B, T, V, dtype=torch.float32, requires_grad=True)
    topk_ids = torch.randint(0, V, (B, T, K))

    out = _chunked_topk_log_probs(logits, topk_ids, chunk_size=4)
    assert not out.is_leaf, "chunked output should be tracked by autograd"
    assert out.requires_grad, "out should have requires_grad=True"

    out.sum().backward()
    assert logits.grad is not None, "gradient should propagate to logits"
    assert torch.isfinite(logits.grad).all(), "gradient should be finite"
    assert (logits.grad.abs() > 0).any(), "gradient should be non-zero"


def test_memory_bound_function_does_not_save_fp32_vocabulary_intermediates():
    logits = torch.randn(2, 8, 256, dtype=torch.bfloat16, requires_grad=True)
    topk_ids = torch.randint(0, 256, (2, 8, 4))
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        _chunked_topk_log_probs(logits, topk_ids, chunk_size=3)

    vocabulary_tensors = [tensor for tensor in saved if tensor.ndim == 2 and tensor.shape[-1] == 256]
    assert vocabulary_tensors
    assert all(tensor.dtype == logits.dtype for tensor in vocabulary_tensors)
    assert all(
        tensor.untyped_storage().data_ptr() == logits.untyped_storage().data_ptr() for tensor in vocabulary_tensors
    )
