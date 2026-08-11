# SPDX-License-Identifier: Apache-2.0
"""The fp8-scheme guard, tested offline on fake methods.

fp8's failure mode is silent, so the OOT tier refuses schemes it has not been verified
against rather than serving wrong output. The guard reads only plain attributes, so it
is tested here without vLLM: block-quant is refused, per-tensor static passes, dynamic
activation warns.
"""

from __future__ import annotations

import logging

import pytest

from vllm_moe_surgeon.compat.fp8_runtime import refuse_unverified_fp8_scheme


class _QuantConfig:
    def __init__(self, activation_scheme: str = "static"):
        self.activation_scheme = activation_scheme


class _Method:
    def __init__(self, *, block_quant=False, weight_block_size=None,
                 activation_scheme="static"):
        self.block_quant = block_quant
        self.weight_block_size = weight_block_size
        self.quant_config = _QuantConfig(activation_scheme)


def test_block_quant_is_refused_by_flag():
    with pytest.raises(ValueError, match="block-quantized"):
        refuse_unverified_fp8_scheme(_Method(block_quant=True))


def test_block_quant_is_refused_by_block_size():
    with pytest.raises(ValueError, match="block-quantized"):
        refuse_unverified_fp8_scheme(_Method(weight_block_size=[128, 128]))


def test_per_tensor_static_passes():
    # The verified case: no raise.
    refuse_unverified_fp8_scheme(_Method())


def test_dynamic_activation_warns_but_does_not_refuse(caplog):
    with caplog.at_level(logging.WARNING):
        refuse_unverified_fp8_scheme(_Method(activation_scheme="dynamic"))
    assert "activation_scheme" in caplog.text
    assert "unverified" in caplog.text


def test_a_method_missing_the_attributes_is_treated_as_verified():
    """A minimal method object (no block_quant / quant_config) must not crash the
    guard -- getattr defaults keep it permissive for the common case."""
    refuse_unverified_fp8_scheme(object())
