# SPDX-License-Identifier: Apache-2.0
"""Configuration checks for the out-of-tree tier, without vLLM.

``check_config`` answers questions about the *request* alone -- no layer, no device,
no vLLM -- so these run in the laptop loop. The refusals are the interesting part:
each is a case where proceeding produces a wrong *result* rather than an error, so it
has to be caught before a weight is allocated. ``validate`` adds the checks that need
a layer and an engine, and is exercised on the GPU boxes.
"""

from __future__ import annotations

import pytest

from vllm_moe_surgeon.compat.runtime import RuntimeConfig, check_config, validate


class _ParallelConfig:
    use_ep = False
    ep_size = 1
    dp_size = 1
    is_sequence_parallel = False


class _MoEConfig:
    experts_per_token = 8
    has_bias = False
    moe_parallel_config = _ParallelConfig()


class _Layer:
    """The little of a RoutedExperts layer that ``validate`` reads."""

    layer_name = "model.layers.0.mlp.experts"
    local_num_experts = 64
    moe_config = _MoEConfig()


def test_use_disk_needs_both_a_directory_and_a_ram_budget():
    assert RuntimeConfig(expert_cache_size=24, store_dir="/s", ram_cache=48).use_disk
    assert not RuntimeConfig(expert_cache_size=24, store_dir="/s", ram_cache=0).use_disk
    assert not RuntimeConfig(
        expert_cache_size=24, store_dir=None, ram_cache=48
    ).use_disk


def test_a_store_without_a_ram_budget_is_refused():
    """The combination that produced a mislabelled measurement.

    Three boot-floor arms were recorded as "tiered, ram_cache 0". They were not tiered:
    ``use_disk`` is ``bool(store_dir) and ram_cache > 0``, so the disk store was never
    built and the provider ran full-DRAM, holding every expert page-locked for the
    process lifetime -- 31.10 GiB against the genuine tiered arm's 23.40 GiB. Both
    modes logged the same "expert cache active" line, so nothing gave it away.

    A store_dir asks for a disk tier and ram_cache=0 refuses to build one. That is a
    contradiction, and silently resolving it in favour of the more expensive mode is
    how the wrong number got recorded.
    """
    config = RuntimeConfig(expert_cache_size=24, store_dir="/store", ram_cache=0)
    with pytest.raises(ValueError, match="asks for the disk tier"):
        check_config(config)

    # The message has to name the way out, because the fix is not obvious from
    # "disabled": raising ram_cache is one, dropping store_dir is the other.
    try:
        check_config(config)
    except ValueError as exc:
        assert "ram_cache >= expert_cache_size" in str(exc)
        assert "drop store_dir" in str(exc)


def test_full_dram_mode_is_still_allowed_when_asked_for_deliberately():
    """No store_dir means the operator wants full-DRAM. That is a valid request."""
    check_config(RuntimeConfig(expert_cache_size=24, ram_cache=0))
    check_config(RuntimeConfig(expert_cache_size=24, store_dir="/s", ram_cache=48))


def test_a_disabled_config_is_not_second_guessed():
    """Nothing is checked when the tier is off -- there is nothing to get wrong."""
    check_config(RuntimeConfig(expert_cache_size=0, store_dir="/s", ram_cache=0))
    check_config(RuntimeConfig(expert_cache_size=0, fp8_store=True))


def test_a_warm_tier_smaller_than_the_resident_set_is_refused():
    """A pool below capacity cannot hold what the GPU has: every eviction hits disk."""
    with pytest.raises(ValueError, match="below expert_cache_size"):
        check_config(
            RuntimeConfig(expert_cache_size=24, store_dir="/s", ram_cache=16)
        )
    check_config(RuntimeConfig(expert_cache_size=24, store_dir="/s", ram_cache=24))


def test_fp8_store_without_a_store_is_refused():
    """fp8_store describes record encoding. With no store it encodes nothing."""
    with pytest.raises(ValueError, match="no .*store is being built"):
        check_config(RuntimeConfig(expert_cache_size=24, fp8_store=True))
    check_config(
        RuntimeConfig(
            expert_cache_size=24, store_dir="/s", ram_cache=48, fp8_store=True
        )
    )


# ------------------------------------------------- the checks that need a live engine


def test_validate_delegates_the_pure_config_checks():
    """validate must not be the only door: the same contradiction has to fail there.

    It reaches vLLM further down, so this asserts only that the delegated check fires
    first -- which is what makes the vLLM-free tests above meaningful.
    """
    config = RuntimeConfig(expert_cache_size=24, store_dir="/store", ram_cache=0)
    with pytest.raises(ValueError, match="asks for the disk tier"):
        validate(config, _Layer())


def test_capacity_below_top_k_is_refused_unless_split_by_expert():
    pytest.importorskip("vllm", reason="validate reaches vLLM for the eager check")
    layer = _Layer()
    with pytest.raises(ValueError, match="fewer than the 8 experts"):
        validate(RuntimeConfig(expert_cache_size=4, ram_cache=0), layer)


def test_bias_and_parallelism_are_refused_by_name():
    layer = _Layer()

    class _Biased(_MoEConfig):
        has_bias = True

    layer.moe_config = _Biased()
    with pytest.raises(ValueError, match="bias"):
        validate(RuntimeConfig(expert_cache_size=24, ram_cache=0), layer)

    class _EP(_ParallelConfig):
        use_ep = True
        ep_size = 2

    class _MoEEP(_MoEConfig):
        moe_parallel_config = _EP()

    layer.moe_config = _MoEEP()
    with pytest.raises(ValueError, match="expert parallelism"):
        validate(RuntimeConfig(expert_cache_size=24, ram_cache=0), layer)
