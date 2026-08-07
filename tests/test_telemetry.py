# SPDX-License-Identifier: Apache-2.0
"""Telemetry aggregation, including the traps that would bias a pruning run.

All CPU, all numpy, no vLLM -- which is the point: this phase rides vLLM's
public ``--enable-return-routed-experts`` output instead of hooking a layer.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

from vllm_moe_surgeon.telemetry import (
    ExpertStats,
    RoutedExpertsUnavailable,
    accumulate,
    decode,
    from_hf_config,
    get_num_experts,
    get_top_k,
    resolve_moe_layers,
)

# ---------------------------------------------------------------- layer rules


def test_olmoe_style_config_is_all_layers():
    cfg = {"num_hidden_layers": 4, "num_experts": 64, "num_experts_per_tok": 8}
    assert resolve_moe_layers(cfg) == [0, 1, 2, 3]
    assert get_num_experts(cfg) == 64
    assert get_top_k(cfg) == 8


def test_qwen3_moe_decoder_sparse_step_and_mlp_only():
    """Qwen convention: sparse when (idx + 1) % step == 0, minus mlp_only."""
    cfg = {
        "num_hidden_layers": 8,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "decoder_sparse_step": 2,
        "mlp_only_layers": [3],
    }
    # (idx+1) % 2 == 0 -> 1, 3, 5, 7; minus mlp_only 3
    assert resolve_moe_layers(cfg) == [1, 5, 7]


def test_deepseek_first_k_dense_replace():
    cfg = {
        "num_hidden_layers": 6,
        "n_routed_experts": 160,
        "num_experts_per_tok": 6,
        "first_k_dense_replace": 2,
    }
    assert resolve_moe_layers(cfg) == [2, 3, 4, 5]
    assert get_num_experts(cfg) == 160


def test_dense_model_has_no_moe_layers():
    assert resolve_moe_layers({"num_hidden_layers": 4, "num_experts": 0}) == []


def test_multimodal_config_descends_into_text_config():
    cfg = {
        "text_config": {
            "num_hidden_layers": 2,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        }
    }
    assert resolve_moe_layers(cfg) == [0, 1]
    assert get_num_experts(cfg) == 8


def test_unresolvable_config_raises():
    with pytest.raises(ValueError, match="could not resolve num_experts"):
        get_num_experts({"num_hidden_layers": 2})


# ------------------------------------------------------------- accumulation

E = 8
TOP_K = 2


def _capture(rows_per_layer: dict[int, list[list[int]]], num_layers: int, seq_len: int):
    """Build a [seq_len, num_layers, top_k] capture, zeros where uncaptured."""
    arr = np.zeros((seq_len, num_layers, TOP_K), dtype=np.int32)
    for layer, rows in rows_per_layer.items():
        arr[:, layer, :] = np.asarray(rows, dtype=np.int32)
    return arr


def test_counts_are_token_weighted():
    """A token contributes to top_k experts; counts are slots, not tokens."""
    capture = _capture({0: [[1, 2], [1, 3]]}, num_layers=1, seq_len=2)
    stats = accumulate([capture], num_experts=E, moe_layers=[0], top_k=TOP_K)

    assert stats.tokens[0, 1] == 2  # expert 1 chosen by both tokens
    assert stats.tokens[0, 2] == 1
    assert stats.tokens[0, 3] == 1
    assert stats.tokens[0].sum() == 4  # 2 tokens * top_k 2
    assert stats.layer_token_rows[0] == 2
    assert stats.top_k_hint == pytest.approx(2.0)


def test_dense_layer_zero_rows_do_not_become_expert_zero():
    """The trap this module exists for.

    Layer 1 is dense, so its rows stay zero in the capture buffer. Counting them
    would report expert 0 as chosen by every token of a layer that has no
    experts -- and would then protect expert 0 across the whole model.
    """
    capture = _capture({0: [[1, 2], [3, 4]]}, num_layers=2, seq_len=2)
    assert (capture[:, 1, :] == 0).all()  # the dense layer, as vLLM leaves it

    # Config says only layer 0 is MoE, so layer 1 is never looked at.
    stats = accumulate([capture], num_experts=E, moe_layers=[0], top_k=TOP_K)
    assert stats.tokens.shape == (1, E)
    assert stats.tokens[0, 0] == 0, "expert 0 was never actually selected"

    # And if a wrong config claims layer 1 too, the degenerate-row check
    # rescues us instead of inventing load for expert 0.
    wrong = accumulate([capture], num_experts=E, moe_layers=[0, 1], top_k=TOP_K)
    assert wrong.tokens[1].sum() == 0
    assert wrong.dropped_degenerate_rows == 2
    assert wrong.silent_layers == {1}


def test_genuine_repeated_row_is_impossible_at_top_k_2():
    """The degenerate-row rule is safe: top-k never repeats an expert.

    A row like [5, 5] cannot come from torch.topk, so treating it as uncaptured
    cannot discard real data.
    """
    capture = _capture({0: [[5, 5]]}, num_layers=1, seq_len=1)
    stats = accumulate([capture], num_experts=E, moe_layers=[0], top_k=TOP_K)
    assert stats.tokens.sum() == 0
    assert stats.dropped_degenerate_rows == 1


def test_top_k_one_refuses_to_guess():
    """At top_k == 1 the trap is undetectable, so an empty layer set is fatal."""
    with pytest.raises(ValueError, match="indistinguishable"):
        accumulate([], num_experts=E, moe_layers=[], top_k=1)


def test_negative_sentinels_are_dropped_not_counted():
    capture = np.array([[[1, -1]], [[2, 3]]], dtype=np.int32)  # [2, 1, 2]
    stats = accumulate([capture], num_experts=E, moe_layers=[0], top_k=TOP_K)
    assert stats.dropped_sentinels == 1
    assert stats.tokens[0].sum() == 3
    assert stats.tokens[0, 1] == 1


def test_out_of_range_expert_id_raises():
    """A wrong num_experts must fail loudly, not wrap or truncate."""
    capture = _capture({0: [[1, 99]]}, num_layers=1, seq_len=1)
    with pytest.raises(ValueError, match="num_experts is"):
        accumulate([capture], num_experts=E, moe_layers=[0], top_k=TOP_K)


def test_depth_mismatch_raises():
    capture = _capture({0: [[1, 2]]}, num_layers=1, seq_len=1)
    with pytest.raises(ValueError, match="disagree about depth"):
        accumulate([capture], num_experts=E, moe_layers=[0, 5], top_k=TOP_K)


def test_layer_share_rows_sum_to_one_and_silent_layers_are_zero():
    capture = _capture({0: [[1, 2], [1, 2]]}, num_layers=2, seq_len=2)
    stats = accumulate([capture], num_experts=E, moe_layers=[0, 1], top_k=TOP_K)
    share = stats.layer_share()
    assert share[0].sum() == pytest.approx(1.0)
    assert share[1].sum() == 0.0  # silent layer, not NaN
    assert not np.isnan(share).any()


def test_cooccurrence_is_symmetric_and_counts_pairs():
    capture = _capture({0: [[1, 2], [1, 2], [1, 3]]}, num_layers=1, seq_len=3)
    stats = accumulate(
        [capture], num_experts=E, moe_layers=[0], top_k=TOP_K, with_cooc=True
    )
    assert stats.cooc is not None
    assert stats.cooc[0][1, 2] == 2
    assert stats.cooc[0][2, 1] == 2  # symmetric
    assert stats.cooc[0][1, 3] == 1
    assert stats.cooc[0][2, 3] == 0


def test_merge_combines_workers():
    a = accumulate(
        [_capture({0: [[1, 2]]}, 1, 1)], num_experts=E, moe_layers=[0], top_k=TOP_K
    )
    b = accumulate(
        [_capture({0: [[1, 3]]}, 1, 1)], num_experts=E, moe_layers=[0], top_k=TOP_K
    )
    a.merge(b)
    assert a.tokens[0, 1] == 2
    assert a.tokens[0, 3] == 1
    assert a.n_sequences == 2


def test_merge_rejects_mismatched_shape():
    a = ExpertStats.empty(E, [0])
    with pytest.raises(ValueError, match="num_experts mismatch"):
        a.merge(ExpertStats.empty(E + 1, [0]))


def test_from_hf_config_sizes_the_accumulator():
    cfg = {
        "num_hidden_layers": 4,
        "num_experts": 16,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 2,
    }
    stats = from_hf_config(cfg)
    assert stats.num_experts == 16
    assert stats.moe_layers == [1, 3]
    assert stats.tokens.shape == (2, 16)


# ---------------------------------------------------------------- transport


def _encode(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_decode_roundtrips_the_server_encoding():
    arr = np.arange(2 * 3 * 2, dtype=np.int32).reshape(2, 3, 2)
    got = decode(_encode(arr))
    np.testing.assert_array_equal(got, arr)


def test_missing_payload_is_loud_by_default():
    """Silently skipping is how you get a confident, empty profile."""
    with pytest.raises(RoutedExpertsUnavailable, match="enable-return-routed-experts"):
        decode(None)
    assert decode(None, strict=False) is None


def test_decode_rejects_wrong_rank():
    with pytest.raises(ValueError, match="expected 3 dims"):
        decode(_encode(np.zeros((2, 2), dtype=np.int32)))


# ----------------------------------------------------------------- persistence


def test_real_configs_with_explicit_nulls():
    """OLMoE ships decoder_sparse_step: null and mlp_only_layers: null.

    A None must read as "no restriction", not crash or exclude everything.
    """
    cfg = {
        "num_hidden_layers": 16,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "decoder_sparse_step": None,
        "mlp_only_layers": None,
        "first_k_dense_replace": None,
        "moe_layer_freq": None,
    }
    assert resolve_moe_layers(cfg) == list(range(16))


def test_save_load_roundtrip_carries_provenance(tmp_path):
    from vllm_moe_surgeon.telemetry import load, save

    stats = accumulate(
        [_capture({0: [[1, 2], [1, 3]]}, 1, 2)],
        num_experts=E,
        moe_layers=[0],
        top_k=TOP_K,
        with_cooc=True,
    )
    path = str(tmp_path / "profile.npz")
    save(stats, path, model="test/tiny", revision="abc")

    back, meta = load(path)
    np.testing.assert_array_equal(back.tokens, stats.tokens)
    np.testing.assert_array_equal(back.cooc, stats.cooc)
    assert back.moe_layers == stats.moe_layers
    assert meta["model"] == "test/tiny"
    assert meta["revision"] == "abc"
    # The caveats must survive: a plan built on a partial profile has to be
    # auditable after the fact.
    assert meta["dropped_degenerate_rows"] == stats.dropped_degenerate_rows
    assert meta["silent_layers"] == []
    assert meta["mean_experts_per_token"] == pytest.approx(2.0)


def test_num_local_experts_alias_is_not_read_as_dense():
    """transformers renames the key, and a narrower guard called it dense.

    nm-testing/tinysmokeqwen3moe ships "num_experts": 8 in config.json, but
    Qwen3MoeConfig.to_dict() surfaces it as "num_local_experts". A guard that
    checked only num_experts/n_routed_experts reported 0 MoE layers for a real
    MoE model -- caught by running against the actual config, not a handwritten
    one.
    """
    cfg = {
        "num_hidden_layers": 6,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
    }
    assert get_num_experts(cfg) == 8
    assert resolve_moe_layers(cfg) == [0, 1, 2, 3, 4, 5]


def test_config_with_no_expert_count_is_dense_not_an_error():
    assert resolve_moe_layers({"num_hidden_layers": 3}) == []
