# SPDX-License-Identifier: Apache-2.0
"""Decode the routed-experts payload the OpenAI-compatible server returns.

JSON cannot carry a raw ndarray, so vLLM's serving layer writes the array with
``np.save`` into a buffer and base64-encodes it
(``entrypoints/openai/completion/serving.py``). This is the read side.

The field is ``None`` when either the request did not finish or the server was
started without ``--enable-return-routed-experts``. Those two cases are worth
distinguishing to the operator, because the second one silently produces an
empty profile.
"""

from __future__ import annotations

import base64
import io

import numpy as np


class RoutedExpertsUnavailable(RuntimeError):
    """The response carried no routing data."""


def decode(payload: str | None, *, strict: bool = True) -> np.ndarray | None:
    """Decode one response's ``routed_experts`` field.

    Returns ``[seq_len, num_layers, top_k]``. With ``strict``, a missing payload
    raises instead of returning ``None`` -- profiling silently over responses
    that carry no routing is how you end up with a confident, empty profile.
    """
    if payload is None:
        if strict:
            raise RoutedExpertsUnavailable(
                "response has no routed_experts. Either the request did not "
                "finish, or the server was started without "
                "--enable-return-routed-experts."
            )
        return None
    raw = base64.b64decode(payload)
    # allow_pickle stays False: this buffer arrives over the network, and a
    # pickled payload would be arbitrary code execution.
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    if array.ndim != 3:
        raise ValueError(
            f"decoded routed_experts has shape {array.shape}, expected 3 dims"
        )
    return array


def iter_choices(response: dict, *, strict: bool = True):
    """Yield decoded captures from an OpenAI completion/chat response dict."""
    for choice in response.get("choices", ()):
        decoded = decode(choice.get("routed_experts"), strict=strict)
        if decoded is not None:
            yield decoded
