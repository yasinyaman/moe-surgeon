# SPDX-License-Identifier: Apache-2.0
"""Per-expert usage telemetry, built on vLLM's public capture API.

The plan assumed this phase would need a ``MoERunner`` subclass registered
through ``PluggableLayer.register_oot``, wrapping ``router.select_experts`` to
read ``topk_ids``. It does not. vLLM already captures exactly that, through
``--enable-return-routed-experts``, and returns it per request:

- offline: ``LLM(..., enable_return_routed_experts=True)`` then
  ``out.outputs[0].routed_experts`` -- ``[seq_len, num_layers, top_k]``
- served: start with ``--enable-return-routed-experts``; the response field
  ``routed_experts`` carries a base64 ``np.save`` buffer (see :mod:`.transport`)

So both telemetry sources cost **zero seams**: this whole subpackage is numpy
over a documented output field, and it dropped three internal seams from
``compat/seams.py`` rather than adding any.

What the public API does not give us is the router *weights* -- the capture is
ids only. Token-weighted counts and co-occurrence are exact; gate mass is not
available without reintroducing a routing hook, and is deferred.
"""

from .layers import get_num_experts, get_top_k, resolve_moe_layers
from .persist import load, save
from .stats import ExpertStats, accumulate, from_hf_config
from .transport import RoutedExpertsUnavailable, decode, iter_choices

__all__ = [
    "ExpertStats",
    "RoutedExpertsUnavailable",
    "accumulate",
    "decode",
    "from_hf_config",
    "get_num_experts",
    "get_top_k",
    "iter_choices",
    "load",
    "resolve_moe_layers",
    "save",
]
