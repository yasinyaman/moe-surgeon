# SPDX-License-Identifier: Apache-2.0
"""Permanent MoE expert surgery for domain-specialised vLLM deployments.

Importing this package must never import vLLM. The two halves are:

- **offline** (:mod:`.surgery`, :mod:`.writer`, :mod:`.server`, :mod:`.store`) --
  reads and writes checkpoints, decides and applies expert placement. Runs on a
  laptop with no GPU and no vLLM installed.
- **runtime** (:mod:`.telemetry`, :mod:`.compat`, :mod:`.plugin`) -- loaded by
  vLLM through the ``vllm.general_plugins`` entry point. Everything that knows
  a vLLM internal name lives behind :mod:`.compat`.

``tests/test_seams.py`` is the contract for the second half: on a vLLM upgrade it
is what fails first.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
