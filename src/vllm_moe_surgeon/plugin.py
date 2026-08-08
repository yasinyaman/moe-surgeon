# SPDX-License-Identifier: Apache-2.0
"""The ``vllm.general_plugins`` entry point.

vLLM calls :func:`register` in every process that touches the model -- process0
during ``EngineArgs.__post_init__``, the engine core, and each worker -- and makes
no promise about how many times. The only contract the plugin system states is
re-entrancy, so this is idempotent and never raises: a plugin that throws takes the
engine down with it, and a disk tier failing to install should degrade to a normal
vLLM run, not to no run at all.

Everything registered here lives behind :mod:`.compat`, so the offline half of the
package stays importable on a host with no vLLM.
"""

from __future__ import annotations

from ._logging import init_logger

logger = init_logger(__name__)

_installed = False


def register() -> None:
    """Install the out-of-tree expert cache. Safe to call repeatedly."""
    global _installed
    if _installed:
        return

    try:
        from .compat.runtime import install

        took = install()
    except Exception as exc:
        # Deliberately broad. vLLM catches and logs plugin exceptions, but an
        # engine that boots without the tier is far better than one that does not
        # boot, and the reason has to be visible either way.
        logger.warning(
            "moe-surgeon did not install its runtime (%s: %s); vLLM will run "
            "without the expert cache",
            type(exc).__name__,
            exc,
        )
        _installed = True
        return

    _installed = True
    if took:
        logger.info("moe-surgeon runtime installed")
    else:
        logger.debug(
            "moe-surgeon runtime not installed (already registered, or vLLM absent)"
        )
