# SPDX-License-Identifier: Apache-2.0
"""A logger factory that stands in for ``vllm.logger``.

The lifted disk-tier modules call ``logger.warning_once(...)``, which is a vLLM
extension rather than a stdlib method. Reimplementing it here is what lets those
modules stop importing vLLM at all -- and the disk store, the provider and the
policies are then usable from the offline surgery pipeline on a machine with no
vLLM and no CUDA.

Dedup key is the *unformatted* message plus args, so a warning that fires once
per layer with a different layer name in it still prints once per layer, while a
genuinely repeated warning prints once.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class _SurgeonLogger(logging.Logger):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._seen: set[tuple[Any, ...]] = set()
        self._seen_lock = threading.Lock()

    def warning_once(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at WARNING, but only the first time this message is seen."""
        key = (msg, args)
        with self._seen_lock:
            if key in self._seen:
                return
            self._seen.add(key)
        self.warning(msg, *args, **kwargs)


_configured = False
_configure_lock = threading.Lock()


def _configure_root() -> None:
    global _configured
    with _configure_lock:
        if _configured:
            return
        _configured = True
        level = os.environ.get("MOE_SURGEON_LOG_LEVEL", "INFO").upper()
        root = logging.getLogger("vllm_moe_surgeon")
        # Do not touch handlers when something else already set them up -- under
        # vLLM the process already has logging configured and adding a second
        # handler would duplicate every line.
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
            root.addHandler(handler)
        root.setLevel(getattr(logging, level, logging.INFO))


def init_logger(name: str) -> _SurgeonLogger:
    """Return a logger for ``name`` that also has ``warning_once``."""
    _configure_root()
    # setLoggerClass is global state, so set it only for the duration of the
    # getLogger call rather than leaving it installed for the whole process.
    previous = logging.getLoggerClass()
    logging.setLoggerClass(_SurgeonLogger)
    try:
        logger = logging.getLogger(name)
    finally:
        logging.setLoggerClass(previous)
    if not isinstance(logger, _SurgeonLogger):
        # Something already created this logger under a different class. Graft
        # the one method we add rather than failing at import time.
        logger.warning_once = _SurgeonLogger.warning_once.__get__(  # type: ignore[attr-defined]
            logger, type(logger)
        )
        logger._seen = set()  # type: ignore[attr-defined]
        logger._seen_lock = threading.Lock()  # type: ignore[attr-defined]
    return logger  # type: ignore[return-value]
