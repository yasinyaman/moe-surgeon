# SPDX-License-Identifier: Apache-2.0
"""Make ``tests/`` importable so the ported suite can reach ``_support``.

The upstream suite lived inside a package with a conftest that did this for it.
Keeping the same flat-import style avoids touching ~40 call sites in the ported
file, which would have turned a mechanical lift into a rewrite.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
