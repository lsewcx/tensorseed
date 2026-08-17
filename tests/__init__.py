"""TensorSeed test suite package."""

from __future__ import annotations

import sys
from pathlib import Path

_PYTHON_PATH = Path(__file__).resolve().parent.parent / "python"
if str(_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(_PYTHON_PATH))
