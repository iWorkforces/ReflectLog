"""Conftest for memory subsystem tests.

Disables numba JIT so that ranx (which depends on numba) can import
cleanly alongside coverage instrumentation.
"""

import importlib
import os
import sys

os.environ["NUMBA_DISABLE_JIT"] = "1"

_numba_modules = [
    key for key in sys.modules if key == "numba" or key.startswith("numba.")
]
for mod_name in _numba_modules:
    del sys.modules[mod_name]

import numba  # noqa: E402

importlib.reload(numba)

_reflectlog_numba = "reflectlog.application.utils.numba_utils"
if _reflectlog_numba in sys.modules:
    importlib.reload(sys.modules[_reflectlog_numba])
