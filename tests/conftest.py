"""Run the prototype's test harness (prep/kernel/test_kernel.py) against src/kernel.py.

The harness imports `kernel.FlashGNM`; we register src.kernel under that name before it
is imported, and reuse prep/kernel/conftest.py's options and fixtures.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "prep" / "kernel"
for p in (str(ROOT), str(PROTO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import src.kernel  # noqa: E402

sys.modules["kernel"] = src.kernel

_spec = importlib.util.spec_from_file_location("proto_conftest", PROTO / "conftest.py")
_proto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_proto)

pytest_addoption = _proto.pytest_addoption
pytest_configure = _proto.pytest_configure
pytest_collection_modifyitems = _proto.pytest_collection_modifyitems
device = _proto.device
triton_runnable = _proto.triton_runnable
