from __future__ import annotations

import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


try:
    import prometheus_client as _prometheus_client  # noqa: F401
except Exception:
    class _Metric:
        def labels(self, *args, **kwargs):
            del args, kwargs
            return self

        def inc(self, *args, **kwargs):
            del args, kwargs

        def dec(self, *args, **kwargs):
            del args, kwargs

        def observe(self, *args, **kwargs):
            del args, kwargs


    fake_module = types.ModuleType("prometheus_client")
    fake_module.Counter = lambda *args, **kwargs: _Metric()
    fake_module.Histogram = lambda *args, **kwargs: _Metric()
    fake_module.Gauge = lambda *args, **kwargs: _Metric()
    fake_module.CONTENT_TYPE_LATEST = "text/plain"
    fake_module.generate_latest = lambda: b""
    sys.modules["prometheus_client"] = fake_module


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "equivalence: numeric equivalence checks between two implementations "
        "(e.g. audioop.ratecv vs soxr.resample). Run by default; skip with "
        "`pytest -m \"not equivalence\"`.",
    )
