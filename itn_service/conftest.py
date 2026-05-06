"""itn_service-local pytest configuration.

Adds the repo root (the directory that *contains* ``itn_service``) to
``sys.path`` so that ``from itn_service.runtime import ...`` resolves
when pytest is run from inside ``itn_service/`` directly. Mirrors the
top-level ``conftest.py`` pattern used by the wider repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
