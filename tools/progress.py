from __future__ import annotations

import logging
import sys
import time
from typing import Any

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None


def _format_details(details: dict[str, Any] | None) -> str:
    if not details:
        return ""
    parts: list[str] = []
    for key, value in details.items():
        if value in (None, "", False):
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


class ProgressReporter:
    def __init__(
        self,
        *,
        total: int,
        description: str,
        enabled: bool,
        log_every: int = 25,
    ) -> None:
        self.total = max(0, int(total))
        self.description = str(description)
        self.log_every = max(1, int(log_every))
        self.count = 0
        self.started_at = time.time()
        self._use_tqdm = bool(enabled and self.total > 0 and sys.stderr.isatty() and _tqdm is not None)
        self._use_inline = bool(enabled and self.total > 0 and sys.stderr.isatty() and not self._use_tqdm)
        self._bar = (
            _tqdm(total=self.total, desc=self.description, unit="row", dynamic_ncols=True, leave=True)
            if self._use_tqdm
            else None
        )

    @property
    def active(self) -> bool:
        return self._use_tqdm or self._use_inline

    def advance(self, *, status: str, details: dict[str, Any] | None = None) -> None:
        if self.total <= 0:
            return

        self.count += 1
        elapsed_sec = max(0.0, time.time() - self.started_at)
        detail_text = _format_details(details)

        if self._bar is not None:
            self._bar.update(1)
            postfix: dict[str, Any] = {"status": status}
            if details:
                postfix.update({key: value for key, value in details.items() if value not in (None, "", False)})
            self._bar.set_postfix(postfix, refresh=self.count == self.total)
            if self.count == self.total:
                self._bar.refresh()
            return

        if self._use_inline:
            remaining = max(0, self.total - self.count)
            rate = (self.count / elapsed_sec) if elapsed_sec > 0 else 0.0
            eta_sec = (remaining / rate) if rate > 0 else None
            percent = (100.0 * self.count / self.total) if self.total > 0 else 100.0
            eta_text = f" eta={eta_sec:.1f}s" if eta_sec is not None else ""
            suffix = f" {detail_text}" if detail_text else ""
            line = (
                f"\r{self.description}: {self.count}/{self.total} ({percent:.1f}%) "
                f"status={status}{suffix} elapsed={elapsed_sec:.1f}s{eta_text}"
            )
            print(line, end="" if self.count < self.total else "\n", file=sys.stderr, flush=True)
            return

        if self.count % self.log_every == 0 or self.count == self.total:
            logging.info(
                "%s progress %s/%s status=%s %s elapsed_sec=%.1f",
                self.description,
                self.count,
                self.total,
                status,
                detail_text,
                elapsed_sec,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
