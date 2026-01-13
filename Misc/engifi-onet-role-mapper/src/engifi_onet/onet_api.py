from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .http import OnetHttpClient


@dataclass(frozen=True)
class OnetApi:
    """
    Minimal wrapper around O*NET Web Services v2 endpoints we need.
    Start with /online/search (keyword search).
    """
    http: OnetHttpClient = OnetHttpClient()

    def search(self, keyword: str, start: int = 1, end: int = 25) -> dict[str, Any]:
        params = {"keyword": keyword, "start": start, "end": end}
        return self.http.get_json("/online/search", params=params)