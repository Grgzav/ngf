from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from .config import settings


@dataclass(frozen=True)
class OnetHttpClient:
    """Thin HTTP client for O*NET Web Services v2."""

    base_url: str = settings.onet_base_url

    def _headers(self) -> dict[str, str]:
        if not settings.onet_api_key or settings.onet_api_key.strip() == "PUT_YOUR_KEY_HERE":
            raise RuntimeError(
                "Missing ONET_API_KEY. Copy .env.example -> .env and set ONET_API_KEY."
            )
        return {"X-API-Key": settings.onet_api_key}

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        resp = requests.get(url, headers=self._headers(), params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()