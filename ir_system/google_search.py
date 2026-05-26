from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import os


@dataclass(frozen=True)
class WebResult:
    title: str
    link: str
    snippet: str
    source: str = "Google"


class GoogleSearchClient:
    def __init__(self, api_key: str = "", cx: str = "") -> None:
        self.api_key = api_key.strip()
        self.cx = cx.strip()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.cx)

    @classmethod
    def from_environment(cls) -> "GoogleSearchClient":
        api_key = os.getenv("GOOGLE_CUSTOM_SEARCH_KEY") or os.getenv("GOOGLE_API_KEY", "")
        cx = os.getenv("GOOGLE_CUSTOM_SEARCH_CX") or os.getenv("GOOGLE_CSE_ID", "")
        return cls(api_key=api_key, cx=cx)

    @lru_cache(maxsize=128)
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.enabled or not query.strip():
            return []

        params = urlencode(
            {
                "q": query,
                "key": self.api_key,
                "cx": self.cx,
                "num": max(1, min(10, top_k)),
                "safe": "active",
            }
        )
        url = f"https://www.googleapis.com/customsearch/v1?{params}"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

        try:
            with urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return []

        results: list[dict[str, Any]] = []
        for item in payload.get("items", [])[:top_k]:
            results.append(
                {
                    "title": item.get("title", "Untitled result"),
                    "link": item.get("link", "#"),
                    "snippet": item.get("snippet", ""),
                    "display_link": item.get("displayLink", ""),
                    "source": "Google",
                }
            )
        return results
