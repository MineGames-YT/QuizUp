"""минимальный клиент для gemini api (google ai for developers)

поддержка:
- models:generateContent
- опционально: grounding with google search через tool google_search

почему отдельный модуль:
- чтобы app.py оставался читаемым
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


class GeminiAPIError(RuntimeError):
    """ошибка при работе с gemini api"""

    def __init__(self, message: str, *, status_code: Optional[int] = None, details: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


@dataclass
class GeminiClient:
    """тонкая обёртка над rest gemini api"""

    api_key: str
    model: str = "gemini-2.5-pro"
    api_base_url: str = "https://generativelanguage.googleapis.com"
    timeout: int = 60

    def generate_content(
        self,
        *,
        prompt: str,
        use_search: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 3500,
        response_mime_type: str = "application/json",
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise GeminiAPIError("не задан gemini api key")

        url = f"{self.api_base_url.rstrip('/')}/v1beta/models/{self.model}:generateContent"

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # если включён поиск, gemini может ругаться на responseMimeType=application/json
        if use_search and response_mime_type == "application/json":
            response_mime_type = "text/plain"

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
                # помогает получить именно json, а не красивый текст с пояснениями
                "responseMimeType": response_mime_type,
            },
        }

        # grounding with google search
        if use_search:
            payload["tools"] = [
                {
                    "google_search": {},
                }
            ]

        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
        except Exception as e:
            raise GeminiAPIError(f"не удалось вызвать gemini api: {e}") from e

        if resp.status_code != 200:
            raise GeminiAPIError(
                "ошибка запроса generateContent",
                status_code=resp.status_code,
                details=(resp.text or "")[:800],
            )

        return resp.json()


def extract_text(response_json: Dict[str, Any]) -> str:
    """вытаскиваем текст первого кандидата

    при response_mime_type=application/json gemini чаще всего возвращает json строкой в parts[].text
    """

    try:
        candidates = response_json.get("candidates") or []
        if not candidates:
            return ""

        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            return ""

        # обычно первый part с text
        text = parts[0].get("text")
        return text if isinstance(text, str) else ""
    except Exception:
        return ""
