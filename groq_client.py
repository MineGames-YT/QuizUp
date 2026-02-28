#минимальный клиент для groq 

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import re

import httpx


class GroqAPIError(RuntimeError):
    """ошибка при работе с groq api"""

    def __init__(self, message: str, *, status_code: Optional[int] = None, details: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


def extract_text(resp_json: Dict[str, Any]) -> str:
    """достаёт message.content из openai-совместимого ответа"""
    try:
        return (resp_json.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
    except Exception:
        return ""


def _extract_json_object(text: str) -> Optional[dict]:
    """пытается вытащить json-объект из строки (на случай лишних слов)"""
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass


    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    m2 = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m2:
        candidate = m2.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


@dataclass
class GroqClient:
    api_key: str
    model: str = "llama-3.3-70b-versatile"
    api_base_url: str = "https://api.groq.com/openai/v1"
    timeout: int = 60

    def chat_completions(
        self,
        *,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1800,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise GroqAPIError("не задан groq api key")

        base = (self.api_base_url or "").rstrip("/")
        if base.endswith("/openai/v1"):
            url = f"{base}/chat/completions"
        elif base.endswith("/v1"):
            url = f"{base}/chat/completions"
        else:
            url = f"{base}/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "ты генерируешь вопросы для викторины. отвечай строго по инструкции пользователя."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(url, headers=headers, json=payload)
        except httpx.RequestError as e:
            raise GroqAPIError(f"ошибка сети при запросе groq: {e}") from e

        if r.status_code >= 400:
            details = None
            try:
                details = r.text
            except Exception:
                details = None
            raise GroqAPIError(
                f"ошибка запроса chat completions (status={r.status_code})",
                status_code=r.status_code,
                details=details,
            )

        try:
            return r.json()
        except Exception as e:
            raise GroqAPIError("не удалось распарсить json от groq", status_code=r.status_code, details=r.text) from e

    def generate_questions_json(
        self,
        *,
        prompt: str,
        temperature: float = 0.6,
        max_tokens: int = 2200,
    ) -> Dict[str, Any]:
        """возвращает dict; если модель не вернула json — бросаем ошибку"""
        resp = self.chat_completions(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
        text = extract_text(resp)
        obj = _extract_json_object(text)
        if not isinstance(obj, dict):
            raise GroqAPIError("модель вернула не-json ответ", details=text[:2000])
        return obj
