"""минимальный клиент для gigachat api

зачем отдельный модуль:
- не размазывать oauth/токены по app.py
- проще тестировать и переиспользовать
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


class GigaChatAPIError(RuntimeError):
    """ошибка при работе с gigachat api"""

    def __init__(self, message: str, *, status_code: Optional[int] = None, details: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


@dataclass
class _Token:
    access_token: str
    expires_at: float  # unix timestamp (seconds)


class GigaChatClient:
    """тонкая обёртка над rest api gigachat (oauth + chat/completions)"""

    def __init__(
        self,
        *,
        credentials: str,
        scope: str = "GIGACHAT_API_PERS",
        model: str = "GigaChat",
        api_base_url: str = "https://gigachat.devices.sberbank.ru",
        oauth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        verify_ssl_certs: bool = True,
        timeout: int = 60,
    ):
        self.credentials = (credentials or "").strip()
        self.scope = (scope or "GIGACHAT_API_PERS").strip()
        self.model = (model or "GigaChat").strip()
        self.api_base_url = api_base_url.rstrip("/")
        self.oauth_url = oauth_url
        self.verify_ssl_certs = bool(verify_ssl_certs)
        self.timeout = int(timeout)

        self._token: Optional[_Token] = None

    def _build_oauth_auth_header(self) -> str:
        """Authorization header для POST /api/v2/oauth.

        в studio обычно отдают base64 строку без префикса.
        но сам oauth эндпоинт использует basic-схему.
        """
        if not self.credentials:
            raise GigaChatAPIError("не задан gigachat credentials (authorization key)")

        if self.credentials.lower().startswith("basic "):
            return self.credentials

        return f"Basic {self.credentials}"

    @staticmethod
    def _parse_expires_at(value: Any) -> float:
        """приводим expires_at / expires_in к unix timestamp"""
        now = time.time()

        # чаще всего прилетает либо expires_at (в мс), либо expires_in (в сек)
        if value is None:
            return now + 30 * 60

        # строка -> число
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return now + 30 * 60

        if isinstance(value, (int, float)):
            value = float(value)

            # похоже на unix ms
            if value > 1_000_000_000_000:
                return value / 1000.0

            # похоже на unix seconds
            if value > 1_000_000_000:
                return value

            # иначе считаем, что это expires_in
            return now + value

        return now + 30 * 60

    def get_access_token(self, *, force_refresh: bool = False) -> str:
        """получить (и закешировать) access token"""
        if not force_refresh and self._token:
            # небольшой запас, чтобы не ловить 401 на границе
            if time.time() < self._token.expires_at - 20:
                return self._token.access_token

        headers = {
            "Authorization": self._build_oauth_auth_header(),
            # rqUID часто советуют добавлять для трассировки
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        # scope должен быть именно form-data/x-www-form-urlencoded
        data = {"scope": self.scope}

        resp = requests.post(
            self.oauth_url,
            headers=headers,
            data=data,
            timeout=self.timeout,
            verify=self.verify_ssl_certs,
        )

        if resp.status_code != 200:
            raise GigaChatAPIError(
                "не удалось получить access token",
                status_code=resp.status_code,
                details=resp.text[:500],
            )

        payload = resp.json()
        access_token = payload.get("access_token")
        if not access_token:
            raise GigaChatAPIError("oauth ответ без access_token", details=str(payload)[:500])

        expires_at = self._parse_expires_at(payload.get("expires_at") or payload.get("expires_in"))
        self._token = _Token(access_token=access_token, expires_at=expires_at)
        return access_token

    def chat_completions(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """POST /api/v1/chat/completions"""
        url = f"{self.api_base_url}/api/v1/chat/completions"
        token = self.get_access_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
            verify=self.verify_ssl_certs,
        )

        # токен мог протухнуть чуть раньше (бывает)
        if resp.status_code == 401:
            token = self.get_access_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                verify=self.verify_ssl_certs,
            )

        if resp.status_code != 200:
            raise GigaChatAPIError(
                "ошибка запроса chat/completions",
                status_code=resp.status_code,
                details=resp.text[:500],
            )

        return resp.json()
