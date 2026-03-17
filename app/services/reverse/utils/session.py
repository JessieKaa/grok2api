"""
Resettable session wrapper for reverse requests.
"""

import asyncio
import time
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

from curl_cffi.requests import AsyncSession
from curl_cffi.const import CurlOpt

from app.core.config import get_config
from app.core.logger import logger

_SENSITIVE_FIELDS = {
    "authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
    "token",
    "access_token",
    "refresh_token",
    "password",
    "passwd",
    "secret",
    "session",
}

_BODY_PREVIEW_LIMIT = 2000


def _should_skip_proxy_ssl() -> bool:
    return bool(get_config("proxy.skip_proxy_ssl_verify")) and bool(
        get_config("proxy.base_proxy_url")
    )


def _is_sensitive_key(key: str) -> bool:
    k = (key or "").strip().lower()
    return k in _SENSITIVE_FIELDS or any(
        part in k for part in ("cookie", "token", "secret", "password", "authorization")
    )


def _truncate_text(value: str, limit: int = 256) -> str:
    if not isinstance(value, str):
        return str(value)
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...(+{len(value) - limit} chars)"


def _sanitize_headers(headers: Any) -> dict:
    if not isinstance(headers, dict):
        return {}
    sanitized = {}
    for key, value in headers.items():
        key_str = str(key)
        if _is_sensitive_key(key_str):
            sanitized[key_str] = "[REDACTED]"
        else:
            sanitized[key_str] = _truncate_text(str(value), 512)
    return sanitized


def _summarize_payload(value: Any, depth: int = 0) -> Any:
    if value is None:
        return None
    if depth >= 2:
        if isinstance(value, (dict, list, tuple)):
            return f"<{type(value).__name__} size={len(value)}>"
        return _truncate_text(str(value), 200)
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, str):
        return _truncate_text(value, 500)
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key_str = str(k)
            if _is_sensitive_key(key_str):
                out[key_str] = "[REDACTED]"
            else:
                out[key_str] = _summarize_payload(v, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        limit = 10
        summarized = [_summarize_payload(v, depth + 1) for v in value[:limit]]
        if len(value) > limit:
            summarized.append(f"...(+{len(value) - limit} items)")
        return summarized
    return _truncate_text(str(value), 300)


def _safe_host(url: Any) -> str:
    if not isinstance(url, str):
        return ""
    try:
        return urlsplit(url).netloc
    except Exception:
        return ""


class ResettableSession:
    """AsyncSession wrapper that resets connection on specific HTTP status codes."""

    def __init__(
        self,
        *,
        reset_on_status: Optional[Iterable[int]] = None,
        **session_kwargs: Any,
    ):
        self._session_kwargs = dict(session_kwargs)
        if not self._session_kwargs.get("impersonate"):
            browser = get_config("proxy.browser")
            if browser:
                self._session_kwargs["impersonate"] = browser
        config_codes = get_config("retry.reset_session_status_codes")
        if reset_on_status is None:
            reset_on_status = config_codes if config_codes is not None else [403]
        if isinstance(reset_on_status, int):
            reset_on_status = [reset_on_status]
        self._reset_on_status = (
            {int(code) for code in reset_on_status} if reset_on_status else set()
        )
        self._skip_proxy_ssl = _should_skip_proxy_ssl()
        self._reset_requested = False
        self._reset_lock = asyncio.Lock()
        self._session = self._create_session()

    def _create_session(self) -> AsyncSession:
        kwargs = dict(self._session_kwargs)
        if self._skip_proxy_ssl:
            opts = kwargs.get("curl_options", {})
            opts[CurlOpt.PROXY_SSL_VERIFYPEER] = 0
            opts[CurlOpt.PROXY_SSL_VERIFYHOST] = 0
            kwargs["curl_options"] = opts
        return AsyncSession(**kwargs)

    async def _maybe_reset(self) -> None:
        if not self._reset_requested:
            return
        async with self._reset_lock:
            if not self._reset_requested:
                return
            self._reset_requested = False
            old_session = self._session
            self._session = self._create_session()
            try:
                await old_session.close()
            except Exception:
                pass
            logger.debug("ResettableSession: session reset")

    async def _request(self, method: str, *args: Any, **kwargs: Any):
        await self._maybe_reset()
        start = time.perf_counter()
        url = args[0] if args else kwargs.get("url", "")
        stream = bool(kwargs.get("stream", False))
        req_log = {
            "method": method.upper(),
            "url": str(url),
            "host": _safe_host(url),
            "stream": stream,
            "timeout": kwargs.get("timeout"),
            "proxy": kwargs.get("proxy"),
            "proxies": _summarize_payload(kwargs.get("proxies")),
            "params": _summarize_payload(kwargs.get("params")),
            "headers": _sanitize_headers(kwargs.get("headers")),
            "json": _summarize_payload(kwargs.get("json")),
            "data": _summarize_payload(kwargs.get("data")),
        }
        logger.info("Grok outbound request", extra={"http_request": req_log})
        try:
            response = await getattr(self._session, method)(*args, **kwargs)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "Grok outbound request failed",
                extra={
                    "http_request": req_log,
                    "elapsed_ms": elapsed_ms,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        resp_log = {
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
            "headers": _sanitize_headers(dict(response.headers or {})),
            "content_type": response.headers.get("content-type", ""),
            "stream": stream,
        }
        if stream:
            resp_log["body_preview"] = "<streaming response>"
        else:
            try:
                body_text = response.text or ""
                resp_log["body_preview"] = _truncate_text(body_text, _BODY_PREVIEW_LIMIT)
            except Exception:
                resp_log["body_preview"] = "<unavailable>"

        logger.info("Grok outbound response", extra={"http_response": resp_log, "http_request": req_log})

        if self._reset_on_status and response.status_code in self._reset_on_status:
            self._reset_requested = True
        return response

    async def get(self, *args: Any, **kwargs: Any):
        return await self._request("get", *args, **kwargs)

    async def post(self, *args: Any, **kwargs: Any):
        return await self._request("post", *args, **kwargs)

    async def reset(self) -> None:
        self._reset_requested = True
        await self._maybe_reset()

    async def close(self) -> None:
        if self._session is None:
            return
        try:
            await self._session.close()
        finally:
            self._session = None
            self._reset_requested = False

    async def __aenter__(self) -> "ResettableSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


__all__ = ["ResettableSession"]
