"""
WebSocket helpers for reverse interfaces.
"""

import time
import ssl
import certifi
import aiohttp
from aiohttp_socks import ProxyConnector
from typing import Mapping, Optional, Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunsplit

from app.core.logger import logger
from app.core.config import get_config

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


def _is_sensitive_key(key: str) -> bool:
    k = (key or "").strip().lower()
    return k in _SENSITIVE_FIELDS or any(
        part in k for part in ("cookie", "token", "secret", "password", "authorization")
    )


def _truncate_text(value: Any, limit: int = 500) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(+{len(text) - limit} chars)"


def _sanitize_headers(headers: Optional[Mapping[str, str]]) -> dict[str, str]:
    if not headers:
        return {}
    sanitized: dict[str, str] = {}
    for key, value in headers.items():
        key_str = str(key)
        if _is_sensitive_key(key_str):
            sanitized[key_str] = "[REDACTED]"
        else:
            sanitized[key_str] = _truncate_text(value, 512)
    return sanitized


def _sanitize_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    try:
        parsed = urlparse(url)
        if not parsed.query:
            return url
        masked_pairs = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            if _is_sensitive_key(key):
                masked_pairs.append((key, "[REDACTED]"))
            else:
                masked_pairs.append((key, value))
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(masked_pairs, doseq=True),
                parsed.fragment,
            )
        )
    except Exception:
        return url


def _default_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.load_verify_locations(certifi.where())
    return context


def _normalize_socks_proxy(proxy_url: str) -> tuple[str, Optional[bool]]:
    scheme = urlparse(proxy_url).scheme.lower()
    rdns: Optional[bool] = None
    base_scheme = scheme

    if scheme == "socks5h":
        base_scheme = "socks5"
        rdns = True
    elif scheme == "socks4a":
        base_scheme = "socks4"
        rdns = True

    if base_scheme != scheme:
        proxy_url = proxy_url.replace(f"{scheme}://", f"{base_scheme}://", 1)

    return proxy_url, rdns


def resolve_proxy(proxy_url: Optional[str] = None, ssl_context: ssl.SSLContext = _default_ssl_context()) -> tuple[aiohttp.BaseConnector, Optional[str]]:
    """Resolve proxy connector.
    
    Args:
        proxy_url: Optional[str], the proxy URL. Defaults to None.
        ssl_context: ssl.SSLContext, the SSL context. Defaults to _default_ssl_context().

    Returns:
        tuple[aiohttp.BaseConnector, Optional[str]]: The proxy connector and the proxy URL.
    """
    if not proxy_url:
        return aiohttp.TCPConnector(ssl=ssl_context), None

    scheme = urlparse(proxy_url).scheme.lower()
    if scheme.startswith("socks"):
        normalized, rdns = _normalize_socks_proxy(proxy_url)
        logger.info(f"Using SOCKS proxy: {proxy_url}")
        try:
            if rdns is not None:
                return (
                    ProxyConnector.from_url(normalized, rdns=rdns, ssl=ssl_context),
                    None,
                )
        except TypeError:
            return ProxyConnector.from_url(normalized, ssl=ssl_context), None
        return ProxyConnector.from_url(normalized, ssl=ssl_context), None

    logger.info(f"Using HTTP proxy: {proxy_url}")
    return aiohttp.TCPConnector(ssl=ssl_context), proxy_url


class WebSocketConnection:
    """WebSocket connection wrapper."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        ws: aiohttp.ClientWebSocketResponse,
        *,
        url: str = "",
    ) -> None:
        self.session = session
        self.ws = ws
        self.url = _sanitize_url(url)
        self._opened_at = time.perf_counter()
        self._send_count = 0
        self._recv_count = 0

    async def send_json(self, data: Any, compress: Optional[int] = None, *, dumps=None) -> None:
        self._send_count += 1
        payload_preview = _truncate_text(data, 1500)
        logger.debug(
            "Grok websocket send",
            extra={
                "ws_event": {
                    "url": self.url,
                    "type": "json",
                    "count": self._send_count,
                    "payload_preview": payload_preview,
                }
            },
        )
        await self.ws.send_json(data, compress=compress, dumps=dumps)

    async def send_str(self, data: str, compress: Optional[int] = None) -> None:
        self._send_count += 1
        logger.debug(
            "Grok websocket send",
            extra={
                "ws_event": {
                    "url": self.url,
                    "type": "text",
                    "count": self._send_count,
                    "payload_preview": _truncate_text(data, 1500),
                }
            },
        )
        await self.ws.send_str(data, compress=compress)

    async def send_bytes(self, data: bytes, compress: Optional[int] = None) -> None:
        self._send_count += 1
        logger.debug(
            "Grok websocket send",
            extra={
                "ws_event": {
                    "url": self.url,
                    "type": "bytes",
                    "count": self._send_count,
                    "size": len(data or b""),
                }
            },
        )
        await self.ws.send_bytes(data, compress=compress)

    async def receive(self, timeout: Optional[float] = None) -> aiohttp.WSMessage:
        msg = await self.ws.receive(timeout=timeout)
        self._recv_count += 1
        details: dict[str, Any] = {
            "url": self.url,
            "count": self._recv_count,
            "msg_type": str(msg.type),
        }
        if msg.type == aiohttp.WSMsgType.TEXT:
            details["payload_preview"] = _truncate_text(msg.data, 1500)
        elif msg.type == aiohttp.WSMsgType.BINARY:
            details["size"] = len(msg.data or b"")
        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
            details["extra"] = _truncate_text(msg.extra, 500)
        logger.debug("Grok websocket receive", extra={"ws_event": details})
        return msg

    async def close(self) -> None:
        if not self.ws.closed:
            await self.ws.close()
        elapsed_ms = round((time.perf_counter() - self._opened_at) * 1000, 2)
        logger.info(
            "Grok websocket closed",
            extra={
                "ws_close": {
                    "url": self.url,
                    "elapsed_ms": elapsed_ms,
                    "send_count": self._send_count,
                    "recv_count": self._recv_count,
                }
            },
        )
        await self.session.close()

    async def __aenter__(self) -> "WebSocketConnection":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.ws, name)


class WebSocketClient:
    """WebSocket client with proxy support."""

    def __init__(self, proxy: Optional[str] = None) -> None:
        self._proxy_override = proxy
        self._ssl_context = _default_ssl_context()

    async def connect(
        self,
        url: str,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
        ws_kwargs: Optional[Mapping[str, object]] = None,
    ) -> WebSocketConnection:
        """Connect to the WebSocket.
        
        Args:
            url: str, the URL to connect to.
            headers: Optional[Mapping[str, str]], the headers to send. Defaults to None.
            ws_kwargs: Optional[Mapping[str, object]], extra ws_connect kwargs. Defaults to None.

        Returns:
            WebSocketConnection: The WebSocket connection.
        """
        # Resolve proxy dynamically from config if not overridden
        proxy_url = self._proxy_override or get_config("proxy.base_proxy_url")
        connector, resolved_proxy = resolve_proxy(proxy_url, self._ssl_context)
        safe_url = _sanitize_url(url)
        req_log = {
            "url": safe_url,
            "headers": _sanitize_headers(headers),
            "timeout": timeout,
            "proxy_url": _sanitize_url(proxy_url or ""),
            "resolved_proxy": _sanitize_url(resolved_proxy or ""),
            "connector": type(connector).__name__,
            "ws_kwargs": dict(ws_kwargs or {}),
        }
        logger.info("Grok websocket connect request", extra={"ws_connect_request": req_log})

        # Build client timeout
        total_timeout = (
            float(timeout)
            if timeout is not None
            else float(get_config("voice.timeout") or 120)
        )
        client_timeout = aiohttp.ClientTimeout(total=total_timeout)

        # Create session
        session = aiohttp.ClientSession(connector=connector, timeout=client_timeout)
        start = time.perf_counter()
        try:
            # Cast to Any to avoid Pylance errors with **extra_kwargs
            extra_kwargs: dict[str, Any] = dict(ws_kwargs or {})
            skip_proxy_ssl = bool(get_config("proxy.skip_proxy_ssl_verify")) and bool(proxy_url)
            if skip_proxy_ssl and urlparse(proxy_url).scheme.lower() == "https":
                proxy_ssl_context = ssl.create_default_context()
                proxy_ssl_context.check_hostname = False
                proxy_ssl_context.verify_mode = ssl.CERT_NONE
                try:
                    ws = await session.ws_connect(
                        url,
                        headers=headers,
                        proxy=resolved_proxy,
                        ssl=self._ssl_context,
                        proxy_ssl=proxy_ssl_context,
                        **extra_kwargs,
                    )
                except TypeError:
                    logger.warning(
                        "proxy.skip_proxy_ssl_verify is enabled, but aiohttp does not support proxy_ssl; keeping proxy SSL verification enabled"
                    )
                    ws = await session.ws_connect(
                        url,
                        headers=headers,
                        proxy=resolved_proxy,
                        ssl=self._ssl_context,
                        **extra_kwargs,
                    )
            else:
                ws = await session.ws_connect(
                    url,
                    headers=headers,
                    proxy=resolved_proxy,
                    ssl=self._ssl_context,
                    **extra_kwargs,
                )
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "Grok websocket connect response",
                extra={
                    "ws_connect_response": {
                        "url": safe_url,
                        "elapsed_ms": elapsed_ms,
                        "closed": ws.closed,
                    }
                },
            )
            return WebSocketConnection(session, ws, url=safe_url)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "Grok websocket connect failed",
                extra={
                    "ws_connect_request": req_log,
                    "elapsed_ms": elapsed_ms,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            await session.close()
            raise


__all__ = ["WebSocketClient", "WebSocketConnection", "resolve_proxy"]
