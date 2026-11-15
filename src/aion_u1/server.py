"""Bounded local Chat Completions transport for TankAI's existing adapter."""
from __future__ import annotations

import hmac
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pydantic import ValidationError

from .inference import ChatRequest, InferenceEngine, InferenceInputError

MAX_REQUEST_BYTES = 1_048_576


class InferenceServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(
        self,
        address: tuple[str, int],
        engine: InferenceEngine,
        api_key: str,
        *,
        max_new_tokens: int = 2048,
    ) -> None:
        if not api_key or api_key != api_key.strip():
            raise ValueError("AION_U1_API_KEY fehlt oder enthält äußere Leerzeichen.")
        if not 1 <= max_new_tokens <= 8192:
            raise ValueError("Ungültiges Ausgabetokenlimit.")
        self.engine = engine
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.generation_slot = threading.BoundedSemaphore(1)
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: InferenceServer
    server_version = "AION-U1"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format: str, *args: Any) -> None:
        # Prompts, bearer tokens and request paths never enter access logs.
        pass

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, status: int, code: str, message: str) -> None:
        self._json(status, {
            "error": {"message": message, "type": "invalid_request_error", "code": code}
        })

    def _discard_small_overflow(self, length: int) -> None:
        # Draining a bounded overflow lets Windows deliver the 413 response
        # without resetting a connection that still contains unread bytes.
        if length > 2 * MAX_REQUEST_BYTES:
            return
        remaining = length
        self.connection.settimeout(1)
        try:
            while remaining:
                chunk = self.rfile.read(min(remaining, 65_536))
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            pass

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        self._error(code, "unsupported_request", "Nicht unterstützte HTTP-Anfrage.")

    def _authorized(self) -> bool:
        expected = ("Bearer " + self.server.api_key).encode("utf-8")
        supplied = self.headers.get("Authorization", "").encode("utf-8")
        if not hmac.compare_digest(supplied, expected):
            self._error(401, "invalid_api_key", "Authentifizierung erforderlich.")
            return False
        return True

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {
                "status": "ready",
                "model": self.server.engine.model_id,
                "quality_verified": False,
                "production_ready": False,
            })
            return
        if not self._authorized():
            return
        if self.path == "/v1/models":
            self._json(200, {
                "object": "list",
                "data": [{
                    "id": self.server.engine.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "tankai",
                }],
            })
            return
        self._error(404, "not_found", "Endpoint nicht gefunden.")

    def do_POST(self) -> None:
        if not self._authorized():
            return
        if self.path != "/v1/chat/completions":
            self._error(404, "not_found", "Endpoint nicht gefunden.")
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self._error(400, "invalid_request", "Transfer-Encoding wird nicht unterstützt.")
            return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            self._error(415, "invalid_content_type", "application/json erforderlich.")
            return
        size = self.headers.get("Content-Length", "")
        if not size.isascii() or not size.isdigit() or len(size) > 10:
            self._error(411, "length_required", "Content-Length erforderlich.")
            return
        length = int(size)
        if not 1 <= length <= MAX_REQUEST_BYTES:
            self._discard_small_overflow(length)
            self._error(413, "request_too_large", "Ungültige Anfragegröße.")
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete body")
            payload = json.loads(body, object_pairs_hook=_unique_object)
            request = ChatRequest.model_validate(payload)
        except (ValueError, UnicodeError, ValidationError, OSError, RecursionError):
            self._error(400, "invalid_request", "Ungültige Chat-Anfrage.")
            return
        if request.model != self.server.engine.model_id:
            self._error(404, "model_not_found", "Unbekanntes Modell.")
            return
        if request.stream or request.max_tokens > self.server.max_new_tokens:
            self._error(400, "unsupported_request", "Streaming oder Ausgabelimit nicht unterstützt.")
            return
        if not self.server.generation_slot.acquire(blocking=False):
            self._error(503, "model_busy", "Modell ist ausgelastet.")
            return
        try:
            completion = self.server.engine.complete(request)
        except InferenceInputError:
            self._error(400, "context_or_template_error", "Kontextlimit oder Chat-Template verletzt.")
            return
        except Exception:
            self._error(503, "inference_failed", "Lokale Inferenz fehlgeschlagen.")
            return
        finally:
            self.server.generation_slot.release()
        self._json(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.server.engine.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": completion.content},
                "finish_reason": completion.finish_reason,
            }],
            "usage": {
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.prompt_tokens + completion.completion_tokens,
            },
        })


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result
