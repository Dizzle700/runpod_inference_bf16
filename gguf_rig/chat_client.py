"""Streaming chat client for vLLM OpenAI-compatible API."""
from __future__ import annotations

import base64
import http.client
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import RigConfig
from .process_manager import VllmServerManager
from .streaming import iter_sse_data


class ChatClient:
    """Streams chat completions from a local vLLM server."""

    def __init__(self, config: RigConfig, manager: VllmServerManager):
        self.config = config
        self.manager = manager

    def _audio_data_url(self, audio_path: str | None) -> str | None:
        """Convert a Gradio-uploaded audio file to a vLLM-compatible data URL."""
        if not audio_path:
            return None
        path = Path(audio_path)
        mime_types = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".mp4": "audio/mp4",
            ".webm": "audio/webm",
        }
        mime_type = mime_types.get(path.suffix.lower())
        if mime_type is None:
            raise ValueError("Upload WAV, MP3, OGG, FLAC, M4A, MP4, or WEBM audio")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ValueError("The uploaded audio file is unavailable") from exc
        if size > self.config.max_audio_upload_bytes:
            raise ValueError(
                f"Audio upload is {size / 1024**2:.1f} MiB; the limit is "
                f"{self.config.max_audio_upload_bytes / 1024**2:.0f} MiB"
            )
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError as exc:
            raise ValueError("Could not read the uploaded audio file") from exc
        return f"data:{mime_type};base64,{encoded}"

    def stream(
        self,
        message: str,
        history: list[dict[str, Any]],
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        repetition_penalty: float,
        top_k: int,
        presence_penalty: float,
        frequency_penalty: float,
        min_p: float,
        audio_file: str | None = None,
    ):
        """Yield incremental text as SSE chunks arrive from vLLM."""
        status = self.manager.status()
        if not status["healthy"]:
            yield "❌ Server is not ready or stopped. Please activate a model first."
            return

        messages: list[dict[str, Any]] = []

        # Add system prompt if provided.
        system_prompt = (system_prompt or "").strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for h in history:
            if isinstance(h, dict):
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
            else:
                messages.append({"role": getattr(h, "role", "user"), "content": getattr(h, "content", "")})
        try:
            audio_url = self._audio_data_url(audio_file)
        except ValueError as exc:
            yield f"❌ {exc}"
            return
        if audio_url:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": message or "Describe this audio."},
                    {"type": "audio_url", "audio_url": {"url": audio_url}},
                ],
            })
        else:
            messages.append({"role": "user", "content": message})

        model_name = self.manager.served_model_name()
        parsed_url = urlparse(self.config.local_api_url)
        host = parsed_url.hostname or "127.0.0.1"
        port = parsed_url.port or self.config.api_port

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": temperature,
            "max_tokens": int(max_tokens),
            "top_p": top_p,
        }
        if repetition_penalty != 1.0:
            payload["repetition_penalty"] = repetition_penalty
        if presence_penalty != 0.0:
            payload["presence_penalty"] = presence_penalty
        if frequency_penalty != 0.0:
            payload["frequency_penalty"] = frequency_penalty
        if top_k != -1:
            payload["top_k"] = int(top_k)
        if min_p != 0.0:
            payload["min_p"] = min_p

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        body = json.dumps(payload).encode("utf-8")
        headers["Content-Length"] = str(len(body))

        t0 = time.monotonic()
        latency = 0.0
        accumulated = ""
        chunk_count = 0
        completion_tokens: int | None = None
        error_occurred = False
        finish_reason = None
        conn = None

        try:
            conn = http.client.HTTPConnection(host, port, timeout=120)
            conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
            response = conn.getresponse()

            if response.status != 200:
                error_msg = response.read().decode(errors="replace")
                error_occurred = True
                yield f"❌ HTTP {response.status}: {error_msg[:500]}"
                return

            for data_str in iter_sse_data(response):
                try:
                    event = json.loads(data_str)
                    usage = event.get("usage")
                    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                        completion_tokens = int(usage["completion_tokens"])
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    if content:
                        accumulated += content
                        chunk_count += 1
                        yield accumulated
                except (TypeError, ValueError, json.JSONDecodeError, IndexError, KeyError):
                    pass
        except Exception as e:
            error_occurred = True
            yield f"❌ Error: {e}"
            return
        finally:
            if conn:
                conn.close()
            latency = time.monotonic() - t0
            self.manager.record_api_call(
                tokens=completion_tokens or 0, latency=latency, error=error_occurred
            )

        if not error_occurred and completion_tokens is not None:
            speed = completion_tokens / latency if latency > 0 else 0.0
            stats = f"\n\n⚡ *{speed:.1f} tok/s | {completion_tokens} tokens | {latency:.2f}s*"
            accumulated += stats
        elif not error_occurred and chunk_count > 0:
            speed = chunk_count / latency if latency > 0 else 0.0
            stats = f"\n\n⚡ *{speed:.1f} chunks/s | {chunk_count} chunks | {latency:.2f}s*"
            accumulated += stats

        # Warn if response was truncated.
        if finish_reason == "length":
            accumulated += "\n\n⚠️ *Response truncated (max_tokens reached)*"

        yield accumulated
