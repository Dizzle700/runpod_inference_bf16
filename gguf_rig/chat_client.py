"""Streaming chat client for vLLM OpenAI-compatible API."""
from __future__ import annotations

import http.client
import json
import time
from typing import Any
from urllib.parse import urlparse

from .config import RigConfig
from .process_manager import VllmServerManager


class ChatClient:
    """Streams chat completions from a local vLLM server."""

    def __init__(self, config: RigConfig, manager: VllmServerManager):
        self.config = config
        self.manager = manager

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
    ):
        """Yield incremental text as SSE chunks arrive from vLLM."""
        status = self.manager.status()
        if not status["healthy"]:
            yield "❌ Server is not ready or stopped. Please activate a model first."
            return

        messages: list[dict[str, str]] = []

        # Add system prompt if provided.
        system_prompt = (system_prompt or "").strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for h in history:
            if isinstance(h, dict):
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
            else:
                messages.append({"role": getattr(h, "role", "user"), "content": getattr(h, "content", "")})
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
        usage_tokens = 0
        total_tokens = 0
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

            buffer = ""
            done = False
            while not done:
                chunk = response.read(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            done = True
                            break
                        try:
                            event = json.loads(data_str)
                            choice = event.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                            # Parse usage from stream (vLLM sends it in the final chunk).
                            usage = event.get("usage")
                            if isinstance(usage, dict):
                                ct = usage.get("completion_tokens")
                                if ct is not None:
                                    usage_tokens = int(ct)
                            if content:
                                accumulated += content
                                chunk_count += 1
                                yield accumulated
                        except (json.JSONDecodeError, IndexError, KeyError):
                            pass
        except Exception as e:
            error_occurred = True
            yield f"❌ Error: {e}"
            return
        finally:
            if conn:
                conn.close()
            latency = time.monotonic() - t0
            total_tokens = usage_tokens or chunk_count
            self.manager.record_api_call(tokens=total_tokens, latency=latency, error=error_occurred)

        if not error_occurred and total_tokens > 0:
            speed = total_tokens / latency if latency > 0 else 0.0
            stats = f"\n\n⚡ *{speed:.1f} tok/s | {total_tokens} tokens | {latency:.2f}s*"
            accumulated += stats

        # Warn if response was truncated.
        if finish_reason == "length":
            accumulated += "\n\n⚠️ *Response truncated (max_tokens reached)*"

        yield accumulated
