from __future__ import annotations

from collections.abc import Iterable, Iterator


def iter_sse_data(lines: Iterable[bytes]) -> Iterator[str]:
    """Yield complete SSE data payloads and stop immediately at [DONE]."""
    data_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines.clear()
                if payload.strip() == "[DONE]":
                    return
                yield payload
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)

    if data_lines:
        payload = "\n".join(data_lines)
        if payload.strip() != "[DONE]":
            yield payload
