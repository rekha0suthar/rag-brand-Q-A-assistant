"""Result packaging shared by every ask() entry point."""

from typing import Any


def pack_result(answer: str, sources, latency: float, extra: dict[str, Any] | None = None):
    """Shape an answer into the dict contract returned by ask().

    Returns {"answer", "sources" (deduped + sorted), "latency" (rounded, s)}
    plus any `extra` keys merged in (e.g. refusal diagnostics).
    """
    payload = {
        "answer": answer,
        "sources": sorted(set(sources or [])),
        "latency": round(latency, 3),
    }
    if extra:
        payload.update(extra)
    return payload
