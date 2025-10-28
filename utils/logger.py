import time
from contextlib import contextmanager
from typing import Dict, Any

@contextmanager
def timer():
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"[timer] {t1 - t0:.3f}s")

def pack_result(answer: str, sources, latency: float, extra: Dict[str, Any] | None = None):
    payload = {
        "answer": answer,
        "sources": sorted(set(sources or [])),
        "latency": round(latency, 3)
    }
    if extra:
        payload.update(extra)
    return payload
