"""FastAPI service exposing the RAG chain as POST /ask (plus /healthz).

The chain is initialized once at startup via the lifespan hook; init failures
keep the server up (healthz reports degraded, /ask returns 503) so the
container is debuggable instead of crash-looping.
"""

import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.errors import QuotaExceededError  # light import, no LangChain

ask = None  # the callable returned by load_chain(), set at startup
ask_init_error = None  # full init traceback, kept for /healthz diagnostics


class Query(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ask, ask_init_error
    try:
        # Lazy import here so we see the REAL traceback if chain.py explodes
        from chain import load_chain

        ask = load_chain()
        ask_init_error = None
        print("[api] load_chain OK")
    except Exception as e:
        ask = None
        ask_init_error = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        # Don't raise: /healthz and /docs must still work; /ask returns 503.
        print("[api] load_chain FAILED\n" + ask_init_error)
    yield


app = FastAPI(title="RAG Marketing Assistant", version="1.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h3>RAG Marketing Assistant API</h3>
    <p>Use <code>POST /ask</code> with JSON: {"question": "your question"}</p>
    <p>See interactive docs: <a href="/docs">/docs</a> or <a href="/redoc">/redoc</a>.</p>
    """


@app.get("/healthz")
def healthz():
    if ask_init_error:
        return {"status": "degraded", "error": ask_init_error.splitlines()[-1]}
    return {"status": "ok"}


@app.get("/ask")
def ask_get():
    return JSONResponse(
        status_code=405, content={"detail": 'Use POST /ask with JSON body: {"question": "..."}'}
    )


@app.post("/ask")
def ask_endpoint(q: Query):
    if ask_init_error:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Model pipeline not initialized.",
                "hint": "Check server logs.",
                "error_tail": ask_init_error.splitlines()[-1],
            },
        )
    try:
        return ask(q.question)
    except QuotaExceededError as e:
        # str(e) is the user-facing quota message from utils.errors.
        raise HTTPException(status_code=429, detail=str(e)) from e
    except Exception as e:
        # Log the traceback server-side; never leak exception details to clients.
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal error while answering.") from e
