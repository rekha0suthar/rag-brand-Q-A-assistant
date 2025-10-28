# api/main.py
from pathlib import Path
import sys, traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(title="RAG Marketing Assistant", version="1.0")

ask = None                  # will hold the callable returned by load_chain()
ask_init_error = None       # capture init error text for debugging

class Query(BaseModel):
    question: str

@app.on_event("startup")
def startup():
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
        # Don’t raise, so /healthz and /docs still work; /ask will return 503
        print("[api] load_chain FAILED\n" + ask_init_error)

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
        status_code=405,
        content={"detail": 'Use POST /ask with JSON body: {"question": "..."}'}
    )

@app.post("/ask")
def ask_endpoint(q: Query):
    if ask_init_error:
        return JSONResponse(status_code=503, content={
            "detail": "Model pipeline not initialized.",
            "hint": "Check server logs.",
            "error_tail": ask_init_error.splitlines()[-1]
        })
    try:
        out = ask(q.question)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
