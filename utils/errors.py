"""Provider-error classification shared by the chain, Streamlit UI and API.

Lives in utils (not chain.py) so the FastAPI layer can import the exception
type without pulling in chain.py's heavy LangChain/embeddings imports.
"""

QUOTA_MESSAGE = (
    "The free AI quota is temporarily exhausted — please try again in a few minutes."
)

# Substrings that identify a rate/quota failure in provider error text.
# Groq's SDK formats HTTP errors as "Error code: 429 - {...'Rate limit reached'...}".
_QUOTA_PATTERNS = ("rate limit", "rate_limit", "quota", "too many requests", "error code: 429")


class QuotaExceededError(RuntimeError):
    """The LLM provider rejected the call for rate-limit/quota reasons.

    str(exc) is a user-facing message, safe to render directly in the UI.
    """

    def __init__(self, message: str = QUOTA_MESSAGE):
        super().__init__(message)


def is_quota_error(exc: BaseException) -> bool:
    """Classify an exception as a provider rate/quota failure.

    Checks the exception and its __cause__/__context__ chain (LangChain may
    wrap provider errors) for an HTTP 429 status attribute or rate/quota
    wording in the message. Pure and side-effect free.
    """
    seen = set()
    e = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        for attr in ("status_code", "http_status", "status"):
            if getattr(e, attr, None) == 429:
                return True
        if getattr(getattr(e, "response", None), "status_code", None) == 429:
            return True
        text = str(e).lower()
        if any(pattern in text for pattern in _QUOTA_PATTERNS):
            return True
        e = e.__cause__ or e.__context__
    return False
