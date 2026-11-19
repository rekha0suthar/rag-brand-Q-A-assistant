import pytest

from utils.errors import QUOTA_MESSAGE, QuotaExceededError, is_quota_error


class FakeAPIError(Exception):
    """Provider-style error carrying an HTTP status attribute."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


# ---- positive cases: everything Groq/langchain might throw at us ----


def test_status_code_attribute_429():
    assert is_quota_error(FakeAPIError("boom", status_code=429))


def test_response_status_code_429():
    exc = Exception("opaque")
    exc.response = FakeResponse(429)
    assert is_quota_error(exc)


def test_groq_sdk_message_format():
    # Groq SDK formats HTTP errors like this even without status attributes.
    msg = "Error code: 429 - {'error': {'message': 'Rate limit reached for model ...'}}"
    assert is_quota_error(Exception(msg))


@pytest.mark.parametrize(
    "message",
    [
        "Rate limit reached for requests per day",
        "rate_limit_exceeded",
        "You have exceeded your quota",
        "Too Many Requests",
    ],
)
def test_message_patterns(message):
    assert is_quota_error(Exception(message))


def test_wrapped_cause_is_detected():
    # LangChain may wrap the provider error; the chain must be walked.
    try:
        try:
            raise FakeAPIError("provider says no", status_code=429)
        except FakeAPIError as inner:
            raise RuntimeError("generation failed") from inner
    except RuntimeError as outer:
        assert is_quota_error(outer)


# ---- negative cases: ordinary failures must NOT be mapped to quota ----


@pytest.mark.parametrize(
    "exc",
    [
        Exception("connection refused"),
        FakeAPIError("invalid api key", status_code=401),
        FakeAPIError("server error", status_code=500),
        ValueError("question must be a string"),
        Exception("timeout after 42900 ms waiting for socket"),
    ],
)
def test_non_quota_errors(exc):
    assert not is_quota_error(exc)


def test_quota_exception_message_is_user_facing():
    assert str(QuotaExceededError()) == QUOTA_MESSAGE
    assert "try again" in QUOTA_MESSAGE
