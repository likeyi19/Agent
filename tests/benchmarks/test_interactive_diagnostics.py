"""Q1c diagnostic extraction/redaction with SDK-shaped offline exceptions."""
import json
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from benchmarks.interactive.diagnostics import sanitize_provider_error


class SDKError(Exception):
    def __init__(self, *, status_code=None, code=None, body=None, message=None, headers=None):
        super().__init__("Raw SDK exception text is not a diagnostic source.")
        self.status_code = status_code
        self.code = code
        self.body = body
        self.message = message
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


def rendered(record):
    value = json.dumps(record, allow_nan=False)
    assert json.loads(value) == record
    return value


def test_nested_sdk_body_fields_and_numeric_http_status():
    error = SDKError(status_code=400, body={"error": {"code": "invalid_api_key", "type": "authentication_error",
        "message": "The API key is not valid.", "metadata": {"private": "must-never-be-recorded"}}})
    record = sanitize_provider_error(error)
    assert record["http_status"] == 400
    assert record["provider_error"]["code"] == "invalid_api_key"
    assert record["provider_error"]["type"] == "authentication_error"
    assert record["provider_error"]["message"] == "The API key is not valid."
    assert "must-never-be-recorded" not in rendered(record)


def test_google_numeric_code_status_and_structured_reason():
    error = SDKError(code=400, body={"error": {"code": 400, "status": "INVALID_ARGUMENT",
        "message": "API key not valid. Please pass a valid API key.", "details": [{
            "reason": "API_KEY_INVALID", "metadata": {"consumer": "operator-specific-project"}}]}})
    record = sanitize_provider_error(error)
    assert record["http_status"] == 400
    assert record["provider_error"]["code"] == 400
    assert record["provider_error"]["status"] == "INVALID_ARGUMENT"
    assert record["provider_error"]["reason"] == "API_KEY_INVALID"
    assert "operator-specific-project" not in rendered(record)


def test_wrapped_production_error_keeps_underlying_provider_cause():
    cause = SDKError(status_code=429, body={"error": {"code": "quota_exhausted", "message": "Account quota exhausted."}})
    wrapper = RuntimeError("Generic planner error")
    wrapper.__cause__ = cause
    record = sanitize_provider_error(wrapper)
    assert record["http_status"] == 429
    assert record["provider_error"]["code"] == "quota_exhausted"
    assert record["provider_error"]["message"] == "Account quota exhausted."


@pytest.mark.parametrize("field", ["type", "code", "status", "reason", "message"])
def test_known_credential_is_redacted_in_every_provider_scalar(field):
    secret = "actual-credential-sentinel"
    record = sanitize_provider_error(SDKError(body={"error": {field: "prefix " + secret + " suffix"}}), secrets=(secret,))
    assert secret not in rendered(record)
    assert "REDACTED" in record["provider_error"][field]


@pytest.mark.parametrize("private_values", [
    {"secrets": ("123456",)}, {"sensitive_values": ("123456",)},
])
def test_known_private_numeric_provider_code_is_omitted(private_values):
    error = SDKError(status_code=429, code=123456,
        body={"error": {"code": 123456, "message": "Quota exhausted."}})
    record = sanitize_provider_error(error, **private_values)
    assert record["provider_error"]["code"] is None
    assert record["http_status"] == 429
    assert "123456" not in rendered(record)


@pytest.mark.parametrize("code", [400, 1300])
def test_nonprivate_numeric_provider_codes_remain_integers(code):
    error = SDKError(code=code, body={"error": {"code": code, "message": "Provider rejected request."}})
    record = sanitize_provider_error(error, sensitive_values=("123456",))
    assert record["provider_error"]["code"] == code
    assert type(record["provider_error"]["code"]) is int
    assert record["http_status"] == (400 if code == 400 else None)


def test_workspace_url_and_encoded_operator_values_remain_private():
    workspace = "ws-operator/private id"
    secret = "actual+secret/value"
    error = SDKError(message=f"Workspace {workspace} rejected; encoded {quote(workspace, safe='')}; credential {quote(secret, safe='')}.")
    record = sanitize_provider_error(error, secrets=(secret,), sensitive_values=(workspace,))
    serialized = rendered(record)
    for value in (workspace, quote(workspace, safe=""), secret, quote(secret, safe="")):
        assert value not in serialized
    assert "Workspace" in record["provider_error"]["message"]


@pytest.mark.parametrize("message", [
    "Authorization: Bearer never-serialize-this-token",
    "Authorization: Basic never-serialize-this-token",
    'api_key="never-serialize-this-token" failed',
    "x-api-key: never-serialize-this-token failed",
    "request_signature=never-serialize-this-token invalid",
    "Workspace_ID='never-serialize-this-token' invalid",
])
def test_credential_syntax_is_redacted_without_knowing_its_value(message):
    record = sanitize_provider_error(SDKError(message=message))
    assert "never-serialize-this-token" not in rendered(record)


def test_url_userinfo_query_and_fragment_are_removed():
    message = "Unsupported region at https://private-user:private-password@example.com/api/v1?key=query-token#fragment-token"
    record = sanitize_provider_error(SDKError(message=message))
    serialized = rendered(record)
    for forbidden in ("private-user", "private-password", "query-token", "fragment-token"):
        assert forbidden not in serialized
    assert "REDACTED URL" in record["provider_error"]["message"]


def test_entire_configured_url_can_be_treated_as_operator_metadata():
    endpoint = "https://example.com/workspaces/private-workspace/api/v1"
    record = sanitize_provider_error(SDKError(message=f"Workspace at {endpoint} does not support this endpoint."), sensitive_values=(endpoint,))
    assert endpoint not in rendered(record)
    assert "private-workspace" not in rendered(record)


@pytest.mark.parametrize("url", [
    "https://unknown-workspace.cn-beijing.maas.aliyuncs.com/api/v1",
    "https://unknown-subscriber.example.com/projects/unknown-project/generate",
    "https://example.com/accounts/unknown-account/workspaces/unknown-workspace",
])
def test_unprovided_operator_identifiers_inside_urls_are_not_disclosed(url):
    record = sanitize_provider_error(SDKError(message=f"Region does not support Responses at {url}"))
    serialized = rendered(record)
    for identifier in ("unknown-workspace", "unknown-subscriber", "unknown-project", "unknown-account"):
        assert identifier not in serialized
    assert "Region does not support Responses" in record["provider_error"]["message"]


@pytest.mark.parametrize("message,identifier", [
    ("Workspace ID unknown-workspace does not support Responses.", "unknown-workspace"),
    ("Project unknown-project has API disabled.", "unknown-project"),
    ("Quota exhausted for account unknown-account.", "unknown-account"),
    ("Request consumer projects/unknown-project is rejected.", "unknown-project"),
    ("API disabled on resource projects/unknown-project.", "unknown-project"),
])
def test_unprovided_operator_identifiers_in_provider_messages_are_redacted(message, identifier):
    record = sanitize_provider_error(SDKError(message=message))
    assert identifier not in rendered(record)


@pytest.mark.parametrize("message", [
    "API key not valid. Please pass a valid API key.",
    "Account quota exhausted.", "Workspace is missing.",
    "Project API configuration is disabled.", "Region does not support Responses.",
])
def test_operational_reason_remains_useful_after_redaction(message):
    record = sanitize_provider_error(SDKError(message=message))
    assert record["provider_error"]["message"] == message


def test_rate_limit_headers_are_explicitly_allowlisted_and_value_bounded():
    headers = {
        "Retry-After": "5", "X-RateLimit-Limit-Requests": "30",
        "x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-requests": "1m2s",
        "x-ratelimit-reset-tokens": "2026-10-07T01:00:00Z",
        "Authorization": "Bearer never-record-header", "Set-Cookie": "never-record-cookie",
        "x-request-id": "operator-specific-request", "x-ratelimit-unknown": "never-record-unknown",
    }
    record = sanitize_provider_error(SDKError(status_code=429, headers=headers))
    assert record["rate_limit"] == {
        "retry-after": "5", "x-ratelimit-limit-requests": "30",
        "x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-requests": "1m2s",
        "x-ratelimit-reset-tokens": "2026-10-07T01:00:00Z",
    }
    serialized = rendered(record)
    for forbidden in ("never-record-header", "never-record-cookie", "operator-specific-request", "never-record-unknown"):
        assert forbidden not in serialized


def test_retry_after_http_date_is_preserved():
    value = "Wed, 21 Oct 2015 07:28:00 GMT"
    record = sanitize_provider_error(SDKError(headers={"Retry-After": value}))
    assert record["rate_limit"] == {"retry-after": value}


@pytest.mark.parametrize("value", ["credential-in-header", "30; secret=credential-in-header", "nan", "inf", "1" * 100])
def test_arbitrary_rate_header_values_are_not_persisted(value):
    record = sanitize_provider_error(SDKError(headers={"Retry-After": value, "x-ratelimit-limit-tokens": value}))
    assert record["rate_limit"] == {}


def test_known_sensitive_numeric_header_value_is_dropped():
    record = sanitize_provider_error(SDKError(headers={"Retry-After": "123456"}), secrets=("123456",))
    assert record["rate_limit"] == {}
    assert "123456" not in rendered(record)


def test_message_and_scalar_bounds_are_hard():
    record = sanitize_provider_error(SDKError(body={"error": {
        "message": "x" * 1000, "code": "c" * 1000, "type": "t" * 1000, "status": "s" * 1000}}))
    assert len(record["provider_error"]["message"]) == 400
    for key in ("code", "type", "status"):
        assert len(record["provider_error"][key]) == 80


def test_oversized_message_cannot_expose_truncated_secret_prefix():
    secret = "private-credential-" + "z" * 9000
    record = sanitize_provider_error(SDKError(message=secret), secrets=(secret,))
    assert "private-credential-" not in rendered(record)
    assert len(record["provider_error"]["message"]) <= 400


def test_unknown_exception_does_not_use_raw_exception_repr_or_args():
    error = RuntimeError("raw-private-response-body credential-and-workspace")
    record = sanitize_provider_error(error)
    assert record["provider_error"]["message"] == "Provider request failed."
    assert record["provider_error"]["type"] == "RuntimeError"
    assert record["http_status"] is None
    assert "raw-private-response-body" not in rendered(record)


@pytest.mark.parametrize("message", ['{"unreviewed_private_field":"unknown-sensitive-value"}',
    "Error code 400: {'error': {'metadata': 'unknown-sensitive-value'}}"])
def test_sdk_message_does_not_promote_embedded_raw_body_to_diagnostic(message):
    record = sanitize_provider_error(SDKError(message=message))
    assert "unknown-sensitive-value" not in rendered(record)
    assert record["provider_error"]["message"] == "Provider field contained structured details."


def test_sdk_objects_and_unreviewed_fields_are_never_serialized():
    error = SDKError(body={"error": {"message": {"private": "nested-secret"}, "code": object(),
        "status": float("nan"), "request": {"Authorization": "body-secret"}}})
    record = sanitize_provider_error(error)
    assert record["provider_error"]["message"] == "Provider request failed."
    assert record["provider_error"]["code"] is None
    assert record["provider_error"]["status"] is None
    assert "nested-secret" not in rendered(record)
    assert "body-secret" not in rendered(record)


def test_cyclic_cause_context_and_body_are_bounded():
    error = SDKError(status_code=400)
    error.__cause__ = error
    body = {"message": "Provider rejected the region."}
    body["error"] = body
    error.body = body
    record = sanitize_provider_error(error)
    assert record["http_status"] == 400
    assert record["provider_error"]["message"] == "Provider rejected the region."


@pytest.mark.parametrize("code", [True, False, 0, 99, 600, -400, object(), float("nan")])
def test_invalid_numeric_status_is_not_treated_as_http(code):
    record = sanitize_provider_error(SDKError(code=code))
    assert record["http_status"] is None
    rendered(record)
