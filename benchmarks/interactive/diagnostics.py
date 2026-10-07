"""Bounded provider-access diagnostics; raw SDK bodies and headers stay private."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from email.utils import parsedate_to_datetime
import json
from itertools import islice
import re
from urllib.parse import quote, quote_plus, unquote


MESSAGE_LIMIT = 400
FIELD_LIMIT = 80
_RATE_HEADERS = frozenset({
    "retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset",
    "x-ratelimit-limit-requests", "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens",
    "ratelimit-limit", "ratelimit-remaining", "ratelimit-reset",
})
_CONTAINERS = ("error", "errors", "details", "body", "response_json")
_ASSIGNMENT = re.compile(
    r"\b(?:authorization|api[_ -]?key|x-api-key|access[_ -]?token|request[_ -]?signature|"
    r"signature|password|credential|secret|workspace[_ -]?id)[\"']?\s*[:=]\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}\])]+)", re.IGNORECASE)
_BEARER = re.compile(r"\b(?:Bearer|Basic)\s+[^\s,;\"'<>]+", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s<>\"'()]+", re.IGNORECASE)
_OPERATOR = re.compile(
    r"\b(?:workspace|project|account|subscriber|subscription|consumer|organization|tenant)"
    r"(?:[_ -]?id)?\b[\"']?\s*(?:[:=]\s*|is\s+)?"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[A-Za-z0-9_./:@%+-]+)", re.IGNORECASE)
_RESOURCE = re.compile(
    r"\b(?:projects|workspaces|accounts|organizations|subscriptions|tenants|consumers)"
    r"/[A-Za-z0-9_.:@%+-]+(?:/[^\s,;\"'<>]+)?", re.IGNORECASE)
_OPERATOR_WORDS = frozenset((
    "not", "missing", "invalid", "quota", "exhausted", "disabled", "unavailable",
    "unsupported", "does", "requires", "has", "restriction", "access", "billing",
    "activation", "enabled", "rate", "limit", "key", "must", "configuration", "api",
))
_KEY = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{12,})\b")
_DURATION = re.compile(r"(?:\d+(?:\.\d+)?(?:ms|s|m|h|d))+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _get(value, key):
    try:
        return value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)
    except Exception:
        return None


def _status(value):
    if type(value) is str and re.fullmatch(r"\d{3}", value):
        value = int(value)
    return value if type(value) is int and 100 <= value <= 599 else None


def _operator_value(match):
    token = match.group("value").strip("\"'.,;").lower()
    return match.group() if token in _OPERATOR_WORDS else "[REDACTED OPERATOR FIELD]"


def _redactor(secrets, sensitive_values):
    known = set()
    for values in (secrets, sensitive_values):
        values = (values,) if isinstance(values, str) else values
        for value in values:
            if type(value) is str and value:
                for text in (value, unquote(value)):
                    known.update((text, quote(text, safe=""), quote_plus(text),
                                  json.dumps(text, ensure_ascii=True)[1:-1]))
    patterns = [re.compile(re.escape(v), re.IGNORECASE) for v in sorted(known, key=len, reverse=True)]

    def redact(value, limit):
        if type(value) is not str:
            return None
        # Reject oversized values instead of exposing a truncated credential prefix.
        if len(value) > 8192:
            return "Provider field exceeded diagnostic bound."
        # Some SDKs embed an entire response representation in their message.
        # Structured fields were already considered; never retain this raw fallback.
        if re.search(r"[\{\[]\s*[\"']", value):
            return "Provider field contained structured details."
        for pattern in patterns:
            value = pattern.sub("[REDACTED]", value)
        value = _BEARER.sub("[REDACTED AUTHORIZATION]", value)
        value = _URL.sub("[REDACTED URL]", value)
        value = _ASSIGNMENT.sub("[REDACTED CREDENTIAL FIELD]", value)
        value = _OPERATOR.sub(_operator_value, value)
        value = _RESOURCE.sub("[REDACTED OPERATOR RESOURCE]", value)
        value = _KEY.sub("[REDACTED KEY]", value)
        value = " ".join("".join(c for c in value if c.isprintable() or c.isspace()).split())
        return value[:limit] or None
    return redact


def _nodes(value, depth=0):
    """Follow only reviewed SDK error containers, never arbitrary metadata fields."""
    if depth > 3:
        return
    if isinstance(value, Mapping):
        yield value
        for key in _CONTAINERS:
            child = _get(value, key)
            if isinstance(child, (list, tuple)):
                for item in child[:4]:
                    yield from _nodes(item, depth + 1)
            elif isinstance(child, Mapping):
                yield from _nodes(child, depth + 1)


def _rate_value(value, name, redact):
    if type(value) in (int, float):
        value = str(value)
    cleaned = redact(value, FIELD_LIMIT)
    if cleaned is None or cleaned != value or len(cleaned) > FIELD_LIMIT:
        return None
    if _NUMBER.fullmatch(cleaned) and len(cleaned) <= 20:
        return cleaned
    if "reset" in name and _DURATION.fullmatch(cleaned):
        return cleaned
    if name == "retry-after":
        try:
            if parsedate_to_datetime(cleaned).tzinfo is not None:
                return cleaned
        except (TypeError, ValueError, OverflowError):
            pass
    if "reset" in name:
        try:
            if datetime.fromisoformat(cleaned.replace("Z", "+00:00")).tzinfo is not None:
                return cleaned
        except ValueError:
            pass
    return None


def sanitize_provider_error(exception, *, secrets=(), sensitive_values=()):
    """Return selected strict-JSON facts after redacting supplied private values.

    Credentials/operator identifiers are supplied only in memory. Unknown
    exceptions get a generic message instead of their unrestricted ``str``.
    """
    redact = _redactor(secrets, sensitive_values)
    chain, pending, seen = [], [exception], set()
    while pending and len(chain) < 8:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current)); chain.append(current)
        pending.extend((_get(current, "__cause__"), _get(current, "__context__")))
    http_status = None
    headers = {}
    sources = []
    for current in chain:
        response = _get(current, "response")
        for obj in (current, response):
            for key in ("status_code", "code", "status"):
                http_status = http_status or _status(_get(obj, key))
        raw_headers = _get(response, "headers") or _get(current, "headers")
        if isinstance(raw_headers, Mapping):
            # Header names are read, but only explicit non-secret numeric/time fields survive.
            for index, name in enumerate(raw_headers):
                if index >= 128:
                    break
                if type(name) is str and name.lower() in _RATE_HEADERS:
                    value = _rate_value(_get(raw_headers, name), name.lower(), redact)
                    if value is not None:
                        headers[name.lower()] = value
        structured = []
        for key in ("body", "details", "error", "response_json"):
            value = _get(current, key)
            if isinstance(value, (list, tuple)):
                for item in value[:4]:
                    structured.extend(islice(_nodes(item), max(0, 32 - len(structured))))
            else:
                structured.extend(islice(_nodes(value), max(0, 32 - len(structured))))
        sources.append((current, structured))
    error = dict(type=None, code=None, status=None, reason=None, message=None)
    # Prefer actual provider fields from the underlying SDK over a generic wrapper.
    for current, structured in reversed(sources):
        for node in [*reversed(structured), current]:
            for key in error:
                if error[key] is not None:
                    continue
                value = _get(node, key)
                if (key == "code" and type(value) is int and abs(value) <= 10**9
                        and redact(str(value), FIELD_LIMIT) == str(value)):
                    error[key] = value
                elif type(value) is str:
                    error[key] = redact(value, MESSAGE_LIMIT if key == "message" else FIELD_LIMIT)
            for key in ("status_code", "code", "status"):
                http_status = http_status or _status(_get(node, key))
    error["type"] = error["type"] or redact(type(exception).__name__, FIELD_LIMIT)
    error["message"] = error["message"] or "Provider request failed."
    result = dict(http_status=http_status, provider_error=error, rate_limit=headers)
    # A strict JSON round trip rejects non-finite or accidental SDK objects.
    return json.loads(json.dumps(result, allow_nan=False, ensure_ascii=True))
