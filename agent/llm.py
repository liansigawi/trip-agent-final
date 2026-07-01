"""LLMod.ai client via the official OpenAI SDK (LLMod is OpenAI-compatible).

Hardened for production:
- Fail-fast, cached config validation (clear error instead of a bare KeyError deep in a call).
- `chat()` sets an explicit timeout + retries and remembers whether the model supports
  JSON mode, so a single logical call is never billed twice (protects the $9 budget).
- `parse_json()` is TOTAL: it returns a value or None and never raises, using a
  balanced-brace scan instead of a greedy regex."""
import os
import re
import json
from openai import OpenAI

try:  # convenience for local dev; harmless on Vercel
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

MODEL = os.environ.get("LLMOD_MODEL", "gpt-4o-mini")  # reuse your LLMod model value

# Per-call ceiling and retry budget. Kept small so several calls still fit inside the
# Vercel function limit even in a worst-case timeout.
_TIMEOUT_S = 20
_MAX_RETRIES = 1

_client = None
_json_mode_supported = None  # None = unknown, True/False once probed


class ConfigError(RuntimeError):
    """Raised when required LLM configuration is missing/blank. Caught by the API
    layer and turned into a clean error envelope — never a raw stack trace."""


def _require_config():
    """Validate required env vars once, up front. Returns (api_key, base_url)."""
    api_key = (os.environ.get("LLMOD_API_KEY") or "").strip()
    if not api_key:
        raise ConfigError("LLMOD_API_KEY is not set")
    base_url = (os.environ.get("LLMOD_BASE_URL") or "https://api.llmod.ai/v1").strip()
    return api_key, base_url


def _get_client():
    global _client
    if _client is None:
        api_key, base_url = _require_config()
        _client = OpenAI(api_key=api_key, base_url=base_url,
                         timeout=_TIMEOUT_S, max_retries=_MAX_RETRIES)
    return _client


def _looks_like_unsupported_json_mode(err):
    """True if the error is a 'this model/proxy doesn't accept response_format' style
    400 — the only case where retrying WITHOUT json mode is the right move."""
    if getattr(err, "status_code", None) == 400 or err.__class__.__name__ == "BadRequestError":
        return True
    msg = str(err).lower()
    return "response_format" in msg or "json_object" in msg


def chat(messages, temperature=0.3, json_mode=False, max_tokens=1200):
    """One chat completion. Returns the content string ("" if the model returns none).

    When `json_mode` is requested we ask for a strict JSON object. If (and only if) the
    endpoint rejects that parameter, we fall back once and remember it, so we don't pay
    for two calls on every request. Transient errors (auth/rate-limit/timeout) are NOT
    silently retried here — they propagate to the caller's handling."""
    global _json_mode_supported
    client = _get_client()
    base = dict(model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens)

    if json_mode and _json_mode_supported is not False:
        try:
            c = client.chat.completions.create(**base, response_format={"type": "json_object"})
            _json_mode_supported = True
            return c.choices[0].message.content or ""
        except Exception as e:
            if not _looks_like_unsupported_json_mode(e):
                raise  # real failure — don't burn a second billed call
            _json_mode_supported = False  # probe result: remember for next time

    c = client.chat.completions.create(**base)
    return c.choices[0].message.content or ""


def _first_json_object(text):
    """Return the first balanced {...} substring, respecting string literals/escapes.
    Fixes the old greedy `\\{.*\\}` that over-captured across multiple objects."""
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json(text):
    """Best-effort JSON extraction from an LLM reply. TOTAL: returns the parsed value
    (usually a dict) or None. Tolerant of ```json fences and surrounding prose."""
    cleaned = re.sub(r"```json|```", "", str(text)).strip()
    for candidate in (cleaned, _first_json_object(cleaned)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None
