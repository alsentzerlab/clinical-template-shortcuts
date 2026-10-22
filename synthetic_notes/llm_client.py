import itertools
import json
import os
import threading
import time
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# Azure-OpenAI-compatible gateway. Point LLM_API_BASE_URL at the deployments root,
# e.g. https://<host>/azure-openai/deployments
_AUTH_HEADER = "api-key"
_BASE_URL = os.environ.get("LLM_API_BASE_URL", "").rstrip("/")
_API_VERSION = os.environ.get("LLM_API_VERSION", "2025-04-01-preview")
MODELS = {
    m: f"{_BASE_URL}/{m}/chat/completions?api-version={_API_VERSION}"
    for m in ("gpt-5", "gpt-5-5", "gpt-5-mini", "gpt-5-nano")
}

# Thread-safe global token counter (kept for backward compat; for per-job
# tracking under concurrency, use the per-instance LLMClient.tokens attribute).
_token_lock = threading.Lock()
_token_totals = {"prompt": 0, "completion": 0, "calls": 0}

def get_token_totals() -> dict:
    with _token_lock:
        return dict(_token_totals)

def reset_token_totals() -> None:
    with _token_lock:
        _token_totals["prompt"] = 0
        _token_totals["completion"] = 0
        _token_totals["calls"] = 0

RETRYABLE = {"connection", "timeout", "rate limit", "overloaded", "service unavailable", "429", "503", "500"}
# Auth failures retried *with key rotation* (each attempt picks the next round-robin key).
AUTH_RETRYABLE = {"401", "access denied", "invalid subscription", "unauthorized"}

# Default path to the .env holding the API keys (override with --env_file / load_keys()).
DEFAULT_ENV_FILE = ".env"
# Env var names for the round-robin API keys, in priority order.
KEY_VARS = ["LLM_API_KEY_1", "LLM_API_KEY_2"]


def _parse_env_file(path: str) -> dict:
    out = {}
    try:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#") or "=" not in ln:
                    continue
                k, _, v = ln.partition("=")
                out[k.replace("export", "").strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return out


def load_keys(env_file: Optional[str] = None) -> str:
    """Resolve API keys (comma-separated) for round-robin.

    Priority: LLM_API_KEY_* already in the environment, else parsed from
    env_file (defaults to DEFAULT_ENV_FILE), else legacy GPT_API_KEY.
    """
    keys = [os.environ[v] for v in KEY_VARS if os.environ.get(v)]
    if not keys:
        env = _parse_env_file(env_file or DEFAULT_ENV_FILE)
        keys = [env[v] for v in KEY_VARS if env.get(v)]
    if not keys and os.environ.get("GPT_API_KEY"):
        keys = [os.environ["GPT_API_KEY"]]
    return ",".join(keys)


def validate_keys(keys: list[str], model: str = "gpt-5") -> list[str]:
    """Ping each key once; return only those that authenticate (drop 401s)."""
    url = MODELS[model]
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": "Reply OK."}]})
    good = []
    for k in keys:
        try:
            r = requests.post(url, headers={_AUTH_HEADER: k, "Content-Type": "application/json"},
                              data=payload, timeout=60)
            if r.ok:
                good.append(k)
            else:
                logger.warning(f"Dropping key ending …{k[-4:]}: HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"Dropping key ending …{k[-4:]}: {e}")
    return good


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        reasoning_effort: Optional[str] = None,
        check_keys: bool = False,
    ):
        if model not in MODELS:
            raise ValueError(f"Unsupported model '{model}'. Choose from: {list(MODELS)}")
        if not _BASE_URL:
            raise ValueError("Set LLM_API_BASE_URL to the gateway's deployments root.")

        # Support multiple comma-separated keys for round-robin. Falls back to
        # LLM_API_KEY_* / .env / GPT_API_KEY via load_keys() when api_key is None.
        raw = api_key or load_keys()
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise ValueError(
                "No API keys found. Set LLM_API_KEY_1/LLM_API_KEY_2, pass api_key, "
                "or ensure the .env file exists."
            )
        if check_keys:
            keys = validate_keys(keys, model) or keys  # keep originals if all fail, to surface the real error

        self._key_cycle = itertools.cycle(keys)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.url = MODELS[model]
        # Per-instance token tracking — safe under concurrency since each thread
        # can use its own LLMClient and read its own counts.
        self._tokens_lock = threading.Lock()
        self.tokens = {"prompt": 0, "completion": 0, "calls": 0}
        logger.info(f"LLMClient initialized with {len(keys)} API key(s)")

    def get_tokens(self) -> dict:
        with self._tokens_lock:
            return dict(self.tokens)

    def reset_tokens(self) -> None:
        with self._tokens_lock:
            self.tokens = {"prompt": 0, "completion": 0, "calls": 0}

    def _headers(self) -> dict:
        return {
            _AUTH_HEADER: next(self._key_cycle),
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: list[dict], max_tokens: int, temperature: float,
                       reasoning_effort: Optional[str] = None) -> dict:
        # Reasoning models: no temperature/max_tokens (deployment controls these).
        # reasoning_effort (minimal|low|medium|high) caps reasoning-token output cost.
        payload = {"model": self.model, "messages": messages}
        effort = reasoning_effort or self.reasoning_effort
        if effort:
            payload["reasoning_effort"] = effort
        return payload

    def _extract_text(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"]

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 8000,
        temperature: float = 0.0,
        max_retries: int = 5,
        request_delay: float = 2.0,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        payload = self._build_payload(messages, max_tokens, temperature, reasoning_effort)
        if request_delay > 0:
            time.sleep(request_delay)
        for attempt in range(max_retries):
            try:
                resp = requests.post(self.url, headers=self._headers(), data=json.dumps(payload), timeout=300)
                if not resp.ok:
                    logger.error(f"API error {resp.status_code}: {resp.text[:500]}")
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {}) or {}
                p = usage.get("prompt_tokens", 0) or 0
                c = usage.get("completion_tokens", 0) or 0
                with _token_lock:
                    _token_totals["prompt"] += p
                    _token_totals["completion"] += c
                    _token_totals["calls"] += 1
                with self._tokens_lock:
                    self.tokens["prompt"] += p
                    self.tokens["completion"] += c
                    self.tokens["calls"] += 1
                return self._extract_text(data)
            except Exception as e:
                msg = str(e).lower()
                is_rate_limit = "429" in msg or "too many requests" in msg
                # Transient 401s can happen on the gateway even with valid keys.
                # Each retry's _headers() rotates to the next round-robin key, so a
                # transient auth failure on one key recovers on the other. Permanently
                # dead keys are dropped up front by validate_keys(--check_keys), so this
                # won't loop forever on a bad key.
                is_auth = any(k in msg for k in AUTH_RETRYABLE)
                is_retryable = is_rate_limit or is_auth or any(k in msg for k in RETRYABLE)
                if attempt < max_retries - 1 and is_retryable:
                    if is_rate_limit:
                        # try to parse retry-after from response, otherwise use longer backoff
                        try:
                            retry_after = int(resp.headers.get("Retry-After", 0))
                        except Exception:
                            retry_after = 0
                        wait = retry_after if retry_after > 0 else 15 * (attempt + 1)
                    elif is_auth:
                        wait = 2  # not congestion — rotate key and retry fast
                    else:
                        wait = 10 * (attempt + 1)
                    logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}. "
                                   f"{'Rotating key, ' if is_auth else ''}retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    def generate_json(
        self,
        messages: list[dict],
        max_tokens: int = 8000,
        temperature: float = 0.0,
        max_retries: int = 3,
        reasoning_effort: Optional[str] = None,
    ) -> dict:
        text = self.generate(messages, max_tokens, temperature, max_retries,
                             reasoning_effort=reasoning_effort)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > 0:
                return json.loads(text[start:end])
            raise ValueError(f"Could not parse JSON from response:\n{text}")
