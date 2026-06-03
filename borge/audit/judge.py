"""OpenAI-compatible LLM judge client for the contradiction-precision filter.

The judge calls a caller-provided LLM that speaks OpenAI's ``/chat/completions``
API. This is intentionally the *customer's* endpoint — either their own cloud
LLM or a local Ollama (``http://localhost:11434/v1``) — never a you-hosted API.

Dependency-light by design: stdlib ``urllib`` only, no ``openai`` SDK, no new
package. ``_post_json`` is isolated so tests can monkeypatch it without real
network access.

The returned ``judge(text_a, text_b)`` matches the contract that
``find_contradictions`` expects: ``{"contradict": bool, "stale_id": "a"|"b"|None}``
or ``None`` (abstain — never crashes the audit).
"""

import json
import urllib.request
from typing import Callable, Optional

_SYSTEM_PROMPT = (
    "You compare two agent-memory entries, A and B. Decide if they assert "
    "CONTRADICTORY FACTS about the SAME subject — NOT merely the same topic. "
    "Same topic but compatible facts = NOT a contradiction. Reply with ONLY a "
    'JSON object: {"contradict": true|false, "stale_id": "a"|"b"|null} where '
    "stale_id is the entry more likely outdated (or null)."
)


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 30) -> dict:
    """Thin ``urllib`` JSON POST. Isolated so tests monkeypatch it."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def make_openai_judge(
    base_url: str,
    model: str,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> Callable[[str, str], Optional[dict]]:
    """Build a ``judge(text_a, text_b)`` backed by an OpenAI-compatible endpoint.

    ``api_key`` is optional — a local Ollama needs none, so the Authorization
    header is only sent when a key is given.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"

    def judge(text_a: str, text_b: str) -> Optional[dict]:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"A: {text_a}\nB: {text_b}"},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = _post_json(url, payload, headers, timeout)
            content = resp["choices"][0]["message"]["content"]
            d = json.loads(content)
            return {"contradict": bool(d["contradict"]), "stale_id": d.get("stale_id")}
        except Exception:
            # Any failure (HTTP, KeyError, JSON parse) → abstain. The audit must
            # never crash on a judge failure.
            return None

    return judge
