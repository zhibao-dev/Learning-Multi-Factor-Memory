"""
Run BorgeAgent with any OpenAI-compatible provider.

The cognitive layer (BorgeAgent) is model-agnostic. This example drives it
with the `openai` Python SDK pointed at any of the following providers:

    Provider    | base_url                                       | model example
    ------------|------------------------------------------------|----------------------
    Kimi        | https://api.moonshot.cn/v1                     | moonshot-v1-8k
    MiniMax     | https://api.minimax.chat/v1                    | abab6.5s-chat
    DeepSeek    | https://api.deepseek.com                       | deepseek-chat
    Zhipu (智谱) | https://open.bigmodel.cn/api/paas/v4           | glm-4-flash
    OpenAI      | (default)                                      | gpt-4o-mini
    Ollama      | http://localhost:11434/v1                      | llama3.2 / qwen2.5 / ...
    vLLM        | http://<your-host>:8000/v1                     | (any served model)

Usage:
    pip install openai borge-agent
    export PROVIDER=kimi
    export KIMI_API_KEY=sk-...
    python examples/multi_provider.py "help me debug this auth bug"
"""
from __future__ import annotations

import os
import sys
import uuid

from borge.agent import BorgeAgent

# (base_url, default_model, api_key_env_var)
PROVIDERS: dict[str, tuple[str | None, str, str | None]] = {
    "kimi":     ("https://api.moonshot.cn/v1",                "moonshot-v1-8k",  "KIMI_API_KEY"),
    "minimax":  ("https://api.minimax.chat/v1",               "abab6.5s-chat",   "MINIMAX_API_KEY"),
    "deepseek": ("https://api.deepseek.com",                  "deepseek-chat",   "DEEPSEEK_API_KEY"),
    "zhipu":    ("https://open.bigmodel.cn/api/paas/v4",      "glm-4-flash",     "ZHIPU_API_KEY"),
    "openai":   (None,                                        "gpt-4o-mini",     "OPENAI_API_KEY"),
    "ollama":   ("http://localhost:11434/v1",                 "llama3.2",        None),
    "vllm":     (os.environ.get("VLLM_URL", "http://localhost:8000/v1"), "served-model", None),
}


def main() -> None:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai  # required for this example")

    provider = os.environ.get("PROVIDER", "kimi").lower()
    if provider not in PROVIDERS:
        sys.exit(f"Unknown PROVIDER={provider!r}. Choose: {', '.join(PROVIDERS)}")

    base_url, default_model, key_var = PROVIDERS[provider]
    api_key = os.environ.get(key_var, "EMPTY") if key_var else "EMPTY"
    model = os.environ.get("MODEL", default_model)

    client = OpenAI(api_key=api_key, base_url=base_url)
    borge = BorgeAgent(agent_backend=None)
    borge.on_session_start()

    user_message = sys.argv[1] if len(sys.argv) > 1 else "Tell me what you can do."
    history: list[dict] = []
    session_id = str(uuid.uuid4())[:8]

    print(f"\n[Borge + {provider}/{model}]  session={session_id}\n")

    while user_message and user_message not in {"/exit", "/quit"}:
        # 1) Cognitive layer: pre-turn — extracts emotion, returns context-injection string
        ctx = borge.pre_turn(user_message, history)
        effective = f"{ctx}\n\n{user_message}" if ctx else user_message
        history.append({"role": "user", "content": effective})

        # 2) Call LLM (any OpenAI-compatible endpoint)
        response = client.chat.completions.create(model=model, messages=history)
        reply = response.choices[0].message.content or ""
        history.append({"role": "assistant", "content": reply})

        # 3) Cognitive layer: post-tool — Bayesian belief update + value satisfaction
        borge.post_tool("assistant_turn", reply)

        print(f"[cognitive] {ctx!r}")
        print(f"[borge]     {reply}\n")

        try:
            user_message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

    # 4) Cognitive layer: on-session-end — runs the 7-step consolidation pipeline
    borge.on_session_end(session_id=session_id, messages=history)
    print(f"\n[session {session_id} ended — consolidation complete]")


if __name__ == "__main__":
    main()
