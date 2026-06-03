def test_make_openai_judge_parses_verdict(monkeypatch):
    import borge.audit.judge as J
    captured = {}
    def fake_post(url, payload, headers, timeout):
        captured["url"] = url; captured["payload"] = payload; captured["headers"] = headers
        return {"choices": [{"message": {"content": '{"contradict": true, "stale_id": "a"}'}}]}
    monkeypatch.setattr(J, "_post_json", fake_post)
    judge = J.make_openai_judge(base_url="http://x/v1", model="m", api_key="k")
    v = judge("memory A text", "memory B text")
    assert v == {"contradict": True, "stale_id": "a"}
    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["model"] == "m"
    assert captured["headers"].get("Authorization") == "Bearer k"

def test_make_openai_judge_handles_unparseable(monkeypatch):
    import borge.audit.judge as J
    monkeypatch.setattr(J, "_post_json", lambda *a, **k: {"choices": [{"message": {"content": "not json"}}]})
    judge = J.make_openai_judge(base_url="http://x/v1", model="m")
    assert judge("a", "b") is None      # unparseable → None (abstain → find_contradictions drops the pair)

def test_make_openai_judge_handles_http_error(monkeypatch):
    import borge.audit.judge as J
    def boom(*a, **k): raise OSError("connection refused")
    monkeypatch.setattr(J, "_post_json", boom)
    judge = J.make_openai_judge(base_url="http://x/v1", model="m")
    assert judge("a", "b") is None      # HTTP failure → None (never crash the audit)

def test_no_api_key_omits_auth_header(monkeypatch):
    import borge.audit.judge as J
    captured = {}
    monkeypatch.setattr(J, "_post_json", lambda url, payload, headers, timeout: (captured.update(headers=headers) or {"choices":[{"message":{"content":'{"contradict": false, "stale_id": null}'}}]}))
    judge = J.make_openai_judge(base_url="http://localhost:11434/v1", model="llama3")   # local Ollama, no key
    judge("a", "b")
    assert "Authorization" not in captured["headers"]
