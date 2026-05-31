def test_savings_char4_deterministic():
    from borge.audit.savings import estimate_savings
    texts = ["a" * 40, "b" * 80]   # 120 chars → 30 tokens char/4
    out = estimate_savings(texts, retrieval_freq=10.0, price_per_1k=0.003, tokenizer="char4")
    assert out["tokens_per_retrieval"] == 30          # 120/4
    assert out["tokens_saved_per_month"] == 300       # 30 * 10
    assert abs(out["usd_per_month"] - 300/1000*0.003) < 1e-9
    assert "retrieval_freq" in out["assumptions"] and "price_per_1k" in out["assumptions"]
    assert out["assumptions"]["tokenizer"] == "char4"


def test_savings_tiktoken_positive():
    from borge.audit.savings import estimate_savings
    out = estimate_savings(["The project deadline is March 15th."],
                           retrieval_freq=1.0, price_per_1k=0.003, tokenizer="tiktoken")
    assert out["tokens_per_retrieval"] > 0
    assert out["assumptions"]["tokenizer"] == "tiktoken"
