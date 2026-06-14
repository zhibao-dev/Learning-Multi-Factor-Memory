from lmfm.value import learn_weights


def test_learner_improves_a_simple_objective():
    def obj(w):
        return w["a"] - w["b"]
    best, hist = learn_weights(obj, ("a", "b"), seed=1, iters=80)
    assert best["a"] >= best["b"]
    rets = [h["best_return"] for h in hist]
    assert rets == sorted(rets)
