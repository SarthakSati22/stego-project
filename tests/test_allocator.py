import pytest

from src.allocator.allocator import STRATEGIES, Allocator, Carrier


def make_alloc():
    return Allocator(Carrier(3000, 0.5, 1.0), Carrier(4000, 0.5, 0.5))


def fake_embed(n):
    def embed(share):
        n_a = int(round(share * n))
        return n_a / n, (n - n_a) / n, n_a, n - n_a
    return embed


def fake_detect(sa, sv):
    p_a, p_v = 0.05 + 0.9 * sa, 0.05 + 0.3 * sv       # audio is far riskier
    return p_a, p_v, max(p_a, p_v)


def test_five_strategies_exist():
    assert len(STRATEGIES) == 5


def test_fixed_strategies():
    for name, share in (("audio_only", 1.0), ("video_only", 0.0), ("fixed_50_50", 0.5)):
        out = make_alloc().run(name, 1000, fake_embed(1000))
        assert out["share"] == pytest.approx(share)
        assert out["n_a"] + out["n_v"] == 1000


def test_more_robust_carrier_gets_larger_share():
    out = make_alloc().run("adaptive_no_feedback", 1000, fake_embed(1000))
    assert 0.5 < out["share"] < 1.0


def test_capacity_is_respected():
    alloc = Allocator(Carrier(300, 0.5, 1.0), Carrier(4000, 0.5, 1.0))
    out = alloc.run("audio_only", 1000, fake_embed(1000))
    assert out["n_a"] == 300 and out["n_v"] == 700


def test_feedback_moves_payload_away_from_the_risky_carrier():
    alloc, emb = make_alloc(), fake_embed(1000)
    base = alloc.run("adaptive_no_feedback", 1000, emb)
    base_risk = fake_detect(*emb(base["share"])[:2])[2]
    out = alloc.run("adaptive_with_feedback", 1000, emb, fake_detect)
    assert out["share"] < base["share"]
    assert out["risk"] < base_risk


def test_unknown_strategy_and_missing_detector():
    with pytest.raises(ValueError):
        make_alloc().run("magic", 100, fake_embed(100))
    with pytest.raises(ValueError):
        make_alloc().run("adaptive_with_feedback", 100, fake_embed(100))