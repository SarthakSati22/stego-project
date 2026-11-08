import numpy as np

from src.crossmodal import erasure as E
from src.crossmodal.pipeline3 import PLANS, embed, extract
from src.crossmodal.riskshare import bounded_shares, safe_capacity, shares_from_safe
from src.crypto.secret import derive_key
from tests.test_crossmodal import _toy


def test_weighted_split_roundtrip():
    frame = bytes(range(124))
    for sh in ((0.2, 0.4, 0.4), (0.1, 0.7, 0.2), (0.33, 0.33, 0.34), (0.7, 0.1, 0.2)):
        lens = E.weighted_lengths(len(frame), sh)
        assert sum(lens) == len(frame) and min(lens) >= 1
        frags = E.split_weighted(frame, sh)
        assert [len(f) for f in frags] == lens
        assert E.join_weighted(frags, len(frame)) == frame
        assert E.join_weighted([None] + frags[1:], len(frame)) is None


def test_shares_follow_safe_capacity_only():
    flat = shares_from_safe({"audio": 512, "video": 512, "image": 512})
    assert np.allclose(list(flat.values()), [1 / 3] * 3)          # no built-in preference
    risky = shares_from_safe({"audio": 8, "video": 300, "image": 30})
    assert abs(sum(risky.values()) - 1) < 1e-9
    assert risky["audio"] < risky["image"] < risky["video"]       # safer carrier -> bigger piece
    assert all(0.05 - 1e-9 <= v <= 0.80 + 1e-9 for v in risky.values())
    swap = shares_from_safe({"audio": 300, "video": 8, "image": 30})
    assert swap["video"] < swap["image"] < swap["audio"]          # follows the data, not the carrier
    assert abs(bounded_shares([1, 1, 1]).sum() - 1) < 1e-9


def test_safe_capacity_curve():
    sizes = (16, 32, 64, 128, 256)
    assert safe_capacity(sizes, [0, 0, 0, 0, 0]) == 512
    assert safe_capacity(sizes, [1, 1, 1, 1, 1]) == 8
    c = safe_capacity(sizes, [0, 0, 0.2, 0.9, 1.0])
    assert 64 < c < 128


def test_risk_split_roundtrip_and_needs_all_three():
    a, f, i = _toy()
    key = derive_key("pw")
    shares = {"audio": 0.1, "video": 0.7, "image": 0.2}
    for name in ("risk_split3", "risk_split3_motion"):
        sa, sf, si, meta = embed(a, f, i, b"risk weighted secret", key, PLANS[name], shares=shares)
        assert meta["shares"] == shares and len(set(meta["frag_lens"])) == 3
        got, status = extract(sa, sf, si, key, meta)
        assert got == b"risk weighted secret", status
    wrecked = np.random.default_rng(1).integers(0, 255, sf.shape).astype(np.uint8)
    got, status = extract(sa, wrecked, si, key, meta)
    assert got is None and status[1] == "erased"
