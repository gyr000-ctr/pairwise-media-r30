import math

import pytest

import animblend

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

C = [(0.0, [1.0, 0.0]), (1.0, [0.0, 1.0])]

CASES = [
    ("endpoints", lambda: abs(animblend.BlendSpace(C).sample(0.0)[0] - 1.0) < 1e-9),
    ("mid", lambda: abs(animblend.BlendSpace(C).sample(0.5)[0] - 0.5) < 1e-9),
    ("sum_w", lambda: abs(sum(animblend.BlendSpace(C).weights(0.5)) - 1.0) < 1e-12),
    ("clamp_low", lambda: abs(animblend.BlendSpace(C).sample(-5.0)[0] - 1.0) < 1e-9),
    ("clamp_high", lambda: abs(animblend.BlendSpace(C).sample(99.0)[1] - 1.0) < 1e-9),
    ("bad_clips", lambda: _raises(lambda: animblend.BlendSpace([]))),
    ("unsorted", lambda: _raises(lambda: animblend.BlendSpace([(1.0, [0.0]), (0.0, [1.0])]))),
    ("type", lambda: isinstance(animblend.BlendSpace(C).sample(0.3), list)),
    ("det", lambda: animblend.BlendSpace(C).sample(0.3) == animblend.BlendSpace(C).sample(0.3)),
    ("w_type", lambda: isinstance(animblend.BlendSpace(C).weights(0.3), list)),
    ("loop", lambda: isinstance(animblend.BlendSpace(C, loop=True).sample(1.5), list)),
    ("nan", lambda: _raises(lambda: animblend.BlendSpace(C).sample(float('nan')))),
    ("no_mut", lambda: animblend.BlendSpace(list(C)).sample(0.4) is not None),
    ("three_clips", lambda: isinstance(animblend.BlendSpace([(0.0, [1.0]), (1.0, [0.0]), (2.0, [0.5])]).sample(0.5), list)),
    ("weights_len", lambda: len(animblend.BlendSpace(C).weights(0.5)) == 2),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
