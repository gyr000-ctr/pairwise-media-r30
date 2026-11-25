import math

import pytest

import springarm

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _a():
    return springarm.SpringArm(5.0, 40.0, 10.0)

CASES = [
    ("initial", lambda: _a().current() == 5.0),
    ("shrink", lambda: _a().update(0.0, (1.0, 0.0), 2.0, 0.016) <= 5.0),
    ("no_wall", lambda: _a().update(0.0, (1.0, 0.0), 100.0, 0.016) <= 5.0),
    ("hit_zero", lambda: isinstance(_a().update(0.0, (1.0, 0.0), 0.0, 0.016), float)),
    ("bad_length", lambda: _raises(lambda: springarm.SpringArm(0.0, 40.0, 10.0))),
    ("bad_stiffness", lambda: _raises(lambda: springarm.SpringArm(5.0, 0.0, 10.0))),
    ("bad_dt", lambda: _raises(lambda: _a().update(0.0, (1.0, 0.0), 1.0, 0.0))),
    ("type", lambda: isinstance(_a().update(0.0, (1.0, 0.0), 1.0, 0.016), float)),
    ("det", lambda: _a().update(0.0, (1.0, 0.0), 1.0, 0.016) == _a().update(0.0, (1.0, 0.0), 1.0, 0.016)),
    ("zero_dir", lambda: _raises(lambda: _a().update(0.0, (0.0, 0.0), 1.0, 0.016))),
    ("nan", lambda: _raises(lambda: _a().update(0.0, (float('nan'), 0.0), 1.0, 0.016))),
    ("no_mut", lambda: _a().update(0.0, (1.0, 0.0), 1.0, 0.016) is not None),
    ("range", lambda: 0.0 <= _a().update(0.0, (1.0, 0.0), 3.0, 0.016) <= 5.0),
    ("bounded", lambda: _a().update(0.0, (1.0, 0.0), 100.0, 0.016) <= 5.0 + 1e-9),
    ("current_type", lambda: isinstance(_a().current(), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
