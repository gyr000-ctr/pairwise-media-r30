import math

import pytest

import ddagrid

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _d():
    return ddagrid.Dda()

CASES = [
    ("start", lambda: _d().traverse((0.5, 0.5), (1.0, 0.0), 4, 4)[0] == (0, 0)),
    ("count", lambda: len(_d().traverse((0.5, 0.5), (1.0, 0.0), 4, 4)) <= 5),
    ("diagonal", lambda: len(_d().traverse((0.5, 0.5), (1.0, 1.0), 4, 4)) <= 7),
    ("boundary", lambda: len(set(_d().traverse((1.0, 0.5), (1.0, 0.0), 4, 4))) == len(_d().traverse((1.0, 0.5), (1.0, 0.0), 4, 4))),
    ("zero_dir", lambda: _raises(lambda: _d().traverse((0.0, 0.0), (0.0, 0.0), 4, 4))),
    ("bad_size", lambda: _raises(lambda: _d().traverse((0.0, 0.0), (1.0, 0.0), 0, 4))),
    ("type", lambda: isinstance(_d().traverse((0.5, 0.5), (1.0, 0.0), 4, 4), list)),
    ("det", lambda: _d().traverse((0.5, 0.5), (1.0, 0.0), 4, 4) == _d().traverse((0.5, 0.5), (1.0, 0.0), 4, 4)),
    ("hit_cells", lambda: len(_d().hit_cells((0.5, 0.5), (3.5, 0.5), 4, 4)) >= 3),
    ("hit_type", lambda: isinstance(_d().hit_cells((0.5, 0.5), (1.5, 0.5), 4, 4), list)),
    ("nan", lambda: _raises(lambda: _d().traverse((float('nan'), 0.5), (1.0, 0.0), 4, 4))),
    ("no_mut", lambda: _d().traverse((0.5, 0.5), (1.0, 0.0), 4, 4) is not None),
    ("neg_dir", lambda: isinstance(_d().traverse((3.5, 3.5), (-1.0, -1.0), 4, 4), list)),
    ("pair", lambda: all(len(c) == 2 for c in _d().traverse((0.5, 0.5), (1.0, 0.0), 2, 2))),
    ("int", lambda: all(isinstance(v, int) for c in _d().traverse((0.5, 0.5), (1.0, 0.0), 2, 2) for v in c)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
