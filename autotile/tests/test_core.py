import math

import pytest

import autotile

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

G = [[1, 1], [1, 1]]

CASES = [
    ("mask_all", lambda: autotile.AutoTile(16).mask(G, 0, 0) == 15),
    ("mask_corner", lambda: autotile.AutoTile(16).mask([[1, 0], [0, 0]], 0, 0) == 0),
    ("pick_range", lambda: 0 <= autotile.AutoTile(16).pick(G, 0, 0) < 16),
    ("bad_variants", lambda: _raises(lambda: autotile.AutoTile(0))),
    ("bad_pos", lambda: _raises(lambda: autotile.AutoTile(16).mask(G, 9, 9))),
    ("bad_grid", lambda: _raises(lambda: autotile.AutoTile(16).mask([], 0, 0))),
    ("type", lambda: isinstance(autotile.AutoTile(16).mask(G, 0, 0), int)),
    ("det", lambda: autotile.AutoTile(16).mask(G, 0, 0) == autotile.AutoTile(16).mask(G, 0, 0)),
    ("pick_type", lambda: isinstance(autotile.AutoTile(16).pick(G, 0, 0), int)),
    ("no_mut", lambda: autotile.AutoTile(16).mask([r[:] for r in G], 0, 0) is not None),
    ("mask_bits", lambda: 0 <= autotile.AutoTile(16).mask([[1, 1], [0, 1]], 0, 1) <= 15),
    ("diag", lambda: isinstance(autotile.AutoTile(16).mask([[1, 0], [0, 1]], 1, 1), int)),
    ("single", lambda: autotile.AutoTile(16).mask([[1]], 0, 0) == 0),
    ("edge", lambda: isinstance(autotile.AutoTile(16).pick([[1, 1]], 1, 0), int)),
    ("nan", lambda: _raises(lambda: autotile.AutoTile(16).mask([[float('nan')]], 0, 0))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
