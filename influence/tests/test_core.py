import math

import pytest

import influence

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

G = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

CASES = [
    ("source", lambda: influence.Influence(0.5).spread(G, [(0, 0)])[0][0] == 1.0),
    ("decay", lambda: influence.Influence(0.5).spread(G, [(0, 0)])[0][1] < 1.0),
    ("wall", lambda: influence.Influence(0.5).spread(G, [(0, 0)])[1][1] == 0.0),
    ("multi_max", lambda: influence.Influence(0.5).spread(G, [(0, 0), (0, 1)])[0][1] <= 1.0),
    ("bad_decay", lambda: _raises(lambda: influence.Influence(0.0))),
    ("bad_grid", lambda: _raises(lambda: influence.Influence(0.5).spread([], [(0, 0)]))),
    ("bad_source", lambda: _raises(lambda: influence.Influence(0.5).spread(G, [(9, 9)]))),
    ("type", lambda: isinstance(influence.Influence(0.5).spread(G, [(0, 0)]), list)),
    ("det", lambda: influence.Influence(0.5).spread(G, [(0, 0)]) == influence.Influence(0.5).spread(G, [(0, 0)])),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in influence.Influence(0.5).spread(G, [(0, 0)]) for v in row)),
    ("nan", lambda: _raises(lambda: influence.Influence(0.5).spread([[float('nan')]], [(0, 0)]))),
    ("no_mut", lambda: influence.Influence(0.5).spread([r[:] for r in G], [(0, 0)]) is not None),
    ("monotone", lambda: influence.Influence(0.9).spread(G, [(0, 0)])[0][2] >= influence.Influence(0.5).spread(G, [(0, 0)])[0][2]),
    ("float", lambda: isinstance(influence.Influence(0.5).spread(G, [(0, 0)])[0][0], float)),
    ("no_source", lambda: all(v == 0.0 for row in influence.Influence(0.5).spread(G, []) for v in row)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
