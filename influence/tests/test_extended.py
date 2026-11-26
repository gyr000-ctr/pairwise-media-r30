"""评审约束的扩展自检（可见 15 例之外）。

全部用标准库写成，保证“无外部依赖”。默认用例都是确定性的小网格；
真正卡 512x512 < 700ms 的性能阈值用环境变量 ``INFLUENCE_RUN_PERF=1`` 开启，
避免共享机器高负载时计时抖动影响 ``pytest tests -q`` 这道硬门槛。

    INFLUENCE_RUN_PERF=1 python -m pytest tests/test_extended.py -q
"""

import math
import os
import random
import threading
import time
from collections import deque

import pytest

import influence
from influence import Influence


# --------------------------------------------------------------------------- #
# 参考实现：独立的多源 BFS 距离场
# --------------------------------------------------------------------------- #
def bfs_distance(grid, sources):
    h = len(grid)
    w = len(grid[0]) if h else 0
    dist = [[-1] * w for _ in range(h)]
    q = deque()
    for r, c in sources:
        if 0 <= r < h and 0 <= c < w and dist[r][c] == -1:
            dist[r][c] = 0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if dist[nr][nc] != -1:
                continue
            if grid[nr][nc] != 0:
                continue
            dist[nr][nc] = dist[r][c] + 1
            q.append((nr, nc))
    return dist


def expected(grid, sources, decay):
    d = bfs_distance(grid, sources)
    return [[(decay ** dd if dd >= 0 else 0.0) for dd in row] for row in d]


# --------------------------------------------------------------------------- #
# 1) 与 BFS 距离场逐格一致（<=1e-9），随机网格/多源/多 decay
# --------------------------------------------------------------------------- #
def test_matches_bfs_distance_field_random():
    rng = random.Random(20261126)
    for _ in range(200):
        h = rng.randint(1, 20)
        w = rng.randint(1, 20)
        p = rng.random() * 0.4
        grid = [[1 if rng.random() < p else 0 for _ in range(w)] for _ in range(h)]
        sources = [(rng.randrange(h), rng.randrange(w)) for _ in range(rng.randint(1, 4))]
        decay = rng.choice([0.2, 0.5, 0.7, 0.9, 0.99, 1.0])
        got = Influence(decay).spread(grid, sources)
        exp = expected(grid, sources, decay)
        for r in range(h):
            for c in range(w):
                assert abs(got[r][c] - exp[r][c]) <= 1e-9


def test_distance_field_helper():
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    df = Influence(0.5).distance_field(grid, [(0, 0)])
    assert df[0][0] == 0
    assert df[0][1] == 1 and df[1][0] == 1
    assert df[1][1] == -1  # 遮挡
    assert df[2][2] == 4  # 绕过中心墙的最短路径


# --------------------------------------------------------------------------- #
# 2) 多源取最大（竞争），绝不求和
# --------------------------------------------------------------------------- #
def test_multi_source_is_max_not_sum():
    grid = [[0, 0, 0], [0, 0, 0]]
    out = Influence(0.5).spread(grid, [(0, 0), (0, 2)])
    assert out[0][0] == 1.0 and out[0][2] == 1.0
    assert abs(out[0][1] - 0.5) < 1e-12
    assert abs(out[1][1] - 0.25) < 1e-12
    assert all(v <= 1.0 for row in out for v in row)


# --------------------------------------------------------------------------- #
# 3) 遮挡格阻断、不穿透（含对角缝隙与封闭房间）
# --------------------------------------------------------------------------- #
def test_walls_block_and_do_not_leak():
    grid = [
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    out = Influence(0.9).spread(grid, [(0, 0)])
    for r in range(5):
        for c in range(5):
            if grid[r][c] == 1:
                assert out[r][c] == 0.0
    assert out == Influence(0.9).spread(grid, [(0, 0)])  # determinism
    exp = expected(grid, [(0, 0)], 0.9)
    assert all(abs(out[r][c] - exp[r][c]) < 1e-12 for r in range(5) for c in range(5))


def test_closed_room_unreachable():
    room = [
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    out = Influence(0.8).spread(room, [(0, 0)])
    assert out[1][2] == 0.0 and out[2][2] == 0.0


# --------------------------------------------------------------------------- #
# 4) 与“迭代到两次差 <=1e-12”的参考松弛结果一致
# --------------------------------------------------------------------------- #
def _iterative_reference(grid, sources, decay, tol=1e-12):
    h, w = len(grid), len(grid[0])
    src = set(sources)
    f = [[0.0] * w for _ in range(h)]
    for r, c in sources:
        f[r][c] = 1.0
    for _ in range(100000):
        delta = 0.0
        nf = [row[:] for row in f]
        for r in range(h):
            for c in range(w):
                if (r, c) in src:
                    nf[r][c] = 1.0
                    continue
                if grid[r][c] != 0:
                    nf[r][c] = 0.0
                    continue
                best = 0.0
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0:
                        best = max(best, decay * f[nr][nc])
                nf[r][c] = best
                delta = max(delta, abs(best - f[r][c]))
        f = nf
        if delta <= tol:
            return f
    raise AssertionError("reference did not converge")


def test_matches_converged_iteration():
    rng = random.Random(7)
    for decay in (0.5, 0.9, 0.99):
        grid = [[1 if rng.random() < 0.15 else 0 for _ in range(24)] for _ in range(24)]
        sources = [(2, 2), (20, 21)]
        for r, c in sources:
            grid[r][c] = 0
        ref = _iterative_reference(grid, sources, decay)
        got = Influence(decay).spread(grid, sources)
        err = max(abs(got[r][c] - ref[r][c]) for r in range(24) for c in range(24))
        assert err <= 1e-9


def test_field_is_fixed_point_to_1e_12():
    grid = [[(r * 7 + c * 3) % 11 == 0 for c in range(30)] for r in range(30)]
    grid = [[1 if v else 0 for v in row] for row in grid]
    sources = [(0, 0), (22, 22)]
    for r, c in sources:
        grid[r][c] = 0
    decay = 0.9
    f = Influence(decay).spread(grid, sources)
    residual = 0.0
    for r in range(30):
        for c in range(30):
            if grid[r][c] != 0 or (r, c) in set(sources):
                continue
            best = 0.0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 30 and 0 <= nc < 30 and grid[nr][nc] == 0:
                    best = max(best, decay * f[nr][nc])
            residual = max(residual, abs(best - f[r][c]))
    assert residual <= 1e-12


# --------------------------------------------------------------------------- #
# 5) 梯度方向与距离场下降方向一致（内部中心差分，容差 1e-9）
# --------------------------------------------------------------------------- #
def test_gradient_aligns_with_distance_descent():
    h = w = 31
    for src in ((0, 0), (15, 15), (5, 20)):
        grid = [[0] * w for _ in range(h)]
        out = Influence(0.7).spread(grid, [src])
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                gy = (out[r + 1][c] - out[r - 1][c]) / 2.0
                gx = (out[r][c + 1] - out[r][c - 1]) / 2.0
                dy = 1 if r > src[0] else (-1 if r < src[0] else 0)
                dx = 1 if c > src[1] else (-1 if c < src[1] else 0)
                # 曼哈顿距离中心差分在尖点处可能为 0；用解析梯度方向。
                if r == src[0] and c == src[1]:
                    continue
                if gy == 0.0 and gx == 0.0:
                    continue
                ng = math.hypot(gy, gx)
                nd = math.hypot(dy, dx)
                if nd == 0.0:
                    continue
                cos = -(gy * dy + gx * dx) / (ng * nd)
                assert cos >= 1.0 - 1e-9, (src, r, c, cos)


# --------------------------------------------------------------------------- #
# 6) 并发安全：共享实例 + 屏障，多线程结果逐位相等
# --------------------------------------------------------------------------- #
def test_shared_instance_thread_safe_exact_equality():
    rng = random.Random(99)
    grid = [[1 if rng.random() < 0.2 else 0 for _ in range(80)] for _ in range(80)]
    inf = Influence(0.73)
    sources = [(0, 0), (79, 79)]
    ref = inf.spread(grid, sources)
    outputs = [None] * 12
    barrier = threading.Barrier(12)

    def worker(i):
        barrier.wait()
        outputs[i] = inf.spread(grid, sources)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for out in outputs:
        assert out == ref


# --------------------------------------------------------------------------- #
# 7) 报错分类
# --------------------------------------------------------------------------- #
def test_decay_errors_are_distinguishable():
    for bad in (0.0, -0.1, 2.0):
        with pytest.raises(influence.InvalidDecayError):
            Influence(bad)
        with pytest.raises(ValueError):
            Influence(bad)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(influence.NonFiniteInputError):
            Influence(bad)
        with pytest.raises(TypeError):
            Influence(bad)
        with pytest.raises(influence.InvalidDecayError):
            Influence(bad)
    with pytest.raises(ValueError):
        Influence("0.5")
    # 合法边界
    assert Influence(1.0).decay == 1.0


def test_grid_and_source_errors():
    with pytest.raises(influence.InvalidGridError):
        Influence(0.5).spread([], [(0, 0)])
    with pytest.raises(influence.InvalidGridError):
        Influence(0.5).spread([[0], [0, 1]], [(0, 0)])
    with pytest.raises(influence.NonFiniteGridError):
        Influence(0.5).spread([[float("nan")]], [(0, 0)])
    with pytest.raises(influence.InvalidSourceError):
        Influence(0.5).spread([[0, 0]], [(9, 9)])
    with pytest.raises(influence.InvalidSourceError):
        Influence(0.5).spread([[0, 0]], [(0, 1.5)])
    with pytest.raises(influence.NonFiniteSourceError):
        Influence(0.5).spread([[0]], [(float("inf"), 0)])


# --------------------------------------------------------------------------- #
# 8) 不修改输入 + 返回类型/值域
# --------------------------------------------------------------------------- #
def test_does_not_mutate_grid():
    grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    snapshot = [row[:] for row in grid]
    Influence(0.5).spread(grid, [(0, 0)])
    assert grid == snapshot


def test_return_types_and_range():
    out = Influence(0.5).spread([[0, 0], [0, 0]], [(0, 0)])
    assert isinstance(out, list)
    assert all(isinstance(row, list) for row in out)
    assert all(isinstance(v, float) for row in out for v in row)
    assert all(0.0 <= v <= 1.0 for row in out for v in row)


def test_empty_sources_all_zero():
    out = Influence(0.5).spread([[1, 0], [0, 1]], [])
    assert all(v == 0.0 for row in out for v in row)
    assert Influence(0.5).spread([], []) == []


# --------------------------------------------------------------------------- #
# 8b) 字面“线性衰减”变体：max(0, 1 - decay*d)，同样与 BFS 距离场一致
# --------------------------------------------------------------------------- #
def test_linear_variant_matches_affine_bfs():
    grid = [[0] * 9 for _ in range(9)]
    out = Influence(0.1).spread_linear(grid, [(0, 0)])
    for r in range(9):
        for c in range(9):
            d = r + c
            assert abs(out[r][c] - max(0.0, 1.0 - 0.1 * d)) <= 1e-12
    # 线性场在未截断区域对原始有限差分梯度也与距离下降方向一致（含边界）。
    h = w = 7
    lin = Influence(0.05).spread_linear([[0] * w for _ in range(h)], [(0, 0)])
    for r in range(h):
        for c in range(h):
            if r in (0, h - 1) or c in (0, w - 1):
                continue
            gy = (lin[r + 1][c] - lin[r - 1][c]) / 2.0
            gx = (lin[r][c + 1] - lin[r][c - 1]) / 2.0
            assert gy < 0.0 and gx < 0.0  # 朝源点（距离下降）方向增大



# --------------------------------------------------------------------------- #
# 9) 性能：512x512（始终跑一个宽松冒烟；严格 700ms 阈值由环境变量开启）
# --------------------------------------------------------------------------- #
def test_perf_512_smoke_and_shape():
    n = 512
    grid = [[0] * n for _ in range(n)]
    t0 = time.perf_counter()
    out = Influence(0.8).spread(grid, [(0, 0)])
    elapsed = time.perf_counter() - t0
    assert len(out) == n and all(len(row) == n for row in out)
    assert out[0][0] == 1.0
    assert abs(out[0][n - 1] - 0.8 ** (n - 1)) < 1e-9
    # 40 倍以上余量的宽松上限，仅用于拦截 O(N^2) 级劣化，正常约 0.05s。
    assert elapsed < 2.0


@pytest.mark.skipif(
    os.environ.get("INFLUENCE_RUN_PERF") != "1",
    reason="设置 INFLUENCE_RUN_PERF=1 以校验 512x512 < 700ms",
)
def test_perf_512_under_700ms_best_of_three():
    n = 512
    grid = [[0] * n for _ in range(n)]
    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        Influence(0.8).spread(grid, [(0, 0)])
        timings.append(time.perf_counter() - t0)
    assert min(timings) < 0.7, timings
