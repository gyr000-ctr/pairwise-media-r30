"""springarm 多段 / 扫掠 / 解析恢复 / 性能 / 错误分类的验收用例。"""

import math
import threading
import time

import pytest

import springarm
from springarm import SpringArm

# ---------------------------------------------------------------------------
# 基础 API
# ---------------------------------------------------------------------------
def test_initial_length():
    assert SpringArm(5.0, 40.0, 10.0).current() == 5.0


def test_update_returns_float():
    assert isinstance(SpringArm(5.0, 40.0, 10.0).update(0.0, (1.0, 0.0), 1.0, 0.016), float)


def test_current_returns_float():
    assert isinstance(SpringArm(5.0, 40.0, 10.0).current(), float)


# ---------------------------------------------------------------------------
# 多段串联与长度守恒
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_joint_count_configurable(n):
    assert SpringArm(5.0, 40.0, 10.0, joints=n).joints == n


def test_joint_count_from_length_sequence():
    arm = SpringArm([1.0, 2.0, 2.0], 40.0, 10.0)
    assert arm.joints == 3
    assert arm.rest_lengths == (1.0, 2.0, 2.0)
    assert arm.rest_length == 5.0


def test_joint_count_mismatch_rejected():
    with pytest.raises(springarm.InvalidJointCountError):
        SpringArm([1.0, 2.0], 40.0, 10.0, joints=3)


def test_segment_sum_equals_total_every_update():
    arm = SpringArm([0.5, 1.0, 1.5, 2.0], 60.0, 8.0)
    for hit in (0.0, 0.7, 1.3, 2.9, 4.1, 100.0, 0.2, 3.3):
        total = arm.update((0.0, 0.0), (1.0, 0.0), hit, 0.01)
        segs = arm.segment_lengths()
        assert len(segs) == 4
        assert abs(math.fsum(segs) - total) <= 1e-12
        assert total <= arm.rest_length + 1e-12


def test_segments_scale_proportionally():
    arm = SpringArm([1.0, 2.0, 2.0], 40.0, 10.0)
    arm.update(0.0, (1.0, 0.0), 2.5, 0.016)
    segs = arm.lengths
    # 等比缩放：各段当前长 / 静止长 相同
    for seg, rest in zip(segs, arm.rest_lengths):
        assert abs(seg / rest - segs[0] / arm.rest_lengths[0]) <= 1e-12


def test_joint_positions_chain():
    arm = SpringArm([1.0, 2.0, 2.0], 40.0, 10.0)
    arm.update((3.0, 4.0, 0.0), (1.0, 0.0), 100.0, 0.016)
    pts = arm.joint_positions()
    assert len(pts) == 4
    assert pts[0] == (3.0, 4.0, 0.0)
    assert abs(pts[-1][0] - (3.0 + 5.0)) <= 1e-12


# ---------------------------------------------------------------------------
# 扫掠球体：不得穿墙
# ---------------------------------------------------------------------------
def test_sweep_hit_exact():
    arm = SpringArm(5.0, 40.0, 10.0)
    assert arm.update(0.0, (1.0, 0.0), 2.0, 0.016) == 2.0


def test_sweep_radius_respected():
    arm = SpringArm(5.0, 40.0, 10.0, radius=0.25)
    assert abs(arm.update(0.0, (1.0, 0.0), 2.0, 0.016) - 1.75) <= 1e-12


def test_sweep_no_penetration_under_fast_updates():
    arm = SpringArm(5.0, 40.0, 10.0, radius=0.3)
    rng = __import__("random").Random(1234)
    worst = 0.0
    for _ in range(3000):
        hit = rng.uniform(0.0, 8.0)
        v = arm.update(0.0, (1.0, 0.0), hit, rng.uniform(1e-4, 0.2))
        allowed = max(0.0, hit - 0.3)
        worst = max(worst, v - allowed)
        assert 0.0 <= v <= 5.0 + 1e-12
    assert worst <= 1e-9


def test_target_inside_arm_zero_length():
    arm = SpringArm(5.0, 40.0, 10.0)
    assert arm.update(0.0, (1.0, 0.0), 0.0, 0.016) == 0.0
    # 目标在臂内（距离小于臂长）→ 总长为 0
    assert arm.update(0.0, (1.0, 0.0), 0.0, 0.016) == 0.0


def test_target_behind_arm_zero_length():
    arm = SpringArm(5.0, 40.0, 10.0)
    assert arm.update(0.0, (1.0, 0.0), -4.0, 0.016) == 0.0


# ---------------------------------------------------------------------------
# 恢复：临界阻尼解析解 / 无过冲
# ---------------------------------------------------------------------------
def _critical_params(length, omega):
    return length, omega * omega, 2.0 * omega


@pytest.mark.parametrize("omega", [2.0, 5.0, 10.0, 20.0])
def test_recovery_matches_critical_analytic(omega):
    L = 5.0
    arm = SpringArm(*_critical_params(L, omega))
    arm.update(0.0, (1.0, 0.0), 1.0, 1e-9)  # 即时回缩到 1
    y0 = arm.current()
    assert abs(y0 - 1.0) <= 1e-6

    dt = 1e-3
    t = 0.0
    u0 = y0 - L
    worst = 0.0
    for _ in range(6000):
        t += dt
        y = arm.update(0.0, (1.0, 0.0), 1e9, dt)
        analytic = L + math.exp(-omega * t) * (u0 + (0.0 + omega * u0) * t)
        worst = max(worst, abs(y - analytic))
    assert worst <= 1e-3


@pytest.mark.parametrize("omega", [2.0, 5.0, 10.0, 20.0])
def test_recovery_overshoot_bounded(omega):
    L = 5.0
    arm = SpringArm(*_critical_params(L, omega))
    arm.update(0.0, (1.0, 0.0), 0.0, 1e-9)  # 回缩到 0
    peak = 0.0
    for _ in range(4000):
        y = arm.update(0.0, (1.0, 0.0), 1e9, 1e-3)
        peak = max(peak, y)
    # 过冲 ≤ 5%（临界阻尼理论值为 0）
    assert peak <= L * 1.05
    assert peak <= L + 1e-9


def test_recovery_monotone_when_critical():
    arm = SpringArm(5.0, 100.0, 20.0)
    arm.update(0.0, (1.0, 0.0), 1.0, 1e-9)
    prev = arm.current()
    for _ in range(2000):
        y = arm.update(0.0, (1.0, 0.0), 1e9, 1e-3)
        assert y >= prev - 1e-12
        prev = y


def test_shrink_is_immediate():
    arm = SpringArm(5.0, 1.0, 1.0)  # 很软
    assert arm.update(0.0, (1.0, 0.0), 2.0, 1e-9) == 2.0


def test_overdamped_no_overshoot():
    arm = SpringArm(5.0, 40.0, 100.0)  # 过阻尼
    arm.update(0.0, (1.0, 0.0), 0.0, 1e-9)
    peak = 0.0
    for _ in range(4000):
        peak = max(peak, arm.update(0.0, (1.0, 0.0), 1e9, 1e-3))
    assert peak <= 5.0 + 1e-9


# ---------------------------------------------------------------------------
# 非法输入：可区分的分类报错
# ---------------------------------------------------------------------------
def test_bad_length_class():
    with pytest.raises(springarm.InvalidArmLengthError):
        SpringArm(0.0, 40.0, 10.0)


def test_negative_length_class():
    with pytest.raises(springarm.InvalidArmLengthError):
        SpringArm([1.0, -2.0], 40.0, 10.0)


def test_bad_stiffness_class():
    with pytest.raises(springarm.InvalidStiffnessError):
        SpringArm(5.0, 0.0, 10.0)


def test_bad_damping_class():
    with pytest.raises(springarm.InvalidDampingError):
        SpringArm(5.0, 40.0, 0.0)


def test_bad_joint_count_class():
    for bad in (0, -1, 2.5):
        with pytest.raises(springarm.InvalidJointCountError):
            SpringArm(5.0, 40.0, 10.0, joints=bad)


def test_bad_dt_class():
    with pytest.raises(springarm.InvalidTimeStepError):
        SpringArm(5.0, 40.0, 10.0).update(0.0, (1.0, 0.0), 1.0, 0.0)


def test_bad_direction_class():
    with pytest.raises(springarm.InvalidDirectionError):
        SpringArm(5.0, 40.0, 10.0).update(0.0, (0.0, 0.0), 1.0, 0.016)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_class(value):
    with pytest.raises(springarm.NotFiniteError):
        SpringArm(value, 40.0, 10.0)


@pytest.mark.parametrize("field", ["dt", "hit", "target", "dir"])
def test_non_finite_update_inputs(field):
    arm = SpringArm(5.0, 40.0, 10.0)
    kw = {"target": 0.0, "dirn": (1.0, 0.0), "hit_dist": 1.0, "dt": 0.016}
    key = {"dt": "dt", "hit": "hit_dist", "target": "target", "dir": "dirn"}[field]
    kw[key] = float("nan") if field != "dir" else (float("nan"), 0.0)
    with pytest.raises(springarm.NotFiniteError):
        arm.update(kw["target"], kw["dirn"], kw["hit_dist"], kw["dt"])


def test_error_classes_are_distinct():
    classes = [
        springarm.InvalidArmLengthError,
        springarm.InvalidJointCountError,
        springarm.InvalidStiffnessError,
        springarm.InvalidDampingError,
        springarm.InvalidTimeStepError,
        springarm.InvalidDirectionError,
        springarm.NotFiniteError,
    ]
    assert len(set(classes)) == len(classes)
    assert all(issubclass(c, ValueError) for c in classes)


def test_invalid_input_leaves_state_unchanged():
    arm = SpringArm(5.0, 40.0, 10.0)
    before = arm.current()
    with pytest.raises(ValueError):
        arm.update(0.0, (0.0, 0.0), 1.0, 0.016)
    assert arm.current() == before


# ---------------------------------------------------------------------------
# 性能：1 万次更新 ≤ 400 ms
# ---------------------------------------------------------------------------
def test_perf_10k_updates():
    arm = SpringArm(5.0, 40.0, 10.0)
    t0 = time.perf_counter()
    for i in range(10000):
        arm.update((i * 0.001, 0.0), (1.0, 0.0), 1.0 + (i % 500) * 0.01, 0.016)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms <= 400.0, f"10k updates took {elapsed_ms:.1f} ms"


# ---------------------------------------------------------------------------
# 并发安全
# ---------------------------------------------------------------------------
def test_concurrent_updates_keep_invariants():
    arm = SpringArm([1.0, 1.0, 1.0, 1.0, 1.0], 40.0, 10.0)
    errors = []

    def worker(seed):
        rng = __import__("random").Random(seed)
        try:
            for _ in range(2000):
                arm.update((0.0, 0.0), (1.0, 0.0), rng.uniform(0.0, 8.0), 0.016)
                snap = arm.snapshot()
                assert 0.0 <= snap["length"] <= snap["rest_length"] + 1e-9
                assert abs(math.fsum(snap["segments"]) - snap["length"]) <= 1e-9
                assert len(snap["segments"]) == 5
        except Exception as exc:  # pragma: no cover - 仅失败时进入
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
