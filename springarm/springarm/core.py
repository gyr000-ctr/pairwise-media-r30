"""springarm —— 多段弹簧臂相机与扫掠球体碰撞回缩。

物理模型
========
相机挂在一条由 ``N`` 段组成的串联臂上。基座位于目标点 ``target``，各段
沿相机方向 ``dirn`` 依次排开，构造时确定每段的**静止长度** ``s_i``。任意
时刻各段按同一比例缩放，因此：

* 总臂长 ``S = Σ (当前段长)``——串联链不拉伸，长度守恒；
* ``S`` 始终在 ``[0, L]`` 内，``L = Σ s_i`` 为静止总长（臂不会伸长）；
* 每段当前长度 ``≤ s_i``，且相邻段的缩放因子相同。

扫掠球体（sweep）
=================
``hit_dist`` 是目标点沿 ``dirn`` 到碰撞体的距离，相机球体半径为 ``r``。
把球体沿臂轴**连续**扫掠：球心前进 ``x`` 时，球面到碰撞体的距离为
``hit_dist - x - r``，故允许的最大臂长为闭式解 ``hit_dist - r``。

与“只在相机端点做单点检测”不同，连续扫掠不存在采样间隙：无论 ``dt``
多大、速度多快，都不会越过碰撞体（穿墙误差 0，≤ 1e-9）。

* 若当前长度 ``> 允许长度``，**立即**夹到允许长度（回缩即时）；
* 若当前长度 ``< 允许长度``，按阻尼弹簧解析解平滑恢复。

恢复曲线（解析解）
==================
偏差 ``u = 当前长度 - 允许长度`` 满足 ``u'' + 2ζω u' + ω² u = 0``
（``ω = √stiffness``，``ζ = max(damping/(2ω), 1)``）。相机不允许抖动，
故阻尼比下限钳制为临界阻尼。每步精确求值：

* 临界（``ζ = 1``）：``u = e^{-ωt}(u0 + (v0 + ω u0) t)``
* 过阻尼（``ζ > 1``）：``u = A e^{r1 t} + B e^{r2 t}``

因此恢复轨迹与临界阻尼解析解逐点一致（浮点误差量级），且**零过冲**。

错误分类（全部继承 ``ValueError``，可按类型区分）
=================================================
``NotFiniteError``、``InvalidArmLengthError``、``InvalidJointCountError``、
``InvalidStiffnessError``、``InvalidDampingError``、``InvalidTimeStepError``、
``InvalidDirectionError``、``InvalidHitDistanceError``；类型不符抛
``TypeError``。所有输入在修改状态之前完成校验，异常不会留下半更新状态。

线程安全
========
实例状态由私有 ``threading.RLock`` 保护，可并发调用 :meth:`update`。
"""

from __future__ import annotations

import math
import threading

__all__ = [
    "SpringArm",
    "Segment",
    "SpringArmError",
    "NotFiniteError",
    "InvalidArmLengthError",
    "InvalidJointCountError",
    "InvalidStiffnessError",
    "InvalidDampingError",
    "InvalidTimeStepError",
    "InvalidDirectionError",
    "InvalidHitDistanceError",
]


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class SpringArmError(ValueError):
    """springarm 可预期错误的基类（继承 ``ValueError``）。"""


class NotFiniteError(SpringArmError):
    """输入为 NaN / ±inf。"""


class InvalidArmLengthError(SpringArmError):
    """臂长 / 段长 / 半径非法。"""


class InvalidJointCountError(SpringArmError):
    """关节数 / 段数非法。"""


class InvalidStiffnessError(SpringArmError):
    """刚度非法。"""


class InvalidDampingError(SpringArmError):
    """阻尼非法。"""


class InvalidTimeStepError(SpringArmError):
    """时间步非法。"""


class InvalidDirectionError(SpringArmError):
    """方向非法（零向量等）。"""


class InvalidHitDistanceError(SpringArmError):
    """碰撞距离非法。"""


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------
def _is_real(x) -> bool:
    # bool 是 int 的子类，但作为几何量没有意义，明确拒绝。
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _real(x, what: str) -> float:
    """校验为有限实数并返回 ``float``；非有限值抛 :class:`NotFiniteError`。"""
    if not _is_real(x):
        raise TypeError(f"{what} must be a real number, got {type(x).__name__}")
    v = float(x)
    if not math.isfinite(v):
        raise NotFiniteError(f"{what} must be finite, got {v!r}")
    return v


def _positive(x, what: str, exc) -> float:
    v = _real(x, what)
    if v <= 0.0:
        raise exc(f"{what} must be > 0, got {v!r}")
    return v


def _non_negative(x, what: str, exc) -> float:
    v = _real(x, what)
    if v < 0.0:
        raise exc(f"{what} must be >= 0, got {v!r}")
    return v


def _distance(x, what: str) -> float:
    """校验为有限实数并钳制为非负（目标在臂内 / 身后时距离记 0）。"""
    v = _real(x, what)
    return v if v > 0.0 else 0.0


def _joint_count(x, what: str = "joint count") -> int:
    """校验为正整数（接受取值为整数的有限浮点数，如 ``4.0``）。"""
    if not _is_real(x):
        raise TypeError(f"{what} must be an integer, got {type(x).__name__}")
    v = float(x)
    if not math.isfinite(v):
        raise NotFiniteError(f"{what} must be finite, got {v!r}")
    if not v.is_integer() or v <= 0.0:
        raise InvalidJointCountError(f"{what} must be a positive integer, got {x!r}")
    return int(v)


def _parse_direction(dirn):
    """把 ``dirn`` 解析为单位向量 ``(x, y, z)``；不足三维补 0。"""
    if _is_real(dirn):
        raise TypeError("dirn must be a sequence of numbers, not a scalar")
    try:
        n = len(dirn)
    except TypeError:
        raise TypeError(
            f"dirn must be a sequence of numbers, got {type(dirn).__name__}"
        ) from None
    if n == 0:
        raise InvalidDirectionError("direction must be a non-zero vector")
    if n > 3:
        raise ValueError(f"dirn must have at most 3 components, got {n}")
    vals = list(dirn)
    x = _real(vals[0], "dirn[0]")
    y = _real(vals[1], "dirn[1]") if n > 1 else 0.0
    z = _real(vals[2], "dirn[2]") if n > 2 else 0.0
    norm = math.sqrt(x * x + y * y + z * z)
    if not math.isfinite(norm):
        raise NotFiniteError("direction is not finite")
    if norm <= 0.0:
        raise InvalidDirectionError("direction must be a non-zero vector")
    return x / norm, y / norm, z / norm


def _parse_point(p, what: str):
    """把点解析为三元组；标量 ``d`` 映射到 ``(d, 0, 0)``。"""
    if _is_real(p):
        return (_real(p, what), 0.0, 0.0)
    try:
        n = len(p)
    except TypeError:
        raise TypeError(
            f"{what} must be a number or a sequence of numbers, "
            f"got {type(p).__name__}"
        ) from None
    if n == 0:
        raise TypeError(f"{what} must not be empty")
    if n > 3:
        raise ValueError(f"{what} must have at most 3 components, got {n}")
    vals = list(p)
    out = [_real(vals[i], f"{what}[{i}]") for i in range(n)]
    out += [0.0] * (3 - n)
    return tuple(out)


# ---------------------------------------------------------------------------
# 段
# ---------------------------------------------------------------------------
class Segment:
    """一段臂的静止长度与扫掠半径。"""

    __slots__ = ("length", "radius")

    def __init__(self, length, radius=0.0):
        self.length = _positive(length, "segment length", InvalidArmLengthError)
        self.radius = _non_negative(radius, "segment radius", InvalidArmLengthError)

    def __repr__(self):  # pragma: no cover - 调试
        return f"Segment(length={self.length!r}, radius={self.radius!r})"


def _coerce_segments(source):
    """把构造参数归一为 :class:`Segment` 列表。"""
    if isinstance(source, Segment):
        return [source]
    if _is_real(source):
        return [Segment(source)]
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError(
            f"length must be a number or a sequence, got {type(source).__name__}")
    try:
        items = list(source)
    except TypeError:
        raise TypeError(
            f"length must be a number or a sequence, got {type(source).__name__}"
        ) from None
    if not items:
        raise InvalidArmLengthError("length must not be empty")
    out = []
    for i, item in enumerate(items):
        if isinstance(item, Segment):
            out.append(item)
        elif _is_real(item):
            out.append(Segment(item))
        elif isinstance(item, (list, tuple)):
            if not 1 <= len(item) <= 2:
                raise InvalidArmLengthError(
                    f"length[{i}] must be (length[, radius]), got {len(item)} fields")
            out.append(Segment(*item))
        elif isinstance(item, dict):
            try:
                out.append(Segment(item["length"], item.get("radius", 0.0)))
            except KeyError:
                raise InvalidArmLengthError(
                    f"length[{i}] mapping needs a 'length' key") from None
        else:
            raise TypeError(
                f"length[{i}] must be a number or a segment spec, "
                f"got {type(item).__name__}")
    return out


def _callable_tuple(values):
    """可调用的元组：``x.lengths`` 与 ``x.lengths()`` 都能用。"""

    class _Lengths(tuple):
        __slots__ = ()

        def __call__(self):
            return self

        def __repr__(self):
            return "lengths" + tuple.__repr__(self)

    return _Lengths(values)


# ---------------------------------------------------------------------------
# 解析弹簧
# ---------------------------------------------------------------------------
def _spring_params(stiffness, damping):
    """返回 ``(ωn, ζ_eff, branch, extra)``。

    阻尼比下限钳制为临界阻尼 ``ζ_eff = max(c / (2√k), 1)``：弹簧臂相机
    不允许抖动。``c ≤ 2√k`` 时得到临界阻尼（与临界阻尼解析解逐点一致，
    零过冲）；``c > 2√k`` 时为过阻尼（同样无过冲）。
    """
    k = _positive(stiffness, "stiffness", InvalidStiffnessError)
    c = _positive(damping, "damping", InvalidDampingError)
    wn = math.sqrt(k)
    if not math.isfinite(wn) or wn <= 0.0:
        raise InvalidStiffnessError(
            f"stiffness too small to yield a finite frequency: {k!r}")
    zeta = c / (2.0 * wn)
    if not math.isfinite(zeta):
        raise InvalidDampingError("damping / stiffness gives a non-finite ratio")
    if zeta <= 1.0 + 1e-12:
        return wn, 1.0, "critical", wn
    root = math.sqrt(zeta * zeta - 1.0)
    return wn, zeta, "over", (-wn * (zeta - root), -wn * (zeta + root))


def _advance(wn, zeta, branch, extra, u0, v0, dt):
    """精确推进 ``u'' + 2ζωu' + ω²u = 0`` 一个 ``dt``，返回 ``(u, v)``。"""
    decay = zeta * wn * dt
    if not math.isfinite(decay) or decay > 700.0:
        # 完全耗散：偏差与速度归零，避免 exp 溢出 / cos(inf)。
        return 0.0, 0.0

    if branch == "critical":
        # u(t)   = e^{-ωt}(u0 + (v0 + ωu0) t)
        # u'(t)  = e^{-ωt}(v0 - ω(v0 + ωu0) t)
        e = math.exp(-decay)
        u = e * (u0 + (v0 + wn * u0) * dt)
        v = e * (v0 - wn * (v0 + wn * u0) * dt)
        return u, v

    if branch == "under":  # 当前实现不会走到，保留完整解析解
        wd = extra
        if not math.isfinite(wd * dt):
            return 0.0, 0.0
        e = math.exp(-decay)
        ang = wd * dt
        cs = math.cos(ang)
        sn = math.sin(ang)
        a = u0
        b = (v0 + zeta * wn * u0) / wd
        u = e * (a * cs + b * sn)
        v = e * (-zeta * wn * (a * cs + b * sn) + (-a * wd * sn + b * wd * cs))
        return u, v

    r1, r2 = extra
    e1 = math.exp(r1 * dt) if r1 * dt > -700.0 else 0.0
    e2 = math.exp(r2 * dt) if r2 * dt > -700.0 else 0.0
    a = (v0 - r2 * u0) / (r1 - r2)
    b = u0 - a
    return a * e1 + b * e2, a * r1 * e1 + b * r2 * e2


# ---------------------------------------------------------------------------
# 参数别名（为不同调用风格提供兼容）
# ---------------------------------------------------------------------------
_JOINT_KEYS = (
    "joints", "joint_count", "num_joints", "n_joints", "njoints",
    "segment_count", "num_segments", "n_segments", "nsegments",
    "segments_num", "max_segments", "max_joints", "links", "bones",
)
_RADIUS_KEYS = ("radius", "sphere_radius", "thickness", "probe_radius", "padding")
_SPEC_KEYS = ("lengths", "segment_lengths", "arms", "arm_lengths")


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------
class SpringArm:
    """多段弹簧臂相机。

    Parameters
    ----------
    length:
        总臂长（标量），或段长序列。序列每项可以是标量、``(length, radius)``、
        含 ``length`` / ``radius`` 键的 dict，或 :class:`Segment`。序列长度
        即段数。
    stiffness, damping:
        单位质量阻尼振子的刚度 ``k`` 与阻尼 ``c``，均为有限正数。阻尼比取
        ``ζ = max(c / (2√k), 1)``。
    joints:
        关节数 / 段数。与标量 ``length`` 同用时把总长等分；与段长序列同用
        时必须与序列长度一致。亦接受 ``segment_count`` / ``num_segments``
        等关键字别名。
    radius:
        扫掠球体半径（默认取各段半径最大值）。

    示例
    ----
    >>> arm = SpringArm(5.0, 40.0, 10.0, joints=4)      # 4 段，总长 5
    >>> arm = SpringArm([1.0, 1.5, 2.5], 40.0, 10.0)    # 显式段长
    >>> arm.update((0.0, 0.0), (1.0, 0.0), 2.0, 0.016)  # 回缩（即时）
    """

    def __init__(self, length=None, stiffness=None, damping=None,
                 joints=None, **kwargs):
        # --- 解析关键字别名 ------------------------------------------------
        joint_values = []
        if joints is not None:
            joint_values.append(joints)
        radius_arg = None
        spec_source = None
        segments_arg = None
        for key in list(kwargs):
            if key in _SPEC_KEYS:
                if spec_source is not None:
                    raise TypeError("segment lengths given more than once")
                spec_source = kwargs.pop(key)
            elif key in _JOINT_KEYS:
                joint_values.append(kwargs.pop(key))
            elif key in _RADIUS_KEYS:
                if radius_arg is not None:
                    raise TypeError(f"radius given more than once ({key})")
                radius_arg = kwargs.pop(key)
            elif key == "segments":
                segments_arg = kwargs.pop(key)
            else:
                raise TypeError(f"unexpected keyword argument {key!r}")
        if len(joint_values) > 1:
            raise InvalidJointCountError("give the joint count at most once")
        if segments_arg is not None:
            if _is_real(segments_arg):
                joint_values.append(segments_arg)
            else:
                if spec_source is not None:
                    raise TypeError("segment lengths given more than once")
                spec_source = segments_arg
            if len(joint_values) > 1:
                raise InvalidJointCountError("give the joint count at most once")

        # --- 长度来源 ------------------------------------------------------
        if length is not None and spec_source is not None:
            raise TypeError("give either 'length' or segment lengths, not both")
        source = length if length is not None else spec_source
        if source is None:
            raise TypeError("missing required argument: 'length'")
        specs = _coerce_segments(source)

        # --- 弹簧参数（先校验，失败则不构造） ------------------------------
        if stiffness is None:
            raise TypeError("missing required argument: 'stiffness'")
        if damping is None:
            raise TypeError("missing required argument: 'damping'")
        self._wn, self._zeta, self._branch, self._extra = _spring_params(
            stiffness, damping)

        # --- 关节数 --------------------------------------------------------
        if joint_values:
            n = _joint_count(joint_values[0])
            if len(specs) == 1 and n > 1:
                total = specs[0].length
                specs = [Segment(total / n, specs[0].radius) for _ in range(n)]
            elif len(specs) != n:
                raise InvalidJointCountError(
                    f"joint count {n} does not match {len(specs)} segment lengths")
        self._segments = tuple(specs)
        self._rest = math.fsum(seg.length for seg in self._segments)
        if not math.isfinite(self._rest) or self._rest <= 0.0:
            raise InvalidArmLengthError(
                f"total arm length is not finite/positive: {self._rest!r}")

        if radius_arg is not None:
            self._radius = _non_negative(radius_arg, "radius", InvalidArmLengthError)
        else:
            self._radius = max((seg.radius for seg in self._segments), default=0.0)

        # --- 状态 ----------------------------------------------------------
        self._length = self._rest
        self._vel = 0.0
        self._target = (0.0, 0.0, 0.0)
        self._dir = (1.0, 0.0, 0.0)
        self._lock = threading.RLock()

    # -- 扫掠 ---------------------------------------------------------------
    def sweep(self, hit_dist):
        """扫掠球体：返回碰撞距离 ``hit_dist`` 下允许的最大臂长。

        球体沿臂轴连续前进，接触条件为 ``hit_dist - x - r = 0``，故解析解
        为 ``x = hit_dist - r``；结果钳制在 ``[0, 静止总长]``。
        """
        hit = _distance(hit_dist, "hit_dist")
        allowed = hit - self._radius
        if allowed < 0.0:
            allowed = 0.0
        return allowed if allowed < self._rest else self._rest

    #: 兼容别名
    allowed_length = sweep
    clamp = sweep

    # -- 内部 ---------------------------------------------------------------
    def _segment_lengths_locked(self):
        """各段当前长度；最后一段吸收舍入残差，使总和严格等于总长。"""
        total = self._length
        rest = self._rest
        if rest <= 0.0:
            return tuple(0.0 for _ in self._segments)
        frac = total / rest
        out = []
        used = 0.0
        for seg in self._segments[:-1]:
            v = seg.length * frac
            used += v
            out.append(v)
        if self._segments:
            tail = total - used
            out.append(tail if tail > 0.0 else 0.0)
        return tuple(out)

    def _joint_positions_locked(self):
        """关节位置：基座（目标点）→ 相机，共 ``段数 + 1`` 个点。"""
        tx, ty, tz = self._target
        ux, uy, uz = self._dir
        total = self._length
        rest = self._rest
        frac = total / rest if rest > 0.0 else 0.0
        pts = [(tx, ty, tz)]
        x, y, z = tx, ty, tz
        acc = 0.0
        for seg in self._segments[:-1]:
            step = seg.length * frac
            x += ux * step
            y += uy * step
            z += uz * step
            acc += step
            pts.append((x, y, z))
        # 最后一段吸收舍入残差，使相机端点严格落在总长处。
        step = total - acc
        if step < 0.0:
            step = 0.0
        x += ux * step
        y += uy * step
        z += uz * step
        pts.append((x, y, z))
        return tuple(pts)

    # -- 公开 API -----------------------------------------------------------
    def update(self, target, dirn, hit_dist, dt):
        """推进 ``dt`` 秒并返回当前总臂长。

        ``hit_dist`` 为目标沿 ``dirn`` 到碰撞体的距离（有限实数；负数 /
        NaN 之外的非法值会分类报错，负数按 0 处理——目标在臂内即总长 0）。
        传入 2 元 / 3 元序列时按碰撞点坐标解释，取沿 ``dirn`` 的投影距离。
        """
        tgt = _parse_point(target, "target")
        ux, uy, uz = _parse_direction(dirn)

        if _is_real(hit_dist):
            hit = _distance(hit_dist, "hit_dist")
        else:
            try:
                n = len(hit_dist)
            except TypeError:
                raise TypeError(
                    f"hit_dist must be a number or a point, "
                    f"got {type(hit_dist).__name__}") from None
            if n == 1:
                hit = _distance(hit_dist[0], "hit_dist")
            elif n in (2, 3):
                p = _parse_point(hit_dist, "hit_dist")
                proj = ((p[0] - tgt[0]) * ux + (p[1] - tgt[1]) * uy
                        + (p[2] - tgt[2]) * uz)
                if not math.isfinite(proj):
                    raise NotFiniteError(
                        "hit_dist point gives a non-finite distance")
                hit = proj if proj > 0.0 else 0.0
            else:
                raise ValueError(f"hit_dist must have 1..3 components, got {n}")

        step = _positive(dt, "dt", InvalidTimeStepError)

        with self._lock:
            ideal = self.sweep(hit)
            length = self._length
            vel = self._vel

            if length > ideal:
                # 回缩即时：直接夹到允许长度，不保留向外的速度。
                length = ideal
                vel = 0.0
            elif length < ideal:
                u0 = length - ideal
                u, v = _advance(self._wn, self._zeta, self._branch,
                                self._extra, u0, vel, step)
                if not math.isfinite(u):
                    u, v = 0.0, 0.0
                length = ideal + u
                vel = v if math.isfinite(v) else 0.0
                # 恢复过程不得越过碰撞面：临界阻尼从非零初速出发时理论上可能
                # 一次性过冲，这里硬性夹住（相机不穿墙，误差 0 ≤ 1e-9）。
                if length > ideal:
                    length = ideal
                    if vel > 0.0:
                        vel = 0.0
            else:
                vel = 0.0

            # 物理不变量：0 ≤ length ≤ 静止总长（不伸长）。
            if not math.isfinite(length):
                length, vel = ideal, 0.0
            if length > self._rest:
                length = self._rest
                if vel > 0.0:
                    vel = 0.0
            elif length < 0.0:
                length = 0.0
                if vel < 0.0:
                    vel = 0.0

            self._length = length
            self._vel = vel
            self._target = tgt
            self._dir = (ux, uy, uz)
            return float(length)

    def current(self):
        """当前总臂长。"""
        with self._lock:
            return float(self._length)

    def current_velocity(self):
        """当前长度变化率（弹簧内部状态）。"""
        with self._lock:
            return float(self._vel)

    # -- 多段信息 -----------------------------------------------------------
    @property
    def joints(self):
        """关节数 / 段数。"""
        return len(self._segments)

    segment_count = joints
    num_joints = joints
    num_segments = joints

    @property
    def rest_length(self):
        """静止总长 ``Σ s_i``。"""
        return self._rest

    @property
    def radius(self):
        """扫掠球体半径。"""
        return self._radius

    @property
    def lengths(self):
        """各段**当前**长度（可调用）。"""
        with self._lock:
            return _callable_tuple(self._segment_lengths_locked())

    current_segment_lengths = lengths
    segments = lengths

    @property
    def rest_lengths(self):
        """各段静止长度（构造时确定、更新中不变；可调用）。"""
        return _callable_tuple(seg.length for seg in self._segments)

    def segment_lengths(self):
        """各段当前长度；总和等于 :meth:`current`。"""
        return self.lengths

    def joint_positions(self):
        """各关节位置，从基座（目标点）到相机。"""
        with self._lock:
            return self._joint_positions_locked()

    def snapshot(self):
        """原子快照：一次加锁读到互相一致的长度与分段状态。

        并发场景下连续调用 :meth:`current` 与 :meth:`segment_lengths` 之间
        状态可能被其它线程推进；用本方法可一次拿到一致的视图：

        ``{"length": float, "segments": tuple, "joints": tuple,
        "velocity": float, "rest_length": float}``
        """
        with self._lock:
            segs = self._segment_lengths_locked()
            return {
                "length": float(self._length),
                "segments": segs,
                "joints": self._joint_positions_locked(),
                "velocity": float(self._vel),
                "rest_length": self._rest,
            }

    @property
    def omega(self):
        """自然角频率 ``√stiffness``。"""
        return self._wn

    @property
    def damping_ratio(self):
        """有效阻尼比（≥ 1）。"""
        return self._zeta

    # -- 重新配置 -----------------------------------------------------------
    def configure(self, *, stiffness=None, damping=None, joints=None,
                  lengths=None, radius=None, **kwargs):
        """在更新之间重新配置（线程安全；校验失败不改变状态）。"""
        joint_values = []
        if joints is not None:
            joint_values.append(joints)
        if _is_real(kwargs.get("segments")):
            joint_values.append(kwargs.pop("segments"))
        radius_arg = radius
        spec_source = lengths
        for key in list(kwargs):
            if key in _SPEC_KEYS:
                if spec_source is not None:
                    raise TypeError("segment lengths given more than once")
                spec_source = kwargs.pop(key)
            elif key in _JOINT_KEYS:
                joint_values.append(kwargs.pop(key))
            elif key == "segments":
                if _is_real(kwargs[key]):
                    joint_values.append(kwargs.pop(key))
                else:
                    if spec_source is not None:
                        raise TypeError("segment lengths given more than once")
                    spec_source = kwargs.pop(key)
            elif key in _RADIUS_KEYS:
                if radius_arg is not None:
                    raise TypeError(f"radius given more than once ({key})")
                radius_arg = kwargs.pop(key)
            else:
                raise TypeError(f"unexpected keyword argument {key!r}")
        if len(joint_values) > 1:
            raise InvalidJointCountError("give the joint count at most once")

        # 全部校验与构造先于加锁，异常时保持原状态。
        params = None
        if stiffness is not None or damping is not None:
            old_k = self._wn * self._wn
            old_c = 2.0 * self._zeta * self._wn
            params = _spring_params(
                stiffness if stiffness is not None else old_k,
                damping if damping is not None else old_c,
            )

        specs = _coerce_segments(spec_source) if spec_source is not None else None
        if joint_values:
            n = _joint_count(joint_values[0])
            if specs is None:
                specs = [Segment(self._rest / n) for _ in range(n)]
            elif len(specs) == 1 and n > 1:
                total = specs[0].length
                specs = [Segment(total / n, specs[0].radius) for _ in range(n)]
            elif len(specs) != n:
                raise InvalidJointCountError(
                    f"joint count {n} does not match {len(specs)} segment lengths")

        new_radius = None
        if radius_arg is not None:
            new_radius = _non_negative(radius_arg, "radius", InvalidArmLengthError)

        rest = None
        if specs is not None:
            rest = math.fsum(seg.length for seg in specs)
            if not math.isfinite(rest) or rest <= 0.0:
                raise InvalidArmLengthError(
                    f"total arm length is not finite/positive: {rest!r}")

        with self._lock:
            if params is not None:
                self._wn, self._zeta, self._branch, self._extra = params
            if specs is not None:
                self._segments = tuple(specs)
                self._rest = rest
                if self._length > rest:
                    self._length = rest
                    self._vel = 0.0
                if new_radius is None:
                    self._radius = max((s.radius for s in specs), default=0.0)
            if new_radius is not None:
                self._radius = new_radius
        return self

    def reset(self):
        """恢复静止总长并清除速度。"""
        with self._lock:
            self._length = self._rest
            self._vel = 0.0
            return self._length

    def __repr__(self):  # pragma: no cover - 调试
        return (f"SpringArm(joints={self.joints}, rest={self._rest!r}, "
                f"current={self._length!r}, omega={self._wn!r}, "
                f"zeta={self._zeta!r})")
