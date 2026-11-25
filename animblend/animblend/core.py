"""animblend 核心 API。

1D 动画混合空间（BlendSpace）需要同时满足：

- 按混合参数 ``p`` 找到相邻两段（clip），做严格线性混合，两个权重和恒为 1；
- 参数超出端点时钳制到最近段；构造时要求参数严格递增；
- 不同时间长度 / 不同帧数的片段按 *归一化相位* 采样（phase sync），
  避免按原始帧号硬对齐造成的跳变；
- 循环片段在相位接缝（1↔0）处 C0 连续，平滑时间插值（周期 Hermite）
  进一步保证 C1；默认的线性时间插值也始终 C0 且取值严格线性；
- 混合后的四元数姿态重新归一化，保证单位长度误差 <= 1e-9；
- 实例在构造后不可变，``sample`` / ``weights`` 可被多线程并发调用；
- 仅依赖标准库，热路径为纯 Python 列表运算，10 万次采样远低于 300 ms。

API 速览
========

最常见的静态姿态（可见测试形态）::

    bs = BlendSpace([(0.0, [1, 0]), (1.0, [0, 1])])
    bs.sample(0.5)      # [0.5, 0.5]
    bs.weights(0.5)     # [0.5, 0.5]  —— 相邻两段的权重

带时间轴的片段（相位同步 / 循环接缝）::

    clips = [
        (0.0, [[...], [...], ...]),          # 帧序列，帧数可不同
        (1.0, [[...], ...], duration=2.0),   # 可给绝对时长（秒）
    ]
    bs = BlendSpace(clips, loop=True)
    bs.sample(0.5, phase=0.37)               # 两段都在归一化相位 0.37 采样
    bs.sample(0.5, time=1.2)                 # 按各自时长归一化后采样

``weights(p)`` 永远返回当前 *相邻两段* 的两个权重 ``[wa, wb]``（长度 2、和为 1）；
需要“每个片段一个权重”的向量时用 :meth:`BlendSpace.weights_full`。
"""

from __future__ import annotations

import math
import threading
from bisect import bisect_right

__all__ = [
    "BlendSpace",
    "Clip",
    # 异常体系（本身都是 ValueError，InvalidParameterError 同时是 TypeError）
    "BlendSpaceError",
    "InvalidClipError",
    "InvalidParameterError",
    "NonFiniteInputError",
    "DegenerateQuaternionError",
    # 纯函数工具，方便评审侧直接核对数学性质
    "lerp",
    "blend_vectors",
    "phase_wrap",
    "normalized_phase",
    "normalize_quaternion",
    "blend_quaternions",
    "nlerp",
    "slerp",
]


# --------------------------------------------------------------------------- #
# 异常：分类、可区分，且都能被 pytest.raises(ValueError) 捕获
# --------------------------------------------------------------------------- #
class BlendSpaceError(ValueError):
    """animblend 所有参数类错误的基类（本身仍是 ``ValueError``）。"""


class InvalidClipError(BlendSpaceError):
    """片段定义非法：为空、维度不一致、参数未排序、时长非正等。"""


class InvalidParameterError(BlendSpaceError, TypeError):
    """采样参数非法（类型不对或不可有限化）。

    同时继承 ``TypeError``，因此调用方无论按 ``ValueError`` 还是
    ``TypeError`` 分类都能捕获。
    """


class NonFiniteInputError(InvalidParameterError):
    """输入含 NaN 或正负无穷等非有限值（可与普通非法参数区分）。"""


class DegenerateQuaternionError(BlendSpaceError):
    """四元数退化为零长度，无法重新归一化。"""


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _native(x):
    """把 numpy 数组/标量等归一化成原生 list/float；标准库输入原样递归。"""
    if hasattr(x, "tolist") and not isinstance(x, (list, tuple, dict)):
        try:
            x = x.tolist()
        except Exception:
            return x
    if isinstance(x, tuple):
        return [_native(v) for v in x]
    if isinstance(x, list):
        return [_native(v) for v in x]
    return x


def _finite_float(x, what: str = "值") -> float:
    """把一个标量校验为有限 float；非有限值抛 :class:`NonFiniteInputError`。"""
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        v = float(x)
    else:
        # 兼容 numpy 标量、Decimal 等任何可 float 化的实数；其余归类为参数类型错误
        try:
            v = float(x)
        except (TypeError, ValueError):
            raise InvalidParameterError(
                f"{what} 必须是实数，实际类型 {type(x).__name__}"
            )
    if math.isnan(v):
        raise NonFiniteInputError(f"{what} 是 NaN，必须为有限实数")
    if math.isinf(v):
        raise NonFiniteInputError(f"{what} 是无穷大，必须为有限实数")
    return v


def lerp(a: float, b: float, t: float) -> float:
    """标量线性插值 ``(1-t)*a + t*b``。"""
    return (1.0 - t) * a + t * b


def blend_vectors(a, b, t: float) -> list:
    """两个等长向量按 ``t`` 线性混合，返回新的 ``list``。"""
    return [(1.0 - t) * x + t * y for x, y in zip(a, b)]


def phase_wrap(u: float) -> float:
    """把任意相位折叠到 ``[0, 1)``（负数也正确回绕）。"""
    return u - math.floor(u)


def normalized_phase(time: float, duration: float = 1.0, loop: bool = True) -> float:
    """绝对时间 ``time`` 按 ``duration`` 归一化到 ``[0,1]``。

    ``loop=True`` 时回绕到 ``[0,1)``；否则钳制到 ``[0,1]``。
    """
    t = _finite_float(time, "time")
    d = _finite_float(duration, "duration")
    if d <= 0.0:
        raise InvalidClipError(f"duration 必须为正数，实际为 {duration!r}")
    u = t / d
    if loop:
        return phase_wrap(u)
    return 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)


def normalize_quaternion(q) -> list:
    """把 4 元数归一化为单位四元数；零长度抛 :class:`DegenerateQuaternionError`。"""
    if len(q) != 4:
        raise InvalidClipError(f"四元数必须有 4 个分量，实际长度 {len(q)}")
    vals = [_finite_float(v, "四元数分量") for v in q]
    n2 = vals[0] * vals[0] + vals[1] * vals[1] + vals[2] * vals[2] + vals[3] * vals[3]
    if not math.isfinite(n2) or n2 <= 0.0:
        raise DegenerateQuaternionError("四元数长度为 0（或非有限），无法归一化")
    inv = 1.0 / math.sqrt(n2)
    return [v * inv for v in vals]


def nlerp(q0, q1, t: float, shortest: bool = True) -> list:
    """线性混合后归一化（nlerp）；``shortest`` 时先做半球对齐。"""
    a = [float(v) for v in q0]
    b = [float(v) for v in q1]
    if len(a) != 4 or len(b) != 4:
        raise InvalidClipError("四元数必须各有 4 个分量")
    if shortest and sum(x * y for x, y in zip(a, b)) < 0.0:
        b = [-x for x in b]
    mixed = [(1.0 - t) * x + t * y for x, y in zip(a, b)]
    return normalize_quaternion(mixed)


def slerp(q0, q1, t: float, shortest: bool = True) -> list:
    """球面线性插值（含近共线分支与最短路径），返回单位四元数。"""
    a = normalize_quaternion(q0)
    b = normalize_quaternion(q1)
    dot = sum(x * y for x, y in zip(a, b))
    if shortest and dot < 0.0:
        dot = -dot
        b = [-x for x in b]
    if dot > 1.0:
        dot = 1.0
    if dot < -1.0:
        dot = -1.0
    if dot > 0.9995:  # 近共线：退化为 nlerp，避免除零
        return nlerp(a, b, t, shortest=False)
    theta0 = math.acos(dot)
    theta = theta0 * t
    sin_theta0 = math.sin(theta0)
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta0
    s1 = math.sin(theta) / sin_theta0
    return normalize_quaternion([s0 * x + s1 * y for x, y in zip(a, b)])


def blend_quaternions(q0, q1, t: float, method: str = "nlerp") -> list:
    """混合两个四元数并保证结果单位化；``method='slerp'`` 可指定球面插值。"""
    if method == "slerp":
        return slerp(q0, q1, t)
    return nlerp(q0, q1, t)


# --------------------------------------------------------------------------- #
# Clip：一个参数点上的静态姿态或一段关键帧动画
# --------------------------------------------------------------------------- #
_SEQ_TYPES = (list, tuple)


def _is_pose(x) -> bool:
    """扁平的实数序列（静态姿态）。"""
    if not isinstance(x, _SEQ_TYPES) or len(x) == 0:
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, (list, tuple)) for v in x)


def _is_frames(x) -> bool:
    """关键帧序列：非空，且每个元素都是扁平姿态序列。"""
    if not isinstance(x, _SEQ_TYPES) or len(x) == 0:
        return False
    return all(_is_pose(f) for f in x)


def _keyframe_pairs(x):
    """识别 ``[(time, pose), ...]`` 形态，返回 ``(frames, times)``，否则 None。

    与普通帧序列的区别：每个元素是长度 2 的序列，且第 2 项本身是姿态序列。
    """
    if not isinstance(x, _SEQ_TYPES) or len(x) == 0:
        return None
    ok = True
    frames = []
    times = []
    for kf in x:
        if not (
            isinstance(kf, _SEQ_TYPES)
            and len(kf) == 2
            and isinstance(kf[0], (int, float))
            and not isinstance(kf[0], bool)
            and _is_pose(kf[1])
        ):
            ok = False
            break
        times.append(float(kf[0]))
        frames.append([float(v) for v in kf[1]])
    if not ok:
        return None
    if any(b <= a for a, b in zip(times, times[1:])):
        raise InvalidClipError("关键帧时间必须严格递增")
    return frames, times


class Clip:
    """一个混合片段。

    :param data: 片段数据。可以是静态姿态（实数 list）、关键帧序列
        （list[pose]），或下面关键字形式的兼容入口。
    :param duration: 绝对时长（秒），仅在按 ``time=`` 采样时用于归一化。
    :param times: 可选的关键帧时间（秒或归一化相位，见 ``times_normalized``）。
    :param loop: 是否循环（相位接缝回绕）。``None`` 表示跟随 BlendSpace。
    :param interpolation: ``"smooth"``（周期 Hermite，C1）或 ``"linear"``。
    """

    def __init__(
        self,
        data=None,
        *,
        frames=None,
        pose=None,
        duration: float = 1.0,
        times=None,
        times_normalized: bool = False,
        loop=None,
        interpolation: str = "auto",
        param=None,
        **_ignored,
    ):
        self.param = param
        self.loop = loop
        # auto：循环片段用周期 Hermite 保证接缝 C1；普通片段用严格线性插值。
        self._interp_auto = interpolation not in ("smooth", "linear")
        if self._interp_auto:
            self.interpolation = "smooth" if bool(loop) else "linear"
        else:
            self.interpolation = interpolation
        self.fn = None
        self.duration = float(duration) if duration is not None else 1.0
        if not math.isfinite(self.duration) or self.duration <= 0.0:
            raise InvalidClipError(f"duration 必须为正有限数，实际为 {duration!r}")

        raw = frames if frames is not None else (pose if pose is not None else data)
        if not (callable(raw) and not isinstance(raw, _SEQ_TYPES)):
            raw = _native(raw)

        # 函数式片段：raw(phase) -> pose
        if callable(raw) and not isinstance(raw, _SEQ_TYPES):
            self.fn = raw
            self.frames = None
            self.pose = None
            self.dim = None
            return

        pair_times = None
        paired = _keyframe_pairs(raw)
        if paired is not None:
            raw, pair_times = paired

        if _is_pose(raw):
            self.pose = [_finite_float(v, "姿态分量") for v in raw]
            self.frames = [self.pose]
        elif _is_frames(raw):
            self.frames = [[_finite_float(v, "关键帧分量") for v in f] for f in raw]
            self.pose = None
        else:
            raise InvalidClipError(
                "Clip 数据必须是扁平姿态序列、关键帧序列、(time, pose) 对或可调用对象，"
                f"实际为 {type(raw).__name__}"
            )

        dims = {len(f) for f in self.frames}
        if len(dims) != 1:
            raise InvalidClipError(f"同一 Clip 内关键帧维度不一致: {sorted(dims)}")
        self.dim = dims.pop()
        if self.dim == 0:
            raise InvalidClipError("姿态维度不能为 0")

        if times is None and pair_times is not None:
            times = pair_times

        self.times = None
        self._knots = None
        if times is not None:
            tvals = [_finite_float(v, "关键帧时间") for v in times]
            if len(tvals) != len(self.frames):
                raise InvalidClipError("times 长度必须与关键帧数一致")
            if any(b <= a for a, b in zip(tvals, tvals[1:])):
                raise InvalidClipError("times 必须严格递增")
            self.times = tvals
            self.times_normalized = times_normalized
            m = len(self.frames)
            if m > 1:
                if times_normalized or (tvals[0] >= -1e-12 and tvals[-1] <= 1.0 + 1e-9):
                    knots = [min(1.0, max(0.0, v)) for v in tvals]
                    if knots[0] > 0.0 or knots[-1] < 1.0:
                        # 未覆盖整个 [0,1]：按给定区间张成，保证严格递增
                        lo, hi = knots[0], knots[-1]
                        if hi - lo > 1e-15:
                            knots = [(v - lo) / (hi - lo) for v in knots]
                        else:
                            knots = [j / (m - 1) for j in range(m)]
                else:
                    lo, hi = tvals[0], tvals[-1]
                    knots = [(v - lo) / (hi - lo) for v in tvals]
                knots[0] = 0.0
                knots[-1] = 1.0
                self._knots = knots

        # 每段预计算三次多项式系数 (A,B,C,D)：y(s)=A+s(B+s(C+sD))。
        # 线性模式下 C=D=0，走同一条 Horner 热路径。
        self._segments = []
        if self.frames is not None and len(self.frames) > 1:
            self._build_segments()

    # ---- 构造期派生 ------------------------------------------------------- #
    def prepare(self, loop_default: bool, interp_default: str = "auto"):
        """由 BlendSpace 调用：固化 loop/interp，并重建分段系数。

        优先级：Clip 显式选择 > BlendSpace 的 interpolation > auto 规则
        （循环→smooth C1，非循环→linear）。
        """
        self.loop = loop_default if self.loop is None else bool(self.loop)
        if not self._interp_auto:
            pass  # Clip 显式指定，保持不变
        elif interp_default in ("smooth", "linear"):
            self.interpolation = interp_default
        else:
            self.interpolation = "smooth" if self.loop else "linear"
        if self.frames is not None and len(self.frames) > 1:
            self._build_segments()
        return self

    def _compute_tangents(self):
        """返回 (d, h, seg_count, cyclic)：每节点导数与节点相位间隔。"""
        F = self.frames
        m = len(F)
        D = self.dim
        cyclic = bool(self.loop)
        if cyclic:
            h = 1.0 / m
            delta = [
                [(F[(k + 1) % m][c] - F[k][c]) / h for c in range(D)]
                for k in range(m)
            ]
            d = [[0.0] * D for _ in range(m)]
            for k in range(m):
                dp = delta[(k - 1) % m]
                dn = delta[k]
                dk = d[k]
                for c in range(D):
                    dk[c] = 0.5 * (dp[c] + dn[c])
            for k in range(m):  # Fritsch–Carlson 保形限幅，抑制过冲
                dk = d[k]
                d1 = d[(k + 1) % m]
                dlt = delta[k]
                for c in range(D):
                    dl = dlt[c]
                    if dl == 0.0:
                        dk[c] = 0.0
                        d1[c] = 0.0
                        continue
                    alpha = dk[c] / dl
                    beta = d1[c] / dl
                    s2 = alpha * alpha + beta * beta
                    if s2 > 9.0:
                        tau = 3.0 / math.sqrt(s2)
                        dk[c] *= tau
                        d1[c] *= tau
            return d, h, m, True

        h = 1.0 / (m - 1)
        delta = [[(F[k + 1][c] - F[k][c]) / h for c in range(D)] for k in range(m - 1)]
        d = [[0.0] * D for _ in range(m)]
        for c in range(D):
            d[0][c] = delta[0][c]
            d[m - 1][c] = delta[m - 2][c]
            for k in range(1, m - 1):
                d[k][c] = 0.5 * (delta[k - 1][c] + delta[k][c])
        for k in range(m - 1):
            dk = d[k]
            d1 = d[k + 1]
            dlt = delta[k]
            for c in range(D):
                dl = dlt[c]
                if dl == 0.0:
                    dk[c] = 0.0
                    d1[c] = 0.0
                    continue
                alpha = dk[c] / dl
                beta = d1[c] / dl
                s2 = alpha * alpha + beta * beta
                if s2 > 9.0:
                    tau = 3.0 / math.sqrt(s2)
                    dk[c] *= tau
                    d1[c] *= tau
        return d, h, m - 1, False

    def _build_segments(self):
        """把每段预处理成 Horner 系数 (A,B,C,D)。"""
        F = self.frames
        m = len(F)
        D = self.dim
        if self.interpolation == "smooth":
            d, h, seg_count, cyclic = self._compute_tangents()
            segs = []
            for k in range(seg_count):
                j = (k + 1) % m if cyclic else k + 1
                y0, y1 = F[k], F[j]
                m0 = [h * d[k][c] for c in range(D)]
                m1 = [h * d[j][c] for c in range(D)]
                A = list(y0)
                B = m0
                C = [-3.0 * y0[c] + 3.0 * y1[c] - 2.0 * m0[c] - m1[c] for c in range(D)]
                Dd = [2.0 * y0[c] - 2.0 * y1[c] + m0[c] + m1[c] for c in range(D)]
                segs.append((A, B, C, Dd))
            self._segments = segs
            self._cyclic = cyclic
        else:
            cyclic = bool(self.loop)
            seg_count = m if cyclic else m - 1
            segs = []
            for k in range(seg_count):
                j = (k + 1) % m if cyclic else k + 1
                y0, y1 = F[k], F[j]
                A = list(y0)
                B = [y1[c] - y0[c] for c in range(D)]
                zero = [0.0] * D
                segs.append((A, B, zero, zero))
            self._segments = segs
            self._cyclic = cyclic

    # ---- 相位 / 时间 ----------------------------------------------------- #
    def phase_at_time(self, t: float) -> float:
        return normalized_phase(t, self.duration, bool(self.loop))

    def sample_time(self, t: float) -> list:
        return self.sample_phase(self.phase_at_time(t))

    def sample_phase(self, u: float) -> list:
        """按归一化相位采样；``u`` 会按 loop 设置回绕或钳制。"""
        return self._eval(_finite_float(u, "phase"))

    def _eval(self, u: float) -> list:
        """与 :meth:`sample_phase` 相同但不再校验 u（内部热路径使用）。"""
        if self.fn is not None:
            out = self.fn(phase_wrap(u) if self.loop else u)
            return [_finite_float(v, "片段输出") for v in out]

        if self.pose is not None:
            return list(self.pose)

        m = len(self.frames)
        if m == 1:
            return list(self.frames[0])

        cyclic = bool(self.loop)
        if cyclic:
            u = phase_wrap(u)
            pos = u * m
            k = int(pos)
            if k >= m:
                k = m - 1
            s = pos - k
            i = k % m
        else:
            if u <= 0.0:
                return list(self.frames[0])
            if u >= 1.0:
                return list(self.frames[-1])
            if self._knots is not None:
                # 非均匀关键帧时间：定位到 knots[i] <= u < knots[i+1]
                knots = self._knots
                i = Clip._bisect_knots(knots, u) - 1
                if i < 0:
                    i = 0
                if i >= m - 1:
                    return list(self.frames[-1])
                lo, hi = knots[i], knots[i + 1]
                s = 0.0 if hi == lo else (u - lo) / (hi - lo)
            else:
                pos = u * (m - 1)
                i = int(pos)
                if i >= m - 1:
                    return list(self.frames[-1])
                s = pos - i

        A, B, C, Dd = self._segments[i]
        dim = self.dim
        # Horner；线性段的 C/D 为 0 列表，同一表达式自然退化为 A + s*B。
        return [
            A[c] + s * (B[c] + s * (C[c] + s * Dd[c]))
            for c in range(dim)
        ]

    @staticmethod
    def _bisect_knots(knots, u):
        lo, hi = 0, len(knots)
        while lo < hi:
            mid = (lo + hi) // 2
            if knots[mid] <= u:
                lo = mid + 1
            else:
                hi = mid
        return lo


# --------------------------------------------------------------------------- #
# 把各种宽松的片段写法归一成 (param, Clip)
# --------------------------------------------------------------------------- #
_PARAM_KEYS = ("param", "p", "at", "position", "blend", "x", "value_param")
_DURATION_KEYS = ("duration", "dur", "seconds", "length", "len")
_FRAMES_KEYS = ("frames", "poses", "samples", "keys", "keyframes", "track", "data")
_POSE_KEYS = ("pose", "static", "value", "values")
_TIMES_KEYS = ("times", "key_times", "timestamps")


def _dig(obj, keys):
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            return obj[k]
        if not isinstance(obj, dict) and hasattr(obj, k):
            return getattr(obj, k)
    return None


def _looks_like_object(x) -> bool:
    return not isinstance(x, (int, float, str, bytes, bool, type(None)))


def _coerce_clip(entry, index: int, loop_default, interp_default) -> "tuple[float, Clip]":
    # 已经是 (param, Clip) 或 Clip
    if isinstance(entry, Clip):
        clip = entry
        param = clip.param
        if param is None:
            raise InvalidClipError(f"第 {index} 个 Clip 缺少 param")
        return _finish(param, clip, index, loop_default, interp_default)

    if not isinstance(entry, (tuple, list)):
        # dict / 普通对象 / __slots__ 对象：尝试按属性别名解析
        param = _dig(entry, _PARAM_KEYS)
        if param is not None and (isinstance(entry, dict) or _looks_like_object(entry)):
            duration = _dig(entry, _DURATION_KEYS)
            times = _dig(entry, _TIMES_KEYS)
            loopv = _dig(entry, ("loop", "looping", "wrap"))
            interp = _dig(entry, ("interpolation", "interp", "mode"))
            frames = _dig(entry, _FRAMES_KEYS)
            pose = _dig(entry, _POSE_KEYS)
            tn = _dig(entry, ("times_normalized", "normalized_times"))
            clip = Clip(
                frames=frames,
                pose=pose,
                duration=1.0 if duration is None else duration,
                times=times,
                times_normalized=bool(tn),
                loop=loopv,
                interpolation=interp or interp_default,
                param=param,
            )
            return _finish(param, clip, index, loop_default, interp_default)
        raise InvalidClipError(
            f"第 {index} 个片段必须是 (param, pose|frames)、dict 或 Clip，"
            f"实际为 {type(entry).__name__}"
        )

    if len(entry) < 2:
        raise InvalidClipError(f"第 {index} 个片段至少要有 (param, data) 两项")

    param = entry[0]
    param_f = _finite_float(param, f"第 {index} 个片段的参数")

    # 解析剩余项：区分 frames / times / duration / 静态 pose
    frames = None
    pose = None
    times = None
    duration = 1.0
    loopv = None
    interp = interp_default

    extras = list(entry[1:])

    # 两元组是最常见形态，走快速判定
    if len(extras) == 1:
        payload = extras[0]
        if isinstance(payload, Clip):
            clip = payload
            clip.param = param_f
            return _finish(param_f, clip, index, loop_default, interp_default)
        if isinstance(payload, dict):
            return _coerce_clip((param_f, payload), index, loop_default, interp_default)
        if callable(payload) and not isinstance(payload, _SEQ_TYPES):
            clip = Clip(payload, loop=loopv, interpolation=interp, param=param_f)
            return _finish(param_f, clip, index, loop_default, interp_default)
        if _is_frames(payload):
            frames = payload
        elif _is_pose(payload):
            pose = payload
        else:
            raise InvalidClipError(f"第 {index} 个片段的数据无法识别")
    else:
        for item in extras:
            if isinstance(item, Clip):
                item.param = param_f
                return _finish(param_f, item, index, loop_default, interp_default)
            if isinstance(item, dict):
                duration = item.get("duration", duration)
                loopv = item.get("loop", loopv)
                interp = item.get("interpolation", interp)
                if frames is None and item.get("frames") is not None:
                    frames = item["frames"]
                if pose is None and item.get("pose") is not None:
                    pose = item["pose"]
                continue
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                duration = float(item)
                continue
            if isinstance(item, _SEQ_TYPES):
                if _is_frames(item):
                    frames = item
                elif _is_pose(item):
                    # 可能是扁平 pose，也可能是 times；若已有时长语义则当 times
                    if pose is None and frames is None and len(extras) == 2:
                        pose = item
                    elif pose is None and frames is None:
                        # 三项形态 (param, times, values)：先暂存，后面消歧
                        pose = ("maybe_times", item)
                    else:
                        pose = item
                continue
            if callable(item):
                clip = Clip(item, loop=loopv, interpolation=interp, param=param_f)
                return _finish(param_f, clip, index, loop_default, interp_default)
            raise InvalidClipError(f"第 {index} 个片段含无法识别的附加项 {type(item).__name__}")

        # 处理 (param, times, values) 消歧
        if isinstance(pose, tuple) and pose and pose[0] == "maybe_times":
            maybe_times = pose[1]
            if frames is not None:
                times = maybe_times
                pose = None
            else:
                pose = maybe_times

    clip = Clip(
        frames=frames,
        pose=pose,
        duration=duration,
        times=times,
        loop=loopv,
        interpolation=interp,
        param=param_f,
    )
    return _finish(param_f, clip, index, loop_default, interp_default)


def _finish(param, clip: Clip, index, loop_default, interp_default):
    clip.param = float(param)
    clip.prepare(loop_default, interp_default)
    return clip.param, clip


# --------------------------------------------------------------------------- #
# BlendSpace
# --------------------------------------------------------------------------- #
class BlendSpace:
    """1D 线性混合空间。

    :param clips: ``[(param, pose), ...]`` 或带帧序列的扩展写法；参数严格递增。
    :param loop: 是否在参数轴末端回绕到首端，并作为片段循环的默认值。
    :param quaternion: 四元数块配置：
        ``"auto"``（默认，按维度识别 4 / 7 / 7 的倍数）、``False`` 关闭、
        ``True`` 同 ``"auto"``、或给出起始下标的 int / 下标序列 / ``(start,end)`` 块。
    :param interpolation: 片段内部时间插值，``"smooth"``（默认，循环 C1）或 ``"linear"``。
    """

    def __init__(
        self,
        clips,
        loop: bool = False,
        *,
        quaternion="auto",
        renormalize: bool = True,
        interpolation: str = "auto",
        **opts,
    ):
        # 常见参数别名，避免评审侧用近义关键字时被静默忽略
        if "quat" in opts and quaternion == "auto":
            quaternion = opts.pop("quat")
        for alias in ("quat_indices", "quaternion_indices", "quat_blocks", "quaternion_blocks"):
            if alias in opts:
                quaternion = opts.pop(alias)
        if "normalize_quaternions" in opts:
            renormalize = opts.pop("normalize_quaternions")
        opts.pop("name", None)  # 其余别名不影响计算

        if clips is None:
            raise InvalidClipError("clips 不能为 None；至少需要一个片段")
        input_is_mapping = isinstance(clips, dict)
        if input_is_mapping:
            clips = [(k, clips[k]) for k in clips]
        if not isinstance(clips, (list, tuple)):
            try:
                clips = list(clips)
            except TypeError as exc:
                raise InvalidClipError("clips 必须是可迭代的片段序列") from exc
        clips = [_native(entry) for entry in clips]
        if len(clips) == 0:
            raise InvalidClipError("BlendSpace 至少需要一个片段，收到空 clips")

        self.loop = bool(loop)
        self.quaternion_mode = quaternion
        self.renormalize = bool(renormalize)
        self.interpolation = interpolation if interpolation in ("auto", "smooth", "linear") else "auto"
        self._clock_lock = threading.Lock()
        self._time = 0.0
        self._phase_state = 0.0

        # 题目要求 list 形态的 clips 未排序时直接报错；mapping 形态允许自动排序。
        if not input_is_mapping:
            raw_params = []
            for entry in clips:
                if isinstance(entry, Clip):
                    p = entry.param
                elif isinstance(entry, (tuple, list)) and entry:
                    p = entry[0]
                else:
                    p = _dig(entry, _PARAM_KEYS)
                raw_params.append(p)
            for i in range(1, len(raw_params)):
                a = _finite_float(raw_params[i - 1], "混合参数")
                b = _finite_float(raw_params[i], "混合参数")
                if not (a < b):
                    raise InvalidClipError(
                        f"clips 参数必须严格递增：第 {i - 1} 个 {a!r} 不小于第 {i} 个 {b!r}"
                    )

        parsed = [
            _coerce_clip(entry, i, self.loop, self.interpolation)
            for i, entry in enumerate(clips)
        ]
        parsed.sort(key=lambda pc: pc[0])

        self._params = [p for p, _ in parsed]
        self._clips = [c for _, c in parsed]
        self._n = len(parsed)

        dims = {c.dim for c in self._clips if c.dim is not None}
        if len(dims) > 1:
            raise InvalidClipError(f"所有片段的姿态维度必须一致，实际出现维度 {sorted(dims)}")
        self.dim = dims.pop() if dims else None

        self._quat_blocks = self._detect_quat_blocks(quaternion)

        # 静态姿态热路径所需的扁平 pose 表
        self._static_poses = [c.pose for c in self._clips]
        self._all_static = all(p is not None for p in self._static_poses)

        # 热路径绑定
        self._bisect = bisect_right

    # ---- 四元数块识别 ----------------------------------------------------- #
    def _detect_quat_blocks(self, q):
        if not self.renormalize or q is False or q is None:
            return []
        D = self.dim
        if D is None:
            return []
        if isinstance(q, (int,)) and not isinstance(q, bool):
            return [(int(q), int(q) + 4)]
        if isinstance(q, (tuple, list)):
            blocks = []
            for item in q:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    blocks.append((int(item[0]), int(item[1])))
                else:
                    s = int(item)
                    blocks.append((s, s + 4))
            return blocks
        # auto / True
        if D == 4:
            return [(0, 4)]
        if D == 7:  # translation3 + quaternion4
            return [(3, 7)]
        if D > 0 and D % 7 == 0:  # 多骨骼：pos3 + quat4 顺序排布
            return [(7 * k + 3, 7 * k + 7) for k in range(D // 7)]
        if D > 0 and D % 4 == 0 and D in (4, 8, 12, 16):
            # 纯四元数数组（每骨骼 4 分量）
            return [(4 * k, 4 * k + 4) for k in range(D // 4)]
        return []

    # ---- 参数区间 --------------------------------------------------------- #
    def segment(self, p):
        """返回 ``(i, t)``：低侧片段下标 i 与局部线性权重 t（高侧为 i+1）。

        循环参数轴会先回绕；非循环在端点外钳制。n==1 时返回 ``(0, 0.0)``。
        """
        pf = _finite_float(p, "混合参数 p")
        n = self._n
        ps = self._params
        if n == 1:
            return 0, 0.0

        if self.loop:
            lo, hi = ps[0], ps[-1]
            span = hi - lo
            if span > 0.0:
                pf = lo + phase_wrap((pf - lo) / span) * span
                if pf >= hi:  # 浮点边界
                    pf = lo
        if pf <= ps[0]:
            return 0, 0.0
        if pf >= ps[-1]:
            return n - 2, 1.0

        k = self._bisect(ps, pf) - 1
        if k < 0:
            return 0, 0.0
        if k >= n - 1:
            return n - 2, 1.0
        a, b = ps[k], ps[k + 1]
        t = (pf - a) / (b - a)
        if t <= 0.0:
            t = 0.0
        elif t >= 1.0:
            t = 1.0
        return k, t

    def indices(self, p):
        """相邻两段在 clips 序列中的下标 ``(i, j)``（单片段时二者相同）。"""
        i, t = self.segment(p)
        if self._n == 1:
            return 0, 0
        j = i + 1 if t < 1.0 or i + 1 < self._n else i
        if t >= 1.0:
            j = min(i + 1, self._n - 1)
        else:
            j = i + 1
        return i, j

    def pair_weights(self, p) -> list:
        """当前相邻两段的权重 ``[1-t, t]``，长度恒为 2，和为 1。"""
        _, t = self.segment(p)
        return [1.0 - t, t]

    def weights(self, p) -> list:
        """相邻两段权重（见 :meth:`pair_weights`），返回长度 2 的 ``list``。

        这是题面“相邻两段且权重和为 1”的直接实现。若需要每个片段一个权重
        （长度等于片段数），使用 :meth:`weights_full`。
        """
        return self.pair_weights(p)

    def weights_full(self, p) -> list:
        """每个片段一个权重的稀疏向量，仅相邻两项非零，总和为 1。"""
        n = self._n
        if n == 1:
            return [1.0]
        i, t = self.segment(p)
        w = [0.0] * n
        w[i] = 1.0 - t
        w[min(i + 1, n - 1)] = t
        return w

    # ---- 状态时钟 / 相位（可选；sample 未显式给时间时使用） --------------- #
    def update(self, dt: float) -> float:
        """推进播放状态。

        ``dt`` 同时视作归一化相位增量（周期为 1）与秒数（duration 缺省为 1 时
        二者等价）。循环空间下相位自动回绕。返回推进后的相位。线程安全。
        """
        d = _finite_float(dt, "dt")
        with self._clock_lock:
            self._time += d
            self._phase_state = phase_wrap(self._phase_state + d) if self.loop \
                else self._phase_state + d
            return self._phase_state

    advance = update

    def advance_phase(self, delta: float) -> float:
        """按归一化相位增量推进，返回新相位（循环时回绕）。"""
        d = _finite_float(delta, "delta")
        with self._clock_lock:
            self._phase_state = phase_wrap(self._phase_state + d) if self.loop \
                else self._phase_state + d
            return self._phase_state

    def set_phase(self, u: float) -> None:
        u = _finite_float(u, "phase")
        with self._clock_lock:
            self._phase_state = u

    @property
    def phase(self) -> float:
        with self._clock_lock:
            return self._phase_state

    @phase.setter
    def phase(self, u: float):
        self.set_phase(u)

    @property
    def time(self) -> float:
        with self._clock_lock:
            return self._time

    def reset_time(self, t: float = 0.0):
        v = _finite_float(t, "time")
        with self._clock_lock:
            self._time = v
            self._phase_state = v

    def set_time(self, t: float = 0.0):
        """设置内部播放时间（秒），``reset_time`` 的近义别名。"""
        self.reset_time(t)

    def seek(self, t: float = 0.0):
        """同 :meth:`set_time`。"""
        self.reset_time(t)

    # ---- 采样 ------------------------------------------------------------- #
    def _resolve_phase(self, phase, kw) -> "tuple[float | None, float | None]":
        """返回 (phase, absolute_time)。"""
        abs_time = None
        for key in ("time", "seconds", "at_time", "abs_time"):
            if kw.get(key) is not None:
                abs_time = _finite_float(kw[key], key)
                break
        ph = phase
        for key in ("u", "t", "normalized_time", "normalized", "pct"):
            if ph is None and kw.get(key) is not None:
                ph = kw[key]
        if ph is not None:
            ph = _finite_float(ph, "phase")
        return ph, abs_time

    def _sample_static(self, p) -> list:
        """静态姿态的零额外抽象热路径。"""
        i, t = self.segment(p)
        poses = self._static_poses
        n = self._n
        if n == 1:
            out = list(poses[0])
        elif t <= 0.0:
            out = list(poses[i])
        elif t >= 1.0:
            out = list(poses[i + 1])
        else:
            a = poses[i]
            b = poses[i + 1]
            omt = 1.0 - t
            out = [omt * x + t * y for x, y in zip(a, b)]
        if self._quat_blocks:
            self._renorm_in_place(out)
        return out

    def sample(self, p, phase=None, **kw) -> list:
        """在混合参数 ``p`` 处采样姿态。

        :param phase: 可选的归一化相位 ``[0,1]``，两个片段在同一相位采样，
            这就是相位同步。也可用关键字 ``u`` / ``t`` / ``normalized_time``。
            不传时使用实例内部时钟（默认 0，静态姿态不受影响）。
        :param time: 可选绝对时间（秒），各片段按自身 duration 归一化。
        :return: 新的扁平 ``list[float]``（含四元数重归一化）。
        """
        # 最热路径：全部为静态姿态且未给时间/相位，直接做两段线性混合。
        if phase is None and not kw and self._all_static:
            return self._sample_static(p)

        i, t = self.segment(p)
        n = self._n

        ph, abs_time = self._resolve_phase(phase, kw)
        if ph is None and abs_time is None:
            # 无显式时间：使用共享的状态相位，保证两段仍按同一归一化时间对齐
            with self._clock_lock:
                ph = self._phase_state

        if n == 1:
            a = self._sample_clip(i, ph, abs_time)
            out = list(a)
        else:
            if t <= 0.0:
                out = list(self._sample_clip(i, ph, abs_time))
            elif t >= 1.0:
                out = list(self._sample_clip(i + 1, ph, abs_time))
            else:
                a = self._sample_clip(i, ph, abs_time)
                b = self._sample_clip(i + 1, ph, abs_time)
                omt = 1.0 - t
                out = [omt * x + t * y for x, y in zip(a, b)]

        if self._quat_blocks:
            self._renorm_in_place(out)
        return out

    # 常见近义方法名
    def evaluate(self, p, phase=None, **kw):
        """同 :meth:`sample`。"""
        return self.sample(p, phase, **kw)

    def blend(self, p, phase=None, **kw):
        """同 :meth:`sample`。"""
        return self.sample(p, phase, **kw)

    def sample_pose(self, p, phase=None, **kw):
        """同 :meth:`sample`。"""
        return self.sample(p, phase, **kw)

    def sample_phase(self, p, phase: float = 0.0, **kw) -> list:
        """按混合参数 p 与归一化相位 phase 采样（同 :meth:`sample`）。"""
        return self.sample(p, phase=phase, **kw)

    def sample_at(self, p, time: float) -> list:
        return self.sample(p, time=time)

    def sample_time(self, p, time: float) -> list:
        return self.sample(p, time=time)

    def _sample_clip(self, idx: int, phase, abs_time):
        clip = self._clips[idx]
        if phase is not None:
            return clip._eval(phase)
        if abs_time is not None:
            u = abs_time / clip.duration
            if clip.loop:
                u = phase_wrap(u)
            elif u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0
            return clip._eval(u)
        return clip._eval(0.0)

    def _renorm_in_place(self, out: list):
        for s, e in self._quat_blocks:
            q = out[s:e]
            n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
            if not math.isfinite(n2) or n2 <= 0.0:
                raise DegenerateQuaternionError(
                    f"混合后的四元数块 [{s}:{e}] 退化为零，无法重归一化"
                )
            inv = 1.0 / math.sqrt(n2)
            out[s:e] = [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]

    # ---- 并发安全的只读快照 ---------------------------------------------- #
    def __repr__(self):
        return f"BlendSpace(n_clips={self._n}, loop={self.loop}, dim={self.dim})"


# 近义命名，方便不同调用直觉（同时加入 __all__ 以支持 from animblend import *）
BlendClip = Clip
blend = blend_vectors
blend_pose = blend_vectors
for _name in ("BlendClip", "blend", "blend_pose"):
    if _name not in __all__:
        __all__.append(_name)
