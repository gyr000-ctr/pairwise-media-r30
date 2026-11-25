"""ddagrid 核心 API：DDA 网格射线遍历（Amanatides & Woo）。

约定：
- 网格为 [0, w) x [0, h)，格 (i, j) 覆盖 [i, i+1) x [j, j+1)。
- 起点归属规则：cell = floor(p)，起点在格边界上时归入正方向一侧的格。
- 角点穿越（tMaxX == tMaxY 的 tie）两轴同时步进，角点只访问一次，
  与逐格超采样得到的格集合一致。
- 越界立即终止；访问格数不超过 |dx| + |dy| + 1。
"""

import math

__all__ = [
    "Dda",
    "DdaError",
    "ZeroDirectionError",
    "NonFiniteInputError",
    "InvalidGridError",
]


class DdaError(ValueError):
    """ddagrid 参数错误基类。"""


class ZeroDirectionError(DdaError):
    """方向向量为零（无法确定步进方向）。"""


class NonFiniteInputError(DdaError):
    """输入含 NaN 或无穷。"""


class InvalidGridError(DdaError):
    """网格尺寸非法。"""


def _check_finite(*vals):
    for v in vals:
        if not math.isfinite(v):
            raise NonFiniteInputError("输入包含非有限值: %r" % (v,))


def _check_grid(w, h):
    if not (isinstance(w, int) and isinstance(h, int)) or w <= 0 or h <= 0:
        raise InvalidGridError("非法网格尺寸: w=%r h=%r" % (w, h))


class Dda:
    """无状态遍历器；所有方法均为纯函数，天然并发安全。"""

    def traverse(self, origin, dirn, w, h):
        """从 origin 沿 dirn 做 DDA 游走，返回访问格 [(i, j), ...]。

        起点在网格外时返回空列表；射线越界立即终止。
        """
        _check_grid(w, h)
        ox, oy = float(origin[0]), float(origin[1])
        dx, dy = float(dirn[0]), float(dirn[1])
        _check_finite(ox, oy, dx, dy)
        if dx == 0.0 and dy == 0.0:
            raise ZeroDirectionError("方向向量为零")

        ix = math.floor(ox)
        iy = math.floor(oy)
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return []

        cells = [(ix, iy)]

        if dx > 0.0:
            step_x = 1
            t_max_x = (ix + 1.0 - ox) / dx
            t_delta_x = 1.0 / dx
        elif dx < 0.0:
            step_x = -1
            t_max_x = (ix - ox) / dx
            t_delta_x = -1.0 / dx
        else:
            step_x = 0
            t_max_x = math.inf
            t_delta_x = math.inf

        if dy > 0.0:
            step_y = 1
            t_max_y = (iy + 1.0 - oy) / dy
            t_delta_y = 1.0 / dy
        elif dy < 0.0:
            step_y = -1
            t_max_y = (iy - oy) / dy
            t_delta_y = -1.0 / dy
        else:
            step_y = 0
            t_max_y = math.inf
            t_delta_y = math.inf

        append = cells.append
        while True:
            if t_max_x < t_max_y:
                ix += step_x
                t_max_x += t_delta_x
            elif t_max_y < t_max_x:
                iy += step_y
                t_max_y += t_delta_y
            else:
                # 角点穿越：两轴同时步进，角点格只进入一次。
                ix += step_x
                iy += step_y
                t_max_x += t_delta_x
                t_max_y += t_delta_y
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                break
            append((ix, iy))
        return cells

    def hit_cells(self, a, b, w, h):
        """返回线段 a -> b 经过的格（含两端所在格）。"""
        _check_grid(w, h)
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        _check_finite(ax, ay, bx, by)
        dx = bx - ax
        dy = by - ay

        ix = math.floor(ax)
        iy = math.floor(ay)
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return []
        if dx == 0.0 and dy == 0.0:
            return [(ix, iy)]

        cells = [(ix, iy)]

        if dx > 0.0:
            step_x = 1
            t_max_x = (ix + 1.0 - ax) / dx
            t_delta_x = 1.0 / dx
        elif dx < 0.0:
            step_x = -1
            t_max_x = (ix - ax) / dx
            t_delta_x = -1.0 / dx
        else:
            step_x = 0
            t_max_x = math.inf
            t_delta_x = math.inf

        if dy > 0.0:
            step_y = 1
            t_max_y = (iy + 1.0 - ay) / dy
            t_delta_y = 1.0 / dy
        elif dy < 0.0:
            step_y = -1
            t_max_y = (iy - ay) / dy
            t_delta_y = -1.0 / dy
        else:
            step_y = 0
            t_max_y = math.inf
            t_delta_y = math.inf

        append = cells.append
        while True:
            if t_max_x < t_max_y:
                if t_max_x > 1.0:
                    break
                ix += step_x
                t_max_x += t_delta_x
            elif t_max_y < t_max_x:
                if t_max_y > 1.0:
                    break
                iy += step_y
                t_max_y += t_delta_y
            else:
                if t_max_x > 1.0:
                    break
                ix += step_x
                iy += step_y
                t_max_x += t_delta_x
                t_max_y += t_delta_y
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                break
            append((ix, iy))
        return cells
