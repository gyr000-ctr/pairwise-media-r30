"""ddagrid 核心 API（初始未实现）。"""


class Dda:
    def traverse(self, origin, dirn, w, h):
        """游走"""
        raise NotImplementedError

    def hit_cells(self, a, b, w, h):
        """连接两点的格"""
        raise NotImplementedError
