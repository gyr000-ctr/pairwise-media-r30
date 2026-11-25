"""autotile 核心 API（初始未实现）。"""


class AutoTile:
    def __init__(self, variants):
        """构造"""
        raise NotImplementedError

    def mask(self, grid, x, y):
        """邻接掩码"""
        raise NotImplementedError

    def pick(self, grid, x, y):
        """选变体"""
        raise NotImplementedError
