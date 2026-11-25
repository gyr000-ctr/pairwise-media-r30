"""springarm 核心 API（初始未实现）。"""


class SpringArm:
    def __init__(self, length, stiffness, damping):
        """构造"""
        raise NotImplementedError

    def update(self, target, dirn, hit_dist, dt):
        """更新"""
        raise NotImplementedError

    def current(self):
        """当前长度"""
        raise NotImplementedError
