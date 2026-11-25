"""animblend 核心 API（初始未实现）。"""


class BlendSpace:
    def __init__(self, clips):
        """构造"""
        raise NotImplementedError

    def sample(self, p):
        """采样"""
        raise NotImplementedError

    def weights(self, p):
        """权重"""
        raise NotImplementedError
