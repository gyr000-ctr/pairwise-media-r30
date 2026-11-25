# animblend

## 题面（可直接粘贴）

子目录：animblend。动画混合空间（1D）。约束：按参数在相邻两段间线性混合且权重和为 1（≤1e-12）；端点外钳制到最近段；循环段在接缝处插值（不得跳变）；混合后姿态必须重归一（≤1e-9）；10 万次不超过 200 ms；非法参数可区分报错。验收：cd animblend && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd animblend && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
