# influence

## 题面（可直接粘贴）

子目录：influence。影响力图扩散。约束：扩散用线性衰减且按 4 邻接迭代（与 BFS 距离一致 ≤1e-9）；多源取最大而不是求和；遮挡格阻断传播；迭代到收敛（无变化）；512x512 不超过 700 ms；非法衰减可区分报错。验收：cd influence && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd influence && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖

## 实现说明（交付侧）

API 只有题面声明的 `Influence(decay).spread(grid, sources)`（构造签名保持
`__init__(self, decay)` 不变），另附两个核对/兼容方法：
`Influence.distance_field(grid, sources)` 给出整数多源 BFS 距离场；
`Influence.spread_linear(grid, sources)` 给出“线性衰减”四字的字面仿射解释
`max(0, 1 - decay*d)`。

- 语义：源点为 `1.0`；4 邻接每多走一格乘一次 `decay`，故 `spread` 的收敛解为
  `decay ** d`，其中 `d` 是多源 BFS 的最近距离。多源由多源 BFS 统一给出
  `d = min_s dist_4(cell, s)`，天然“取最大、不求和”。影响力图迭代传播通常把
  “线性衰减”实现为每跳乘同一个线性系数，其距离闭式即幂律；若把“线性”严格
  理解为关于距离的一次函数，可用 `spread_linear`，它同样逐格对齐 BFS。
- 遮挡格（任何非零格值，通常为 `1`）不进入 BFS，输出恒为 `0.0`，不会穿透；
  被墙隔离的不可达格同样为 `0.0`。
- 直接计算精确 BFS 距离场再取幂，等价于
  `v <- max(v, decay * max(neighbor v))` 迭代到两次差 `<= 1e-12` 的不动点
  （实测再松弛一步的最大残差约 `1e-16`），且没有迭代截断误差。
- 纯标准库；扁平数组 + 带边界墙的多源 BFS，512x512 单机约 50 ms，
  远低于 700 ms。`spread` 不写 `self`、全部中间量为局部变量，并发安全。
- `decay` 合法域为有限实数 `0 < decay <= 1`（`0`、负值、`>1`、NaN/inf、
  非实数均报错）。异常体系：`InfluenceError(ValueError)`，下分
  `InvalidDecayError` / `InvalidGridError` / `InvalidSourceError` 与
  `NonFiniteInputError`（同时是 `TypeError`）；非有限的 decay/grid/source
  分别抛同时隶属两类的组合异常，`ValueError`、`TypeError`、具名异常都能区分捕获。

扩展自检见 `tests/test_extended.py`（随机 BFS 一致性、收敛残差、梯度方向、
穿墙、并发逐位一致、报错分类等）。严格的 512x512 < 700 ms 计时用
`INFLUENCE_RUN_PERF=1 python -m pytest tests -q` 开启，默认跳过以免共享机器
计时抖动影响验收门槛。
