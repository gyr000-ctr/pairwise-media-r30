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
