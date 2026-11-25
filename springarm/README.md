# springarm

## 题面（可直接粘贴）

子目录：springarm。弹簧臂相机与碰撞回缩。约束：臂长按碰撞距离回缩（不得穿墙，误差 ≤1e-9）；回缩即时、恢复按弹簧平滑且过冲 ≤5%；碰撞体与射线求交用与 raybox 一致的参数化；目标在臂内时长度为 0；1 万次更新不超过 300 ms；非法臂长可区分报错。验收：cd springarm && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd springarm && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
