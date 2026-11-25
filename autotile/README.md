# autotile

## 题面（可直接粘贴）

子目录：autotile。自动拼接瓦片（47-tile 简化）。约束：按四邻接掩码选择变体且掩码与邻接一致；对角邻接单独处理（避免孤立角）；未知掩码回退到默认；边界外按空处理；10 万次查询不超过 200 ms；非法掩码可区分报错。验收：cd autotile && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd autotile && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
