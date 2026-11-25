# ddagrid

## 题面（可直接粘贴）

子目录：ddagrid。DDA 网格射线遍历。约束：访问格序列与解析参数化一致（步进方向与 tMax 更新正确）；访问格数不超过 |dx|+|dy|+1；起点在格边界时归属规则确定（不重复访问）；越界立即终止；10 万次不超过 300 ms；非法方向可区分报错。验收：cd ddagrid && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd ddagrid && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
