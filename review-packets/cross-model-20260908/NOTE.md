# 第三张表的 `blind only` 列量的是弃答,不是正确率

`report.log` 是当时的原始输出,**不改**。这里登记一个已修复的可读性缺陷,以及修复
不影响任何数字的证明。

## 缺陷

`stage_report` 打三张表。第二张(配对 McNemar)下面有图例:

> 'blind only' = right without the description in context, wrong with it.

第三张(弃答平衡诊断)复用了同样的列名 `blind only` / `told only`,但含义相反——
它数的是**在 blind 条件下拒绝作答、在 told 条件下作答了**的试次。第三张表当时没有
图例,我自己第一次读就把它当成第二张表的列读了,以为列被写反了。核对
`abstention_pairs(blind, prompted)` 与 `mcnemar` 的返回顺序后确认代码是对的,
错的是我。四个模型都用 `decisive − blind_abs − told_abs − both_abs = pairs`
交叉验证过。

## 修复

`scripts/v4_cross_model.py`(armB 分支)第三张表下补了图例,措辞直接点出两张表同名
不同义。同一 commit 还把 `--out` 的目录创建提前到 `measure` 之前:原来 `write_text`
是第一个碰到该路径的调用,`--out` 指向不存在的目录时,整份报告算完、打完,才在最后
一步抛 `FileNotFoundError`,JSON 丢掉,只能重跑。三个新测试先在旧代码上跑红后跑绿。

## 数字未变

用修好的脚本、同样的两个 run 重跑 `report`,输出与本目录的 `cross_model.json`
**逐字节相同**(`diff` 无差异)。所以 `report.log` 与 `cross_model.json` 都保持冻结,
本文件是唯一的增补记录。
