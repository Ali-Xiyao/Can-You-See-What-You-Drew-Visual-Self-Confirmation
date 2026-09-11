# 项目审计计划

## Goal

基于项目中的文档、源码、配置、测试、产物与 Git 状态，向用户详细解释项目目标、实现方式、系统流程、当前完成度、可运行性及剩余缺口。

## Current Phase

Phase 9 — complete，已基于本机两张 RTX 3090、现有实测速率和未完成的训练实现，给出候选信号与重复验证的条件性工期估计。

## Phases

### Phase 1: 项目结构与文档盘点

- Status: complete
- 识别技术栈、入口、核心目录、设计文档和数据/模型资产。

### Phase 2: 核心实现与运行流程

- Status: complete
- 阅读关键源码，建立模块关系、数据流和算法流程。

### Phase 3: 完成度与质量验证

- Status: complete
- 检查 Git 历史/状态、测试、构建、运行产物和 TODO/占位实现。

### Phase 4: 综合报告

- Status: complete
- 形成面向用户的详细中文说明，并区分事实、推断和风险。

## Next Step

用户已委托执行并授权使用子代理持续推进。测量修复、隔离划分、动态探针和预注册已完成，正在运行本地双 3090 的首条训练轨迹；按实际每轮耗时和证据决定后续工作。

### Phase 5: 方法评估与建议

- Status: complete
- 解释 D*、D_g、在线报警时间，区分已知现象与待验证的预测假设。
- 核查源码和相关原始研究，提出有边界的最小方法修订。

### Phase 6: 修改后的开跑就绪审查

- Status: complete
- 读取新提交、预注册和实测报告，区分代码已修改与新方法已经验证。
- 检查计数题解析、弃答计分、候选缓存、round-0 权重隔离及训练/评测一致性。
- 运行相关 CPU 单元/集成检查，复现必要问题，给出分阶段开跑判断与具体修改项。

### Phase 7: 开放计数实测解释与下一步

- Status: complete
- 核对全量回答、池内排序、冻结数据配对与预注册判定。
- 区分 spec 不符、图像事实错误、跨模型差异和提示词效应。
- 仅用现有工件进行 CPU 诊断；不启动新模型实验、不改门槛或实验源码。

### Phase 8: 修复后 oracle 提议审查

- Status: complete
- 核对新事实准确率报告、残余池诊断及其使用的题目版本。
- 区分生成正确性已定与每道计数事实已定，处理 unnameable 和 agreed_verdict 的未决 disputed。
- 判断完美事实回答能验证什么，给出一个有限的下一步；不改用户正在修改的 STATUS、源码或运行文件。

### Phase 9: 本地 3090 工期估计

- Status: complete
- 读取当前硬件、preview 配置、脚本串行关系和历史生成/检测/梯度耗时。
- 区分能排期的代码与实验产出、不能保证存在的效果脱钩和梯度提前预警。
- 给出假设双卡持续可用下的工程时间范围，并注明单卡需要先修改调度。

### Phase 10: 实施与动态实验

- Status: in_progress
- 已提交 8ee6b97：事实 unknown 修复、训练/评测/探针隔离、真实参数更新检查、每 checkpoint 效果及梯度报告。
- 407 项 CPU 测试通过；InternVL 真实图片 GPU canary 通过。
- 2026-09-05 18:04 UTC 启动 runs/v4/decoupling-pilot-20260906 的冻结 10 轮 supervisor；当前等待首轮实际更新并持续核验。
- 保存所有旧运行和失败判定；无事件是有效结果，不把 pilot 完成等同于找到两个信号。
- 本任务已建立每 30 分钟后续检查，处理故障并推进已授权工作。

## Decisions Made

| Decision | Rationale |
|---|---|
| 历史审计转入授权实施 | 用户现已明确委托代码修复、启动实验并持续推进，可以使用子代理 |
| 证据来自仓库本身 | 当前问题聚焦本地项目，不需要互联网资料 |
| STATUS 优先于 README/RUNBOOK | README 明确规定 STATUS 是唯一当前状态源；RUNBOOK 包含已声明未实现的历史接口 |
| 把 round-0 隔离问题列为当前阻塞风险 | 源码明确共享参数且仅在 previous checkpoint 存在时恢复，测试未覆盖 stage 生命周期；这会破坏 Gate C 配对设计 |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| `rg` 对 `docs/prereg/*.md` 报 Windows 路径语法错误 | 1 | 后续使用 `rg -g '*.md' docs/prereg` 或显式文件路径 |
| 2026-09-05 读取不存在的 L3 TERMINAL/orchestrator、tierb-conf/conf_fork.txt | 1 | L3 当前终止说明在 STOPPED.txt；第二批确证数字以 STATUS §26 与现有答案工件为据，不假称有该报告文件 |
| 2026-09-05 新审查中猜测的 adapter/observer/paired_gradients 路径不存在 | 1 | 改用 rg 定位真实实现；未执行失败路径上的任何操作 |
| 2026-09-05 定向 Ruff 检查退出 1（10 项导入/风格问题） | 1 | 已记录为非核心阻塞项；审查不自动修改源码 |
| 2026-09-06 图像计数审计中 spec noun 唯一性断言失败 | 1 | 继续检查真实 spec；发现 24 池在同义词合并后包含同类别不同颜色条目，问题问总数而答案用条目数，是实质计分缺陷而非审计脚本应忽略的异常 |
| 2026-09-05 America/Los_Angeles 本轮 rg 使用 scripts/v4* 再遇 Windows 路径语法错误 | 1 | 更正为 rg -g 'v4*.py' scripts；其余独立读取成功，不重复失败命令 |
