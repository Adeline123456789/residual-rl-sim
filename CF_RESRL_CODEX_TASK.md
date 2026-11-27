# CF-ResRL：面向 Codex 的最小验证与终版扩展任务书

> **固定研究问题**  
> **Can factorial counterfactual supervision improve the data efficiency of residual-aware replanning decisions for action-chunked robot policies?**  
> 对动作块机器人策略，显式利用“是否施加残差 × 是否刷新计划”的 2×2 配对反事实监督，能否比普通残差条件化模型以更少的真实环境交互学会更好的重规划决策？

本任务书替代本分支上的 PA-ResWM 执行路线，但保留旧文档和旧分支作为研究记录。不要删除或改写历史结果。

---

## 0. 第一原则：不移动终点，不用结果门槛阻塞实验

每个 Codex 子任务开始前先回答一句：

> **“这个实验怎样直接推进最终研究问题？”**

必须遵守以下纪律。

1. **研究问题与强基线固定。** 核心比较始终是 P1 对 B5（同四分支数据）以及 P1 对 B6（同环境分支预算）。实验中途想到更严格条件、辅助模型或新论文，不得自动把它们改成当前阶段的前置任务。
2. **阶段推进只看实现和实验有效性，不看结果方向。** P1 没有超过 B5/B6、自然分布翻转率低、置信区间跨零、某个提升小于预期，都属于实验结果，不是停止后续预注册扩展的理由。
3. **NO-GO 只否定对应 claim。** 例如 P1≈B5 只否定“factorial decomposition 有额外优势”，不否定 residual-aware replanning，也不否定整个项目。
4. **辅助模块不能绑架主实验。** World model、Agent、Memory、Vision/Event Observer、Direction Classifier、selective repair、base distillation 均不是核心依赖。缺失或效果差只能记录，不能阻止主实验继续。
5. **只有实验无效问题才是 Hard Gate。** 包括：
   - 基础轨迹根本无法执行或任务接口错误；
   - 仿真快照不能正确恢复，四分支语义不一致；
   - Student 使用部署时不可用的 privileged information；
   - 动作、碰撞或物理安全限制被破坏；
   - 同一 source episode / snapshot 泄漏到训练和测试；
   - 运行证据丢失、损坏或被伪造。
6. **MVP 有效跑完后进入扩展实验。** 不得反复要求 MVP 重新证明“存在价值”。扩展阶段研究能力边界、跨条件泛化及负结果。

---

## 1. 数据集与环境选择

### 1.1 主数据：Square-MH 图像数据

最小验证使用：

```text
Hugging Face: ankile/robomimic-mh-square-image
任务: Robomimic / Robosuite Square
规模: 300 episodes
频率: 20 Hz
观测: agentview + eye-in-hand 两路 84×84 图像，9 维本体状态
动作: 7 维
```

选择它的原因：

- 它是官方 ResFiT Square 配置直接使用的数据，能够复用 ACT 与 residual TD3 训练入口；
- Square 是精密装配任务，动作块后缀是否仍然有效比简单抓取更容易形成可观察差异；
- Robosuite / MuJoCo 可以恢复物理状态，支持同一中间状态下的配对分支；
- 7 维动作使残差方向和幅度实验可控，工程成本低于双臂任务。

官方 ResFiT 上游固定为：

```text
amazon-far/residual-offpolicy-rl
commit: 66262b06a0caf159fb001504949b7b48ccda5f64
robosuite commit: 77a4751233c29456a5381209e30dd0dbf39a6557
MuJoCo: 3.3.2
```

不要从最新 main 静默升级依赖。任何升级建立独立配置和结果目录。

### 1.2 扩展数据：Can-MH 图像数据

第一项跨任务扩展使用：

```text
ankile/robomimic-mh-can-image
300 episodes, 20 Hz, 2 cameras, 7-D actions
```

Can 用于回答方法是否只在插接/对准任务成立。它不是 MVP 前置条件。

### 1.3 真正的研究数据：派生 2×2 分支数据集

现成示教数据没有 `V00/V01/V10/V11` 标签，因此不能直接验证研究问题。必须在训练好的 ACT + ResFiT 上生成派生数据：

```text
cf_square_factorial_v1
```

每个 source snapshot 固定控制上下文 `h_t`、剩余动作块 `B_t` 和候选残差 `δ_t`，然后恢复同一完整快照运行四个分支：

| 标签 | 当前动作干预 | 后续计划干预 |
|---|---|---|
| V00 | 当前步 residual=0 | KEEP 原缓存块 |
| V01 | 当前步 residual=0 | 当前步后发起 REFRESH |
| V10 | 执行候选 residual `δ_t` | KEEP 原缓存块 |
| V11 | 执行同一 `δ_t` | 当前步后发起 REFRESH |

四个值必须使用相同的剩余 horizon、奖励定义、后续 ACT/ResFiT 策略、随机数协议和延迟语义。

派生量：

```text
G0 = V01 - V00
Gδ = V11 - V10
Iδ = V11 - V10 - V01 + V00
Gδ = G0 + Iδ
```

实际决策依据是 `Gδ`，不是单独依据 `Iδ`。

---

## 2. M0：基础环境、数据和基线复现

### 2.1 数据审计必须先于长训练

执行并提交：

- 数据可下载性和 license 记录；
- episode 数、帧数、相机键、动作维度、控制频率；
- 视频解码小样本；
- action/state 统计量和 NaN/Inf 检查；
- episode 边界、done/truncation 对齐；
- 原始数据 revision / commit / SHA256；
- 训练、验证、测试的 episode-group split manifest。

数据只有官方 `train` split 时，自行按 episode 建立固定 split。禁止按帧随机切分。

### 2.2 ACT 与 ResFiT

优先尝试读取官方配置引用的 checkpoint。不可访问时，按固定上游代码训练，不得临时换另一套算法。

必须保存：

- ACT 最终 checkpoint；
- ResFiT 训练计划中的固定步数 checkpoint，而不是按测试表现挑 checkpoint；
- resolved config、seed、上游 commit、依赖清单；
- 每个 checkpoint 的独立 rollout 结果。

**低成功率不是 Hard Gate。** 只有策略完全无法产生合法任务轨迹、环境接口错误或所有动作被错误裁剪时，才先修复基础实现。

### 2.3 快照恢复契约

完整快照至少包含：

- MuJoCo `qpos/qvel/act/ctrl/time`、mocap、必要的 model/data 状态；
- 环境 episode step、任务对象状态和 RNG；
- ACT 当前动作块、source tick、chunk index、generation、未完成请求；
- 观测历史和时间戳；
- 已实际提交动作账本；
- residual policy 所需历史；
- wrapper、normalizer、collector RNG 状态。

先做 deterministic replay test：同一快照、同一动作和同一随机数状态重复运行应产生一致轨迹。数值容差必须来源于实测并写入 manifest，不得随结果调整。

### 2.4 分支语义测试

在 V00/V01 中，当前实际下发动作必须完全相同；V10/V11 同理。REFRESH 只在当前动作执行后发起。

推理延迟为 `τ` 时：

- 环境继续运行，不得暂停 MuJoCo；
- 等待期间按已登记的旧块/备用控制规则执行；
- 新块按 source tick 对齐，不能从第 0 项重新播放；
- 有未完成请求时按统一规则处理重复请求；
- Keep 与 Refresh 的差异只能来自是否额外请求新块。

这些是 Hard Gate，因为错误会使四分支标签失去意义。

---

## 3. 核心 MVP：Square 上完整跑通 2×2 数据、离线学习和闭环调度

MVP 完成的定义是：**预注册实验全部有效执行、原始证据齐全、结论如实报告。** 不要求任何方法达到指定提升。

### 3.1 E0-A：自然分布集

目的：测量正常 ResFiT 执行中，residual 是否改变 KEEP/REFRESH 的局部最优决策及其发生频率。

计划：

- 使用固定 rollout seeds；
- 从正常 rollout 中无偏抽取 200 个 source snapshot groups；
- snapshot 选择不能依据 residual norm、成功/失败、未来回报或 privileged state；
- 每组采集 V00/V01/V10/V11；
- MVP 主延迟固定为 `τ=2` 控制步；
- 使用 policy 原本提出的 `δ_t`，不人为放大。

输出：

- `G0/Gδ/Iδ` 分布；
- 对一组预注册成本或 ACT-call 预算的 decision flip；
- task phase 只能用于离线分析，不能作为 Student 输入，除非该 phase 可由部署观测推导；
- source episode 聚类后的 bootstrap 区间；
- 每个分支的 control steps、ACT calls 和 wall-clock。

### 3.2 E0-B：机制富集集

目的：研究残差方向、幅度、chunk 位置等条件下交互项的结构，不用于估计自然发生率。

计划：

- 从与 E0-A 不重叠的 episode 中收集 200 个 snapshot groups；
- 仅用部署可见信号预定义富集规则，例如 residual norm、base/residual 夹角、chunk index、最近动作变化；
- 第一版仍采集 actor 原始 `δ_t` 的四分支；
- 扩展候选可加入 `0.5δ`、裁剪后的 `1.5δ`、任务相关平移维度的等幅方向变体；
- 所有候选经过相同动作约束，记录 proposed 与 applied residual。

E0-A 和 E0-B 分开报告，禁止用富集集计算自然事件发生率。

### 3.3 E0-C：小型离线 probe，提前运行强基线

在同一批配对分支数据上并行训练：

1. **Norm**：`F(h,B,||δ||)`；
2. **Linear-full**：完整 residual 的线性/低阶模型；
3. **B5-4branch**：输入完整 `δ` 直接预测 `Gδ`，并使用四分支数据中的 `δ=0` 与 `δ=δ_t` 标签；
4. **P1-4branch**：显式预测 `G0` 和 `Iδ`，相加得到 `Gδ`；
5. **B6-2branch**：只用 V10/V11 直接预测 `Gδ`。

B5 与 P1 必须满足：

- 完全相同的可见输入；
- 相同冻结视觉特征；
- 相近参数量和训练步数；
- 相同 source groups 和数据 split；
- B5 不得被故意削弱。

P1 的 interaction head 使用：

```text
I_hat(h,B,δ) = F(h,B,δ) - F(h,B,0)
```

以满足 `I_hat(0)=0`。

嵌套数据比例固定为：

```text
5%, 10%, 25%, 50%, 100%
```

同一 source episode 的所有 snapshot 和所有分支必须属于同一 split。

### 3.4 两条公平性比较

#### 同数据比较

```text
P1-4branch vs B5-4branch
```

回答 factorial decomposition 的归纳偏置是否有额外价值。

#### 同环境交互预算比较

```text
P1-4branch vs B6-2branch
```

预算必须以真实 branch rollouts 和环境 control steps 计算。

在相同 branch-rollout 预算 B 下：

```text
P1 可使用 floor(B/4) 个 snapshot contexts
B6 可使用 floor(B/2) 个 snapshot contexts
```

禁止只按训练样本行数声称数据效率。

### 3.5 指标

主离线指标：episode-grouped decision regret。

```text
oracle g* = argmax_g [V1g - c*g]
regret = max(V10, V11-c) - [V1,g_hat - c*g_hat]
```

Oracle 只是配对分支局部诊断，不是可部署算法，也不代表全局最优。

同时报告：

- `Gδ` MAE；
- `Iδ` MAE；
- critical decision error；
- calibration；
- 不同分支数据量下的 regret 曲线；
- 自然集与富集集分别的结果。

### 3.6 MVP 闭环验证

将已训练 scheduler 接入冻结 ACT + ResFiT，比较：

- 原版 ResFiT；
- 固定周期 refresh；
- residual norm threshold；
- B5 direct scheduler；
- B6 equal-budget direct scheduler；
- P1 factorial scheduler。

比较固定 ACT-call 预算下的 success/return，也比较固定任务表现下的 ACT calls 和 wall-clock。阈值只用 validation episodes 调整，测试集锁定。

**无论 P1 胜负，只要实验有效且证据齐全，MVP 即完成并进入下一节的扩展矩阵。**

---

## 4. 结果解释规则，不作为阶段门槛

| 观察结果 | 允许支持的结论 | 不允许的结论 |
|---|---|---|
| P1 优于 B5 且优于同预算 B6 | factorial supervision 在该设置下提高数据效率 | 已证明普适最优 |
| P1≈B5，但二者优于简单规则 | residual-aware scheduler 有价值；分解优势未得到支持 | factorial claim 成立 |
| P1 优于 B5，但不优于 B6 | 分解可能改善同 context 学习，但额外四分支成本未被证明值得 | 数据效率更高 |
| 自然 flip 低、富集集明显 | 机制存在但自然发生率有限 | 普遍需要 refresh |
| 所有方法接近 | 在该设置下复杂分解没有额外价值 | 整个 residual RL idea 无价值 |
| 结果方差大 | 当前证据不确定 | 任意方向的确定性结论 |

任何一行都不会自动删除下一阶段已登记的实验。

---

## 5. 终版扩展实现

### 5.1 X1：Square 能力边界

完整运行固定矩阵：

- refresh delay：0、2、4 控制步；
- 自然分布与富集分布；
- 固定多个 ACT/ResFiT checkpoint（按训练步数选，不按测试结果挑）；
- 原 residual、0.5 倍和裁剪后的 1.5 倍 residual；
- 至少 3 个独立训练 seeds，正式结果按资源扩展至 5 个；
- 多个 ACT-call budget 点形成 success-compute Pareto 曲线。

结果较差时照样完成矩阵并记录失效边界，不新增补救模块作为前置条件。

### 5.2 X2：Can 跨任务扩展

复用同一数据采集、模型输入、预算计数和评估代码，仅替换任务、数据和 checkpoint。

必须分别回答：

- 在 Can 内训练/测试是否重现 Square 结论；
- Square 训练的 scheduler 直接迁移到 Can 的结果；
- 少量 Can 分支数据微调是否改变数据效率曲线。

跨任务失败只是否定迁移 claim，不否定 Square 结论。

### 5.3 X3：在线终版——factorized joint value

实现两个端到端变体，二者都要运行：

#### Direct joint value

```text
Q_keep(h,B,δ)
Q_refresh_direct(h,B,δ)
```

#### Factorized joint value

```text
Q_keep(h,B,δ)
G0(h,B)
I(h,B,δ)
Q_refresh = Q_keep + G0 + I
```

调度：

```text
choose REFRESH when Q_refresh - cost > Q_keep
```

残差 actor 可优化：

```text
max(Q_keep, Q_refresh - cost)
```

实现要求：

- ACT 基座保持冻结；
- 实际在线 transition 更新 RL Critic；
- 2×2 分支数据只作为辅助监督；
- residual policy 更新后，旧 interaction 标签标注 policy/checkpoint version，不得假装永远有效；
- 记录 in-flight request、chunk source tick、实际 applied action；
- 对比 direct joint value 与 factorized joint value，而不是只与无 refresh 的弱基线比较；
- 所有额外分支 control steps 纳入数据预算。

X3 不需要 world model、base distillation 或 Agent。

### 5.4 不作为主线前置条件的扩展

以下内容只可进入 backlog，并建立独立实验，不得阻塞 X1-X3：

- selective suffix repair；
- world-model imagined branches；
- real-robot deployment；
- Agent/Memory/Event Observer；
- base-policy distillation；
- 动态残差上限。

---

## 6. 代码结构

目标结构：

```text
cf_resrl/
  factorial.py              # 已提供代数与预算契约
  snapshot.py               # 完整快照/恢复
  branch_runner.py          # V00/V01/V10/V11
  dataset.py                # manifest、group split、无泄漏检查
  features.py               # 仅部署可见输入
  models.py                 # Norm/Linear/B5/B6/P1
  scheduler.py              # KEEP/REFRESH 与请求状态机
  online.py                 # direct/factorized joint value
configs/cf_resrl/
  mvp_square.json
  square_*.yaml
  can_*.yaml
scripts/
  audit_dataset.py
  train_or_load_act.py
  train_or_load_resfit.py
  collect_factorial.py
  train_offline_probe.py
  eval_scheduler.py
  train_online_cf_resrl.py
  summarize_results.py
tests/
  test_factorial.py
  test_snapshot_restore.py
  test_branch_semantics.py
  test_no_leakage.py
  test_budget_accounting.py
results/cf_resrl/
  m0/
  mvp_square/
  expansion_square/
  expansion_can/
  online/
```

不要把占位文件、空 CSV 或随机生成曲线当成完成。

---

## 7. GitHub 与结果上传规则

目标仓库：

```text
https://github.com/Adeline123456789/residual-rl-sim
```

工作分支：

```text
research/cf-resrl-factorial-supervision
```

每个阶段独立 commit，并更新同一 PR。禁止 force-push 和覆盖旧结果。

每次真实运行必须上传或引用：

- 完整命令；
- resolved config；
- Git commit 和 dirty 状态；
- Python/CUDA/MuJoCo/robosuite 环境；
- 数据 revision 和 checkpoint checksum；
- seeds 与 source episode IDs；
- 每分支原始记录、每 episode 结果；
- branch rollouts、environment steps、ACT calls、wall-clock；
- stdout/stderr 日志；
- 汇总脚本生成的表和图；
- 失败与负结果。

大型 branch data 和 checkpoint 使用 Git LFS 或外部存储。普通 Git 中保留 manifest、URL、license、SHA256 和恢复命令。

资源阻塞时提交 `BLOCKED.md`，但已经有效的阶段不得因此改写为“未验证”。辅助模块阻塞不能阻止主线实验。

---

## 8. Codex 首轮执行顺序

1. 阅读本文件、`AGENTS.md`、旧 `CODEX_TASK.md`，明确旧路线仅作历史记录。
2. 运行现有 dependency-free tests。
3. 完成 Square 数据审计和环境 smoke test。
4. 完成 ACT/ResFiT checkpoint 获取或训练路径。
5. 实现并测试完整快照恢复。
6. 用 8 个 snapshot groups 跑四分支 smoke test，检查动作一致性和时间戳。
7. 采集 E0-A 与 E0-B 的预注册数据。
8. 同时训练 B5、B6、P1 和简单基线，生成同数据与同预算曲线。
9. 完成 MVP 闭环调度评估。
10. 不依据胜负改终点，按 X1 → X2 → X3 执行扩展并上传全部结果。

本文件中的样本数和矩阵是首轮预注册计划。若硬件资源要求统一缩小规模，必须对所有方法同比缩减、在运行前提交修改说明，不能根据中途结果选择性增减某个方法的预算。
