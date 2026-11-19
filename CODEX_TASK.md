# PA-ResWM：可执行研究任务书

版本：2026-11-19 / v1.0。目标仓库：`Adeline123456789/residual-rl-sim`。
研究分支：`research/pa-reswm-execution-consistency`。

**状态：这是实施规格，不是已完成的算法或实验。当前仅有执行契约参考代码和单元测试；ACT、ResFiT、世界模型、仿真评测均未运行。不得把单元测试报告写成机器人任务结果。**

## 1. 要解决的问题与贡献边界

保留 PA-ResWM 作为工程代号，但不要同时实现原设想的所有模块。研究问题收敛为：

> 在动作块异步执行、残差改变实际轨迹、且基座随后可能更新的条件下，能否利用真实执行历史进行短时状态预测和残差候选评估，并通过保持复合动作的再中心化，使探索中心渐进移动而不产生策略切换突变？

三个可检验假设：

- H1：在拥有相同传感器和历史信息的条件下，执行动作驱动的短时预测，比简单保留旧特征或历史堆叠更能缓解延迟退化。
- H2：在相同最大残差、相同动作限制下，对候选动作的后续风险和不确定性做短时检查，比仅根据当前 Q 值放大残差更稳健。
- H3：基座吸收已验证改进后，保持复合动作的残差再中心化，比“更新基座后保留旧残差”或“把残差归零”更少引起性能骤降，同时释放新的局部修正空间。

H1 是最小工程验证，不足以单独主张论文创新。H2/H3 是待验证的贡献候选。不得声称“首次将世界模型与残差 RL 结合”“完全避开全部已有论文”或保证发表。

### 1.1 对原方案的必要修正

ResFiT 本身已有逐步观测反馈；论文真机系统也已分离数据采集与 learner。该工程异步不等于本文研究的传感器/策略推理延迟。论文还已提出将复合策略蒸馏回基座作为后续方向。[R1]

“大残差使后续块全部失效”是待测假设，不是必然事实；delta/absolute 控制、执行前缀长度、重规划频率都会影响它。不得故意破坏基线的时间戳、队列或动作归一化来制造优势。

世界模型预测不是“当前真实状态”；模型内滚动不是真实反馈闭环。每次有新反馈应重新估计，默认只执行规划的第一步。

低 ensemble 方差不等于安全，高 Q 不等于无碰撞；LayerNorm 也不保证消除 Q 高估。硬件安全限制不能取消。第一阶段不放大残差，更不能取消全部输出限幅。

DiWA 使用离线 play interactions，不能描述成“只用成功示教”。AC3 的成功轨迹 actor/全数据 critic 非对称更新不能再作为新贡献。[R3,R4]

`Q(s,a_0)-Q(s,b_0)` 不是长度 H 的动作块优势；不得据此把整个 H 步片段认定为优质。失败动作也不应被统一手工赋成负 Q：仍按实际奖励和终止语义学习。

“失败数据只给 RL”与“失败轨迹的好前缀也给 BC”是不同实验设置。默认严格分流；前缀吸收只能作为明确标记的后续消融。

### 1.2 必须覆盖的相关工作

| 工作 | 已覆盖的机制 | 本项目不能声称的新意 |
|---|---|---|
| ResFiT | 冻结 BC、逐步残差、离线/在线混合、蒸馏展望 | 残差 RL、数据混合、笼统的基座蒸馏 |
| Policy Decorator | 模型无关残差、受控探索 | 给大策略加残差、逐渐放开探索 |
| DiWA | 离线世界模型中微调 diffusion policy | 用世界模型减少物理交互 |
| AC3 | 成功轨迹更新 actor，全数据更新 critic | 简单成功/失败双池 |
| RTC | 异步动作块执行、承诺前缀约束 | 异步执行本身 |
| A2C2 | 时间感知逐步动作块修正 | 加块内索引和轻量修正头 |
| REMAC | 块内执行失配、保留前缀的动作块机制 | 发现块内失配、保留已提交动作 |
| DreamActVLA | 旧观测加已承诺动作进行前瞻对齐 | 单独的世界模型延迟补偿 |
| DREAM-Chunk | 潜空间预测辅助块内反应式动作选择 | 轻量模型预测后续动作 |
| WorldSample | 真实轨迹附近的反事实生成、受控虚拟数据利用 | 失败复盘/反事实数据增强这个大想法 |
| ReEnvision | 前瞻性、选择性的剩余动作修复 | “只修坏掉的后续块”这个大想法 |

最近工作中有预印本/项目报告，不能一概声称均已同行评审。最终论文应以实际算法与实验建立差异，不靠更换缩写。

## 2. 仓库和上游实现

任务开始时目标仓库为空，不存在可直接继续训练的旧工程。不要复制另一个项目中的未审计 SAC/ACT 代码，也不要把规划当成已落地。

复用官方 ResFiT：`https://github.com/amazon-far/residual-offpolicy-rl`。
已检查的 upstream SHA：`66262b06a0caf159fb001504949b7b48ccda5f64`。

已确认存在的入口：

```
resfit/lerobot/scripts/train_bc_dexmg.py
resfit/lerobot/policies/act/modeling_act.py
resfit/lerobot/shell/launch_act_square.sh
resfit/lerobot/dataset/convert_robomimic_to_lerobot.py
resfit/rl_finetuning/scripts/train_residual_td3.py
resfit/rl_finetuning/config/residual_td3.py
```

官方 `ResidualTD3SquareConfig` 引用 `ankile/robomimic-mh-square-image`、300 条轨迹上限；`ResidualTD3CanConfig` 引用 `ankile/robomimic-mh-can-image`。配置中的 W&B checkpoint ID 不代表当前账户一定能下载，不得把访问权限当作前提。

将 upstream 固定 SHA 放在 `third_party/resfit` 子模块或保留来源的 vendor 目录。记录 `UPSTREAM.md`、依赖锁、许可证与 THIRD-PARTY 文件；代码头部标记了 CC-BY-NC-4.0，不能删除许可或把整个导入工程重标为 MIT。发布数据/权重前另查其许可。

优先增加 adapter 和小范围补丁，不重写全部 ACT/TD3。原始 ResFiT 复现与受控改造版分开命名；修改编码器、状态输入或任务设置后，不称为严格复现。

## 3. 时间、动作和 replay 的统一契约

定义控制 tick t：观察控制器可用信息，计算/提交动作，`env.step(a_t)` 产生 t+1 状态。严格区分：

- `capture_tick`：相机/传感器采样时间。
- `ready_tick`：编码或 ACT 推理完成、结果可用时间。
- `apply_tick`：动作实际传给环境/执行器接口的时间。
- `generation_id`、`episode_id`、`base_version`：拒绝乱序旧结果、跨回合结果、未发布版本。

动作块 `B_c = [b_{c|c}, ..., b_{c+H-1|c}]` 以源观测时间 c 为索引。结果在 t 到达时，取 `j=t-c` 对应动作，不得把第 0 项再次从 t 开始播放。不得使用 `ready_tick>t` 的结果。已执行动作不可修改；执行器 FIFO 中已承诺动作也不可被新规划覆盖。超期无动作时显式进入 fallback，不偷偷重用队尾。

视觉观测延迟、基座推理延迟、执行器命令延迟分别建模。MVP 执行器延迟设为 0；非零执行器延迟留到终版，并将 FIFO 纳入状态。最小测试中的 `ChunkQueue` 不是完整执行器延迟模拟器。

保存三种动作：`base_action`、`proposed_total_action`、`applied_action`。动力学和 critic 使用经过所有裁剪/限速/过滤后真正传给环境的完整动作；不能使用未通过过滤的提案或只用 residual。执行器接受的命令不等于机器人实际关节运动，两者分别记录。

归一化在整个研究周期内固定。保存 `applied_action_env` 和 `applied_action_normalized` 以及变换哈希。base+residual 必须在同一坐标系相加，再做一致的约束映射。动作单位、delta/absolute 模式、旋转表示、夹爪含义都要写入 manifest。

RL 控制上下文至少包含：最近视觉特征及时间戳、当前本体状态、chunk 后缀/索引/年龄、近期已执行动作、剩余回合时间、基座版本。模型 rollout 和离线 replay 都必须使用同一套队列语义。比较方法拥有相同可用信息，不把隐藏物体真值/未来图像给所提方法。

离线 demo 的完整动作就是 demonstration action，不是 residual。重建 base_action 时，按整条示教的观测历史和指定异步调度运行冻结 ACT；不要每个转移都偷偷查询一个即时、全新 ACT。

### 3.1 最小 replay schema

```
episode_id, tick, capture_tick, feature_ready_tick, base_ready_tick
raw_observation_or_reference, proprio, proprio_timestamp
chunk_id, chunk_source_tick, chunk_index, chunk_suffix, base_version
base_action, residual_proposal, proposed_total_action
applied_action_env, applied_action_normalized
reward, success, constraint_violation, terminated, truncated, final_observation
next_control_context, policy_version, model_version
```

优先保存原始观测或可恢复引用，避免把非平稳 latent 当永久事实。n-step 不跨 episode；auto-reset 的初始图像不是 terminal final observation。

## 4. M0：环境、数据和基线验收（必须先于长训练）

### 4.1 数据来源

先下载公开数据，不先写脚本重新“采集专家示教”。首选与 upstream 匹配的 Square MH 图像数据；不兼容时使用官方 robomimic 数据并显式转换、版本固定。Can 用于环境冒烟；Square 是首个主实验。PH 低维数据可做调试，但不替代 MH 图像主实验，也不混报结果。

官方 robomimic 文档说明不同 robosuite 数据版本可能导致结果变化；必须检查 metadata，不盲目升级到最新版。[R12]

下载不等于重新采集；把已有 simulator states 渲染为图像也不等于增加示教。记录两者区别。

### 4.2 开始 ACT 训练前执行

检查 trajectory 数、长度、obs/next_obs 对齐、图像解码、动作范围和单位、奖励/终止语义、相机方向、state/action 统计、NaN/Inf、训练/验证 trajectory ID 无交叉。normalizer 只拟合训练集。允许从总数据中留出验证，但报告实际训练条数，不能留出后仍称训练用了 300 条。

恢复至少 10 条轨迹的初始 simulator state，并在相同 env/controller 配置下重放动作。检查状态和成功信号；不能只看视频。若提供的转换数据无法恢复，取匹配原始 HDF5 或明确记录验证限制，不伪造通过。

先训练/加载小批样本，确认 BC loss 能下降且动作解码正确，再做正式 ACT。检查一次随机动作和零残差 rollout；零残差复合策略必须在相同调度下与 ACT-only 数值一致。

### 4.3 基座和 RL 配方

能取得可验证的 checkpoint 就加载，否则按官方脚本训练。所有对照共用同一个预训练 checkpoint。不得为提高新方法相对增益而故意选择弱基座。低于 10% 成功且几乎没有正奖励时，先排查数据/控制接口；高于 90% 时采用预先定义的延迟/扰动测试或更难任务，不人为削弱 checkpoint。

先保留 ResFiT 的 TD3 风格确定性 actor、Q ensemble、LayerNorm、target policy smoothing、离线/在线混合；不要悄悄换成 SAC 并引入熵系数这个额外变量。先导默认建议：UTD=4、actor 每 4 次 critic 更新一次、n-step=3、offline:online=1:1。其他值取解析后的 upstream Square 配置并完整存档。高 UTD 是多次使用 replay，不是只更新一回合的第一步。

用一小段训练确认 actor/critic 梯度、target 更新、学习次数、真实 env step 数相互一致。M0 不要求达到论文成功率；要求训练链条真实、可复现、没有明显接口错误。

## 5. M1：最小验证——只做执行历史驱动的状态对齐

**冻结 ACT；固定残差上限；m=1；不更新基座；不用虚拟 RL transition；不做失败前缀 BC；不做大幅探索。**

### 5.1 稳定表征

建立一个冻结的观察编码器 E。优先复用冻结 ACT 的视觉特征，必要时先训练投影再冻结；可选维数 128 或 256。不要把不断由 critic 更新的 ViT latent 缓存在数据集后，继续当固定监督目标。

受控对照中的 residual actor/critic 使用相同特征和输入结构；若使用冻结特征不同于官方默认，命名为 `ResFiT-controlled`，另保留官方未改版的基准记录。低维状态版本仅验证机制/接口，必须标成 privileged/state diagnostic，不能冒充图像部署结果。

### 5.2 世界模型

设 x_t=(z_t,p_t)，p 是本体状态；每个 ensemble 成员预测：

`x_hat_(t+1) = x_t + g_psi(x_t, a_t_applied, dt)`。

先用 3 个独立初始化、独立 bootstrap 数据采样的 3 层、宽 256 MLP。参数和层数是起点，不是性能保证。训练监督包括 h=1,2,4 的真实后继特征和本体状态；用 Huber/标准化 MSE，terminal mask 防跨回合。目标特征由冻结 E 提取。MVP 不做像素重建，也不需要奖励模型。

先用 demo 预训练，再用真实在线成功/失败转移更新。预热采集、专门世界模型采集和重复采集都计入环境交互预算。不得为新方法额外采很多数据却仍报告同预算。

不确定性先用 ensemble disagreement，再在独立验证轨迹上检查它与预测误差的关系。方差小只能叫模型分歧低，不能叫安全证明。

编码器 E 始终冻结；动力学 F 可以更新，但部署只发布带 model_version 的完整快照，第一版在回合边界切换。replay 保存观测和动作历史，训练批次里的当前/下一预测状态用同一 F 快照重新计算，或明确锁定版本；不能混用两个模型生成的 latent 缓存。M1 actor/critic 的梯度不反传到 F，避免仅为抬高 Q 而扭曲预测。

### 5.3 推理

收到 capture time c 的 z_c，在时间 t 查询账本中已执行的 `a_[c:t)`，逐步滚动得到 z_hat_t。当前可测的 p_t 直接来自传感器，不用预测 p_t 冒充测量；预测 p 与实测 p 的差可以作为诊断输入。新图像到达后重新校正，不持续无反馈漂移。

`delta_t = r * tanh(pi_res(z_hat_t, p_t, B_remaining, index, age, history))`

`a_t_applied = project(b_t + delta_t)`。

project 至少含环境动作范围/限速，所有对照同样使用。动作控制器内部的执行效果由模型学习，不能拿隐藏 simulator 下一状态给策略。

### 5.4 对照和预算

| ID | 方法 | 目的 |
|---|---|---|
| B0 | ResFiT-controlled，旧视觉特征+实时本体 | 建立受延迟的基线 |
| B1 | B0 + 动作历史、年龄和块内位置；参数量尽量匹配 | 排除仅因增加历史信息获益 |
| P1 | 与 B1 相同信息 + 执行动作驱动的模型前向对齐 | 检验 H1 |

B1 是针对 A2C2 等工作的时间感知强对照，不冒充完整 A2C2 官方复现。补充 persistence、线性外推预测误差对照；高频重新调用 ACT 仅做算力匹配对照或明确标成 oracle，不免费给额外推理次数。

第一轮：Square，3 个 RL seeds，每个方法 50k 环境步（含 warmup），共 450k 训练交互；这是筛查，不凭未收敛结果宣布失败。通过接口/方向性检查后，在同一配置上延长到每次 150k，总预算按累计 150k 统计，不把早期 50k 重复计算。最终效果需看训练曲线；未收敛只能报告当前预算结果。

控制频率先用 20Hz。新训练 ACT 可用预测 H=16、调用间隔 K=8；已有 checkpoint 则保留其有效训练配置，不能在不训练的情况下随意改动作块维度。所有方法固定相同 H/K/最大残差。

训练随机视觉延迟 d_obs∈{0,1,2,3}，基座推理延迟 d_base=2，执行器延迟 0。验证分别测试 `(d_obs,d_base)=(0,0),(2,2),(4,4)`，并报告实际观测年龄，不把二者相加之后又重复补偿。额外用抖动、丢帧评估时单独标为 OOD。延迟必须用不泄露未来的事件队列实现；不能只 `sleep()` 同时暂停物理世界来冒充异步控制。

所有方法在相同 delay-aware scheduler 上运行、从头训练时见到相同延迟分布。先导主残差半径可用归一化 r=0.1；0.2 做独立的幅度诊断/后续对照，报告实际物理单位。不可把单次调参后选中最有利的幅度包装成预注册设置。

### 5.5 指标与通过条件

每 10k 步做 20 个 validation episodes；最终用独立、锁定的 100 个初始场景/seed/条件评估。学习率/门限/早停不看最终测试。

主指标：相同场景下的任务成功率、环境步学习曲线 AUC、任务约束违反率。附加：多步预测误差、残差 RMS/饱和率、动作变化率、fresh-frame 纠偏误差、每回合 ACT 调用次数、端到端 P50/P95 推理耗时、deadline miss rate。

初筛目标（人为设置的继续研究门槛，不是预期结果）：预测标准化误差相比 persistence/线性外推中最好的一个下降至少 20%；延迟条件下 P1 比 B1 平均提升约 10 个百分点，零延迟下降不超过 5 个百分点，且没有明显增加约束违反。3 seeds 只做初筛；正式结论用至少 5 seeds，并对训练 seed 和 episode 做分层/配对 bootstrap 95% CI。CI 跨 0 时报告证据不足，不宣布有效。

若模型预测好但成功率不改善，说明表征或控制利用不足；若 B1 与 P1 相同，说明历史基线已足够；若只在人为极端延迟下有效，限制贡献范围。不得直接堆下一个模块掩盖失败。

## 6. M2：终版第一扩展——短时残差候选评估

M1 通过后才加入。目的不是让模型凭空生成“高价值样本”，而是在当前真实控制信息下比较有限个残差候选，并检查候选对后续几步的影响。

从 actor 输出生成 N=8 起步的有限候选，包含 residual=0、原 actor、小幅缩放和有界局部扰动。m=3 起步，最大先不超过 5；每个候选滚动全部 ensemble 成员，未来步骤用同一 residual actor 在预测状态及对应 base suffix 上闭环计算。此处“闭环”仅指模型内部，不是提前获得真实反馈。

注意：当前训练的 critic 仍评估完整第一步动作的 Q，不假装它直接评估整个动作块。候选效用可定义为：

`J(delta) = Q_LCB(context_t, b_t+delta) - lambda_u * max_j U_j - lambda_c * sum_j C_UCB_j - lambda_s * action_change_penalty`。

其中 Q_LCB 由 critic ensemble 给出；C 是单独从真实约束事件训练/校准的代价头，U 是动力学分歧。模型负责检查短时后果，不重复把 Q 当即时奖励逐步累加。没有可信 cost labels 时，只做模型误差/动作硬约束版并明确不能宣称接触安全。需要真正的多步 reward-return 规划时，另行增加并验证 reward/termination head，不从一张 Q 图推导不存在的奖励标签。

先过滤超出固定硬动作范围、运动学/碰撞约束和验证门限的候选，再从可接受集合中选 J。集合为空时执行经定义的低幅控制或安全停机/重规划。`alpha=0`/原基座动作不自动等于安全。

只执行第一步；每个 tick 使用最新可用信息重新评估。缓存中的后续修正仅用于带时效与版本标记的 fallback；有新观测、队列变化、模型超差、过期就失效。m 不得超过已知有效 base suffix；不能对尚未完成的 ACT 推理结果做有信息泄露的 rollout。

放宽幅度到 r_max=0.2 时，必须同时比较固定 r=0.2 的 ResFiT/B1，以及不使用模型的 Q-gate。所有方法使用相同硬安全层。不得比较“我们可用 0.2、基线只能用 0.1”后把收益归于预测。

训练仍以真实 replay 为主。M2 可以只把规划当行为策略改进，TD target 保持 residual actor continuation；此时 critic 的含义是 actor continuation 的价值，不是完整 MPC 部署策略的精确价值。若改成 planner target，则需单独实验并记录开销，不能混用语义。

## 7. M3：终版第二扩展——行为保持的基座更新

这是移动探索中心的主要扩展。每个 RL 阶段内基座冻结，阶段边界再更新；不是 base、residual、encoder、world model 同时无约束在线漂移。

### 7.1 数据分流

`D_real` 保存全部成功/失败转移，训练 residual actor、critic、world model。MVP 沿用 offline:online=1:1，不额外人为扩大失败比例。失败片段可用于代价模型的分层采样，但报告/必要时修正采样权重。

`D_distill` 默认仅含真实执行的成功且无约束违反片段，加原始示教防遗忘。保留阶段内固定 teacher 的记录、真实观测、真实执行动作和有效长度 mask。不要把 m=3 的好片段填充成未经验证的 H=16 完整标签。

成功-only 数据回流本身不是贡献，它是蒸馏对照。前期不要为避开 AC3 而强行加入更复杂路由。

可选失败前缀实验：只有在多步评估支持时吸收好前缀；与严格成功-only 分开跑。在仿真里可恢复同一快照，对 teacher prefix 与 base prefix 执行配对分支，之后采用相同 continuation，比较实际回报与约束。此类分支也是 simulator 交互，计入单独且公平的 selector/训练预算。不得免费用测试集、测试快照或隐藏状态训练部署策略。单步 Delta-Q 不足以批准整个片段。没有可验证依据就不吸收失败前缀。

### 7.2 不要“蒸馏后残差直接归零”

旧策略为 `a_old = b_old + delta_old`。新基座已吸收部分修正时，继续使用 delta_old 会重复修正；直接归零又会抹掉新基座尚未学会的部分。

在同一控制历史和相同动作坐标系上建立标签：

`delta_new_target(history_t) = a_teacher(history_t) - b_new_aligned(history_t)`。

于是训练目标是 `b_new + delta_new ≈ a_teacher`，而不是要求残差一律为 0。这是代数层面的保持目标；神经网络拟合和分布变化意味着实际系统仍需验证，不能宣称全状态精确保性能。

基座拟合真实执行的成功动作块；残差再中心化使用更广的 anchor histories，覆盖失败、延迟、边界状态，以免只保留容易状态的行为。用于 policy distillation 的 teacher 输出只是动作标签，绝不能伪造对应的环境转移/成功奖励。超出新 residual radius 的标签不能偷偷 clip 后仍声称行为保持：减小基座更新、重新拟合或拒绝发布。

### 7.3 发布与 replay 一致性

建议先每 25k 环境步评估一次 candidate，阶段内 teacher 固定；不要求每个候选必须上线。用独立 validation scenes 验收，最终 test 不参加发布。

比较三种操作：新 base + 旧 residual；新 base + 零 residual；新 base + 再中心化 residual。相同蒸馏数据、训练预算和 base checkpoint，测发布前后成功率、动作跳变、饱和率。

工程起始门槛：anchor 上 normalized action RMS 偏差≤0.02，至少 99% 的再中心化标签在 residual bounds 内；validation 成功率不明显下降、约束违反不增加。它们只是筛查阈值，需报告样本量和 CI；任何安全相关硬限制不由统计阈值替代。

只在回合边界原子发布 `(base_version,residual_version,normalizer_hash,model_version)`。清空旧 ACT 未执行队列并 reset recurrent/controller state；不得在一个 chunk 中途偷偷换 base。

物理世界模型继续使用全部真实完整动作，不因基座版本改变而改写历史 action。对依赖 base suffix 的 actor/critic replay，必须按新基座重新播放历史观测和已知时序，重建新上下文与 next-context，或限制在版本内训练。不能仅把旧 base_action 字段换成新 ACT 当前一步输出，其他队列信息却保留旧版本。

跨版本 replay relabel 完成之前暂停跨版本训练，不随意把 old residual label 当成 new residual。与 base 无关的物理模型数据仍保留，失败数据不丢弃。

### 7.4 要证明的指标

除成功率外，报告局部残差范数下降、固定 anchor histories 上相对最初 base 的完整动作偏移、发布瞬间动作差异，以及随后固定预算内进一步学习的增益。只有“基座单独 BC loss 下降”不能说明探索中心有效前移。

## 8. M4：可选反事实学习与真机（不是终版的强制堆料）

默认关闭虚拟高价值 transition。先把失败节点附近的模型 rollout 用作候选提案/诊断，真实执行验证后才进入真实 replay。

仅在 held-out 长度 1~3 的预测、reward、termination、cost 和不确定性校准均过关后，增加单独 `D_model`，使用保守回报/代价、明确 synthetic provenance、低比例（起始不超过 critic batch 的 10%）、短 rollout，并比较完全不用虚拟数据的版本。数据不足时宁可不启用。不能把模型自认为成功的动作直接当成功 demo。

“失败一次就学会大幅纠偏”“绝对安全”“不足 1ms”都不是设计所能保证的结果。

真机前先做无执行 shadow mode、传感器时间戳/force 校准、动作幅值/速度/工作空间/碰撞限制、超时停机、人工急停、回退策略和模型误差监控。受控实验仍先在仿真。任务成功和物理安全分开测量；低预测方差不能代替硬件安全系统。

## 9. 终版实验与评价语义

主任务 Square，泛化任务 Can，算力允许再上 TwoArmBoxCleanup；不要第一轮就 DexMimicGen 多任务全家桶。终版至少 2 个任务、5 个 RL seeds；明确 BC seed 是否固定，另测不同 BC checkpoint 的敏感性。

必须保留的对照：ACT-only、ResFiT 原版/受控版、B1 history-aware、P1 alignment、P1+短时评估、后者+naive distill、后者+行为保持再中心化。需要比较完整 A2C2/REMAC/其他新工作时按可用代码与适配范围单列；只移植一个机制就标 `-style adaptation`，不声称官方复现。

重点消融：真实执行动作 vs 计划动作的预测输入；m=1/3/5；固定半径 vs 相同最大半径的模型评估；success-only distillation vs 经过验证的失败前缀；旧残差/清零/再中心化；有/无 synthetic data。仅在前项有效后扩展，不把所有因素做巨大笛卡尔积。

重点压力测试是 residual magnitude × actual observation age × chunk index；已知机器人命令变化与未知外界扰动分开。未知扰动不允许从模拟器内部直接告知所提模型。记录已执行命令无法预测的外界变化这一局限。

### 成功与终止

主协议在首次 env success 时给单次 r=1 并终止，否则 r=0；任务定义的最大时长结束按失败处理，有限时域包含 time-remaining 输入，且该任务 timeout 不 bootstrap。所有组一致。普通采集切片造成的 artificial truncation 可 bootstrap，但必须使用真实 final observation。既不把 timeout 都当成功，也不把所有截断都混成物理终止。

主协议下 `success_any` 与停止时的成功标志一致。需要“最后仍保持成功”的稳定性评估时，另设成功后继续执行固定步数的 probe，并报告 `success_end/maintained_success`，不与主协议混算。复用未早停的环境时必须同时记录 any/end。

### 统计与预算

相同初始场景、相同扰动种子、相同执行/感知延迟轨迹做配对测试。统计训练 seed 之间的变异，不把同一策略的大量 episodes 当作独立训练重复。最终报告每个 seed 的数值、均值、95% CI、配对差异、AUC 和样本数。

总预算分别记录 demo transitions、online train env steps、额外世界模型/分支采集、validation steps、test steps、gradient steps、GPU hours、端到端 latency。固定环境步和固定 wall-clock 两个角度均报告。N 候选 × ensemble × horizon 的开销必须算入控制预算。

## 10. 工程模块与执行顺序

以下为待实现接口，不是现成可执行脚本：

```
third_party/resfit/                    # pinned upstream
pa_reswm/contracts.py                 # 已有：小型契约参考，不是完整运行时
pa_reswm/envs/timestamped_env.py       # 待实现：感知/推理/执行延迟分离
pa_reswm/data/audit.py                 # 待实现：先于 BC 的数据验收
pa_reswm/data/versioned_replay.py      # 待实现：完整动作与版本上下文
pa_reswm/models/latent_dynamics.py     # 待实现：稳定 latent ensemble
pa_reswm/control/state_aligner.py      # 待实现：真实动作历史滚动
pa_reswm/control/residual_evaluator.py # 待实现：M2 有限候选检查
pa_reswm/training/resfit_adapter.py    # 待实现：复用官方 TD3
pa_reswm/training/recenter.py          # 待实现：M3 蒸馏、再中心化、发布
scripts/preflight.py
scripts/train_base.py
scripts/train_residual.py
scripts/train_world_model.py
scripts/evaluate.py
scripts/aggregate.py
scripts/run_stage.py
```

Codex 实现以下统一命令并将其映射到官方入口；在脚本存在且执行过之前，不在 README 标“已验证可运行”：

```
python -m unittest discover -s tests -v
python scripts/preflight.py --task square --config configs/mvp.yaml
python scripts/train_base.py --config configs/mvp.yaml
python scripts/run_stage.py --stage M0 --config configs/mvp.yaml
python scripts/run_stage.py --stage M1 --config configs/mvp.yaml
python scripts/run_stage.py --stage M2 --config configs/full.yaml
python scripts/run_stage.py --stage M3 --config configs/full.yaml
python scripts/aggregate.py --results results --out reports/latest
```

每个阶段应有 dry-run、smoke、train、eval、report，自动依赖前一阶段的验收报告。不要让 Codex只搭目录就声称完成；也不要在没有 MuJoCo/GPU/数据权限时产生假 CSV。

目前契约测试覆盖：完整 applied action 前推、零延迟恒等、未来信息泄露、跨 episode、缺失/重复 action、过期 chunk prefix、未完成 inference、已承诺前缀、乱序结果、base version、再中心化可行性、n-step discount、terminated/truncated 区别、NaN。后续需补数据解码、replay relabel、模型训练和完整环境集成测试。

## 11. GitHub 提交与结果交付（强制）

所有代码、配置、实验结果和失败诊断提交到目标仓库的研究分支，再发 PR；不 force-push、不删除负结果、不覆盖用户已有代码。bootstrap 已初始化 main，后续可在研究分支继续。

每个 run 目录至少包含：

```
results/<run_id>/
  manifest.json             # code SHA/upstream SHA/dataset hash/seeds/versions
  config_resolved.yaml
  train_metrics.csv
  eval_episodes.csv         # 每个 episode 原始结果，不只有均值
  aggregate.json
  environment.txt          # GPU/driver/Python/package versions
  stdout.log
  checkpoint_manifest.json # 外部权重地址、SHA256、许可、恢复命令
reports/<stage>/
  SUMMARY.md
  learning_curve.png
  failure_analysis.md
```

manifest 的 `status` 只能由真实执行产生：`NOT_RUN / BLOCKED / RUNNING / FAILED / COMPLETED`；另有 `experiment_kind=unit_test|smoke|training|evaluation`。只有产生了实际训练和独立评估产物，才能将研究阶段标为完成。

每个 checkpoint 必须关联训练 code SHA、配置哈希和数据哈希；最好训练前先提交代码，避免用结果提交的 SHA 冒充训练代码 SHA。记录 dirty-tree diff hash。大数据和大权重不直接塞入普通 Git；使用获许可的 release/对象存储等，提交校验与下载脚本；不要把云端 API token 或个人数据上传。

结果报告包括成功和失败；统计脚本自动生成图表和表格，不手工编辑“更好看”的数值。README 只展示已经存在的结果。GPU、renderer 或数据访问阻塞时提交 `BLOCKED.md`：完整错误、已完成测试、复现命令与缺失资源，明确尚无 RL 性能结论。

提交节奏建议：chore/upstream+lock → test/data+timing → feat/baseline → exp/M0 → feat/alignment → exp/M1 → feat/lookahead → exp/M2 → feat/recenter → exp/M3。各阶段 PR 说明环境步、seed 数、结论、局限、下一验收条件。

## 12. 完成定义

最小方案完成：数据/环境先验收，ACT 与受控 ResFiT 真正训练/加载并评估，B0/B1/P1 在 Square 有多 seed 原始结果、可重画统计与可恢复 checkpoint，代码和结果都已推送。21 个契约单测通过不满足这个定义。

终版完成：M2/M3 的收益与失败模式经独立消融检验，至少两个任务与五个 RL seeds，处理真实时序、版本、残差预算和 replay 语义，能用一条已验证命令复现；没有收益时诚实提交负结果。反事实想象和真机不是必须加入的装饰性模块。

## 13. 核验过的来源

[R1] Ankile et al., ResFiT, 2025。用户上传的 v2，第 III 节算法、第 IV-B 节真实系统、第 VI 节限制与蒸馏展望。https://arxiv.org/abs/2509.19301

[R2] Policy Decorator, ICLR 2025。https://policydecorator.github.io/

[R3] DiWA, CoRL 2025。https://proceedings.mlr.press/v305/chandra25a.html

[R4] AC3, AAAI 2026。https://ojs.aaai.org/index.php/AAAI/article/view/38937 ; https://arxiv.org/html/2508.11143v2

[R5] RTC, NeurIPS 2025。https://proceedings.neurips.cc/paper_files/paper/2025/hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html

[R6] A2C2, 2025 预印本。https://arxiv.org/abs/2509.23224

[R7] REMAC, 页面注明 ICLR 2026。https://arxiv.org/abs/2601.20130

[R8] DreamActVLA, 2026-06-08 项目报告；不据此宣称同行评审接收。https://dream-act-vla.github.io/

[R9] DREAM-Chunk, 2026-06-17 预印本。https://arxiv.org/abs/2606.18589

[R10] WorldSample, 2026 预印本/作者项目页。https://xxreinsno.github.io/worldsample/ ; https://arxiv.org/abs/2607.02431

[R11] ReEnvision, 2026-10 预印本。https://arxiv.org/abs/2610.17977

[R12] robomimic 官方数据与版本文档。https://robomimic.github.io/docs/datasets/overview.html ; https://robomimic.github.io/docs/datasets/robomimic_v0.1.html

[R13] 已检查官方源码。https://github.com/amazon-far/residual-offpolicy-rl/tree/66262b06a0caf159fb001504949b7b48ccda5f64

检索截至 2026-11-19，不等于穷尽所有文献或取得创新性保证。
