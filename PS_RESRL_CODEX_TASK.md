# PS-ResRL：相位共生残差强化学习（创新点 1）面向 Codex 的实施任务书

> **固定研究问题（Frozen Research Question）**  
> **Can continuous chunk-phase modulation learned jointly with per-step residual RL (`PS-ResRL`) resolve residual-induced chunk drift in contact-rich manipulation with higher success-per-ACT-call efficiency than fixed-clock residual RL and discrete replanning schedulers?**  
> 对于基于动作块（Action Chunking）的机器人策略，将**连续动作块相位推进速度调制**（$\nu_t \in [0, \nu_{\max}]$）与逐帧残差策略（$\delta_t$）联合优化，能否在不增加基座策略推理调用次数（零额外 ACT/VLA 推理、零额外重规划延迟）的前提下，显著缓解残差修正引起的块内相位失配，并在“成功率–计算预算”前沿上优于固定时钟残差 RL（ResFiT）与离散重规划调度器（CF-ResRL）？

**执行前置条件**：本任务书在 `research/cf-resrl-factorial-supervision` 分支完成 **M0（ACT 50k + ResFiT 300k）与 CF-ResRL Square 最小验证（MVP）** 之后启动。**严禁**在当前正在运行的 ResFiT / CF-ResRL 任务未落盘前修改 `/media/midea/work/CF-ResRL/residual-rl-sim` 工作树。

---

## 0. 第一原则：隔离执行、不移动终点、结果不设硬门槛

每个 Codex 子任务开始前先确认一句话：

> **“这一步如何直接推进核心问题：联合学习连续块相位调制 $(\delta_t, \nu_t)$ 相比固定时钟 ResFiT 与离散重规划（CF-ResRL）的真实收益与能力边界？”**

必须严格遵守以下 6 条纪律：

1. **物理与分支隔离（Git Worktree）**：
   - 原分支 `research/cf-resrl-factorial-supervision` 及全部 `CF-ResRL` 结果作为只读历史与强基线，**严禁改写、覆盖或 force-push**。
   - 在独立 worktree（`/media/midea/work/CF-ResRL/residual-rl-sim-ps`）和新分支 `research/ps-resrl-phase-symbiotic` 上开发与运行，只读引用已完成的 M0 权重与 CF-ResRL 快照。
2. **研究问题与强基线锁定**：
   - 必须同时对比：**`B0`（ACT-only）**、**`B1`（固定时钟 ResFiT 7-D）**、**`B2`（解析 Frenet 相位制动启发式 `PS-Rule`）**、**`B3`（CF-ResRL MVP 最优离散调度器 `KEEP/REFRESH`）**、**`B4`（等 ACT 调用预算的固定高频重规划）** 与核心方法 **`P1`（8-D 相位共生残差 TD3，`PS-ResRL`）**。
   - 严禁只对比 `B0 (ACT)` 或弱基线就宣称有效；**最核心对比是 `P1 vs B1`（等 ACT 预算看成功率与残差饱和率）、`P1 vs B2`（学习式相位调制 vs 几何启发式）以及 `P1 vs B3`（零开销连续调相 vs 付费离散重规划）**。
3. **阶段推进只看实现与实验有效性，不看效应方向**：
   - 若 `PS-0` 第五分支恢复率低、`P1 ≈ B2` 或 `P1 ≈ B1`，均属于一等科学结论（负/零结果），**照常完成预注册矩阵并如实归档**，仅收窄对应 claim，不临时堆砌新模块掩盖负结果。
4. **辅助模块不得绑架主线**：
   - 世界模型（World Model）、基座 ACT 蒸馏更新、动态放大残差上限、Agent/Memory 均**不属于**本任务前置依赖。
   - 创新点 2（在途丢弃前缀漂移补偿 `Dead-Reckoned Prefix Alignment`）与混合调度（`PS-ResRL + CF-ResRL Refresh`）仅放在扩展阶段 `PX1`，不得阻塞核心 `PS-MVP` 和 `PS-2`。
5. **仅以下实验无效故障构成 Hard Gate（必须停机修复）**：
   - 零调制退化失败：当 $\delta_t = 0, \nu_t \equiv 1.0$ 时，新调度器/环境封装器输出与原 `AbsoluteTickScheduler` / `BasePolicyVecEnvWrapper` 不一致（误差 $> 10^{-7}$）；
   - 夹爪维度被错误连续插值（出现非法半开合中间态）；
   - 快照恢复后五分支轨迹不一致或首步干预动作不一致；
   - 策略或启发式规则使用了部署时不可见的 privileged object state；
   - 发生动作越界、无限锁相死锁（未触发 `max_chunk_wall_steps` 兜底）或 episode 级训练/测试数据泄漏；
   - 伪造 CSV、丢失日志、未记录 git commit / SHA256 校验值。
6. **奖励函数零篡改原则**：
   - 主实验 `P1` 必须使用与 `B1 (ResFiT)` **完全一致的原始稀疏二值任务奖励**（不加人工时间惩罚或相位平滑 shaping），保证所有收益纯粹来自连续块相位自由度本身。任何相位正则项仅可作为消融组单独报告。

---

## 1. 核心动机与文献差异边界

### 1.1 为什么“离散 Keep / Refresh”还不够？
在 Square 等精密接触装配任务中，ACT 预测的未来 $k$ 步动作块 $B = (b_0, \dots, b_{k-1})$ 隐含了**刚性时间进度假设**（例如第 8 步进入下压、第 16 步松开夹爪）。当存在几毫米横向偏差时：
- ResFiT 的残差 $\delta_t$ 会在法向做水平对准（$\delta_t^\perp$），同时在切向反向制动阻止过早下压（$\delta_t^\parallel$）；
- 但离散调度器仍然每步强制 `chunk_index += 1`，导致机械臂还在对准，动作块时钟已经推进到下压/松爪阶段——**此时几何轨迹形状本身并未作废，仅仅是物理进度落后于动作块内部时钟**；
- `CF-ResRL / BCP / ForkCast` 只能二选一：要么 `KEEP`（继续硬扛错位后缀），要么 `REFRESH`（丢弃整块、支付昂贵的 ACT/VLA 推理开销并承受 $\tau$ 步异步延迟）。

### 1.2 与 2025–2026 相关工作的明确区分

| 相关工作 | 已有机制 | `PS-ResRL` 的清晰边界 |
| :--- | :--- | :--- |
| **ResFiT (2025)** | 冻结动作块基座 + 逐帧 7-D 残差 Off-Policy TD3 | 将刚性 `index += 1` 升级为与残差联合优化的连续块相位推进 $\phi_{t+1} = \phi_t + \nu_t$ |
| **Pace (2606.00537) / AdaTempo (2610)** | 示教节奏/相位对齐，改变动作块播放倍速 | 无闭环残差修正，无法在挂起块时钟（$\nu_t \to 0$）的同时执行正交接触纠偏 $\delta_t^\perp$ |
| **Warp RL (2026)** | 用样条流扭曲单步动作分布几何 | 作用于单步动作空间，不调控动作块的时间索引与后缀相位漂移 |
| **ForkCast / BCP / EQRL / CF-ResRL** | 学习何时丢弃当前块并触发 `REFRESH` | 离散 `REFRESH` 需要额外基座调用并引入延迟 $\tau$；`PS-ResRL` 以 **0 额外 ACT 调用、0 延迟**在块内连续自愈，且可与 `REFRESH` 互补 |
| **CereVLA (2609.27468)** | 预测残差有害时抑制残差 | 抑制残差会放弃当前纠偏；`PS-ResRL` 保留当前纠偏并通过调慢块时钟使剩余后缀保持兼容 |
| **ReEnvision (2610.17977)** | 基于信念修正预测动作块局部修补 | 依赖离线动作差异监督而非任务回报 RL，且不控制连续时间相位 |

---

## 2. 复用已锁定的 M0 与 CF-ResRL 资产（零重复训练基座）

所有实验直接只读复用以下已审计资产（不得重新切分数据、不得更改上游依赖）：

```text
上游基线:
  amazon-far/residual-offpolicy-rl : 66262b06a0caf159fb001504949b7b48ccda5f64
  robosuite                        : 77a4751233c29456a5381209e30dd0dbf39a6557
  lerobot                          : 69901b9b6a2300914ca3de0ea14b6fa6e0203bd4
  MuJoCo                           : 3.3.2
  Conda 环境                       : cf-resrl-m0 (评估/数据), cf-resrl-resfit (RL 训练)

锁定主数据集 (Square-MH):
  repo_id              : ankile/robomimic-mh-square-image
  revision             : 4e5cee7ea7d63b920ff76c2529b781c2474697da
  local_root           : data/cf_resrl/square
  split_manifest       : results/cf_resrl/m0/dataset_audit/attempts/20261124T131023495428Z_a9dcb4b558af/split_manifest.json
  split_manifest_sha256: 629d01857156c691f297218e0cda3e3b3a802a91122f006a1ca70fd31163afcd
  split                : train=240 / validation=30 / test=30 episodes

锁定自训 ACT 基座 (50k 固定步数):
  local_dir            : results/cf_resrl/m0/act_training/attempts/20261125T005956570423Z_3dfd8208847a/training/policy_step_49999/policy
  config_sha256        : 4a6a588a50b1256560c3c74efb16c1dab1fc82ad477c5b2af97611d5e26273bb
  model_sha256         : e5f5b3c7fa7512a2ae202d5912922289a52348f8a3bb5a9da33f606d671cb859

锁定 ResFiT 300k 基线 & CF-ResRL MVP 产物:
  ResFiT checkpoint    : 读取 M0 已完成的 resfit_training 尝试目录中的 checkpoint_manifest.json（记录固定步数权重及 SHA256）
  CF-ResRL MVP 数据    : 读取已完成的 cf_square_factorial_v1（200 natural + 200 enriched source snapshots）与闭环 Pareto 结果
```

---

## 3. 核心数学定义与工程契约（Hard Validity Gates）

### 3.1 连续相位动作块读取（`ContinuousPhaseScheduler`）
设 ACT 一次推理返回的动作块为 $B = (b_0, b_1, \dots, b_{k-1}) \in \mathbb{R}^{k \times d}$，其中 $d = 7$（前 6 维为末端位姿增量 `pose_dims = 0:6`，第 7 维为夹爪指令 `gripper_dim = 6`），有效执行视界为 $H = n_{\mathrm{action\_steps}} \le k$。

调度器维护两个状态量：
- **连续相位指针** $\phi_t \in [0, H)$（新块激活时若无延迟初始化为 $\phi = 0.0$；若有异步延迟 $\tau$ 则按绝对时钟对齐初始化为 $\phi = \tau$，沿用 `AbsoluteTickScheduler` 语义）；
- **当前块墙钟步数** $w_t \in \mathbb{Z}_{\ge 0}$（每次 `after_step` 严格 `+1`，新块激活时归零）。

给定当前 $\phi_t$，定义混合插值基座动作 $\tilde{b}(\phi_t)$：
$$u = \lfloor \phi_t \rfloor, \qquad u' = \min(u + 1, k - 1), \qquad \alpha = \phi_t - u \in [0, 1)$$
$$\tilde{b}_{0:6}(\phi_t) = (1 - \alpha)\, b_{u, 0:6} + \alpha\, b_{u', 0:6}, \qquad \tilde{b}_{6}(\phi_t) = b_{\min(\lfloor \phi_t + 0.5 \rfloor,\, k-1),\, 6}$$

> **硬约束**：夹爪维度（`dim=6`）**禁止线性插值**，只允许最近邻离散采样；当 $\phi_t$ 为整数 $j$ 时，$\alpha = 0$，必须满足 $\tilde{b}(j) \equiv b_j$（比特级相等）。

### 3.2 相位演进与防死锁兜底（Anti-Stall Rule）
给定策略输出的相位推进速度 $\nu_t \in [\nu_{\min}, \nu_{\max}]$（默认 $\nu_{\min} = 0.0, \nu_{\max} = 2.0$）：
$$\phi_{t+1} = \phi_t + \nu_t, \qquad w_{t+1} = w_t + 1$$
当且仅当满足以下任一条件时，自然触发新 ACT 动作块查询（视为正常块耗尽，而非额外提前刷新）：
1. **相位耗尽**：$\phi_{t+1} \ge H$；
2. **防死锁墙钟上限（Stall Cap）**：$w_{t+1} \ge W_{\max}$，默认 $W_{\max} = 2H$（例如 $H = n_{\mathrm{action\_steps}}$ 时允许单块最多挂起减速至 2 倍时长）。若触发该兜底，账本必须记录 `triggered_stall_cap = True` 用于诊断统计。

### 3.3 Frenet 切向–法向残差几何分解（零特权信息）
在归一化动作空间（或物理增量空间，需在配置中固定）下，取当前基座三维平移向量 $v_b = \tilde{b}_{0:3}(\phi_t)$ 与残差三维平移向量 $v_\delta = \delta_{t, 0:3}$，定义：
$$\hat{\tau}_t = \frac{v_b}{\|v_b\|_2 + \epsilon}, \qquad \delta_t^{\parallel} = (v_\delta \cdot \hat{\tau}_t)\hat{\tau}_t, \qquad \delta_t^{\perp} = v_\delta - \delta_t^{\parallel}$$
$$\rho_t^{\mathrm{brake}} = \frac{v_\delta \cdot v_b}{\|v_b\|_2^2 + \epsilon_0}$$
其中 $\epsilon = 10^{-6}, \epsilon_0 = 10^{-3}$。物理含义：
- $\rho_t^{\mathrm{brake}} < 0$：残差在**对抗基座的前进方向**（如阻止提前下压）；
- $\|\delta_t^\perp\|_2$ 大：残差正在执行**横向对准纠偏**。

### 3.4 必须通过的 5 项单元测试契约（`tests/test_ps_resrl_contracts.py`）
1. **零调制比特等价测试**：对随机动作块序列，设定 $\delta_t \equiv 0, \nu_t \equiv 1.0$，断言 `ContinuousPhaseScheduler` 每一步的 `applied_action`、`act_calls`、新块切换时刻与 `AbsoluteTickScheduler` 误差严格为 `0.0`。
2. **夹爪离散保真测试**：对含 `{-1.0, +1.0}` 跳变的动作块，在任意非整数 $\phi_t \in [0, k-1]$ 下断言 $\tilde{b}_6(\phi_t) \in \{-1.0, +1.0\}$。
3. **防死锁测试**：连续输入 $\nu_t = 0.0$ 时，断言在恰好 $W_{\max}$ 步触发新块刷新并记录 `triggered_stall_cap = True`，不会死循环。
4. **复合快照确定性回放测试**：扩展 `CompositeSnapshot` 保存 `phase_cursor` 与 `chunk_wall_steps`，在真实 Square MuJoCo 环境中恢复同一快照两次并执行含非整数 $\nu_t \in \{0.25, 0.7, 1.4\}$ 的动作序列，断言 `qpos / qvel / ctrl / image_hash / applied_action` 差异为 `0.0`。
5. **零初始化退化测试（用于 PS-2 Actor）**：验证 8-D `PhaseSymbioticActor` 在 `actor_last_layer_init_scale = 0.0` 下，第 0 步确定性输出严格满足 $\delta_0 = 0$ 且 $\nu_0 = 1.0$。

---

## 4. 最小验证阶段（PS-MVP）：先证明“相位超前”是主要矛盾（~1–2 小时 GPU）

在投入 6–8 小时训练新 RL 策略之前，先用极低成本完成两步可证伪验证：

### 4.1 阶段 PS-0：复用 CF-ResRL 同源快照的“第五分支因果诊断”（~15 分钟）
直接读取 `CF-ResRL MVP` 已采集的 **200 组自然分布快照（`E0-A`）+ 200 组机制富集快照（`E0-B`）**，在不重训任何网络的情况下，从完全相同的 `CompositeSnapshot` 恢复状态，在原有四分支 `(V00, V01, V10-Keep, V11-Refresh)` 旁边追加两条**零额外 ACT 调用**的相位分支：

| 分支编号 | 首步动作 | 后续块时钟规则 | 额外 ACT 调用 | 诊断目的 |
| :--- | :--- | :--- | :---: | :--- |
| `V10-Keep`（已有） | $b_t + \delta_t$ | 固定 $\nu \equiv 1.0$ | `0` | 原版 ResFiT 基线 |
| `V11-Refresh`（已有） | $b_t + \delta_t$ | 触发异步 `REFRESH` ($\tau=2$) | `+1` | 付费重规划参照 |
| **`V10-PhaseHold`（新增）** | $b_t + \delta_t$ | 当 $\rho_t^{\mathrm{brake}} < 0$ 时置 $\nu_t = \operatorname{clip}(1 + \rho_t^{\mathrm{brake}}, 0, 1)$，持续 $w_{\mathrm{hold}} = 4$ 步后恢复 $\nu=1$ | **`0`** | 检验局部减速/锁相能否代替重规划 |
| **`V10-PhaseFreeze1`（新增）** | $b_t + \delta_t$ | 仅在干预步令 $\nu_t = 0.0$（挂起时钟 1 步），后续恢复 $\nu=1$ | **`0`** | 最小单步锁相消融 |

**必须输出的核心诊断指标（按 `E0-A` 与 `E0-B` 分开报告，并做 episode-level bootstrap 95% CI）**：
1. **相位可恢复率（Phase Recovery Rate）**：在所有“重规划救活失败样本”（`V10-Keep = 0` 且 `V11-Refresh = 1`）中，`V10-PhaseHold = 1` 的占比；
2. **零开销净价值增益**：$\Delta V_{\mathrm{phase}} = V_{10\text{-PhaseHold}} - V_{10\text{-Keep}}$，对比付费重规划净增益 $G_\delta - c = V_{11} - V_{10} - c$；
3. **残差饱和率下降**：`V10-PhaseHold` 后续 10 步内 $\|\delta_{t+j}\|_\infty \ge 0.95 r$ 的比例变化；
4. **切向制动相关性分析**：验证 `V10-PhaseHold > V10-Keep` 是否高度富集在 $\rho_t^{\mathrm{brake}} < 0$ 且 $\|\delta_t^\perp\|_2$ 较大的对准/接触阶段。

### 4.2 阶段 PS-1：零训练启发式闭环评测 `B2 (PS-Rule)`（~20 分钟）
将上述纯几何解析锁相规则：
$$\nu_t^{\mathrm{rule}} = \operatorname{clip}\!\left(1.0 + \kappa \cdot \min(0, \rho_t^{\mathrm{brake}}),\; \nu_{\min}^{\mathrm{rule}},\; 1.0\right)$$
（超参 $\kappa \in \{0.5, 1.0, 1.5\}$ 与 $\nu_{\min}^{\mathrm{rule}} \in \{0.0, 0.25\}$ **仅在 30 个 validation 初始化种子上标定，严禁在测试集上调参**）直接接入已训练好的 7-D ResFiT，在锁定的 100 个测试场景上与 `B0 (ACT)`, `B1 (ResFiT)`, `B3 (CF-ResRL MVP Best)`, `B4 (Fixed High-Freq Refresh)` 做全闭环对比，记录：
- `success_rate`（及 95% bootstrap CI）；
- `act_calls_per_episode`（每回合基座推理次数）；
- `episode_steps`（平均完成步数）；
- `residual_saturation_rate`（残差触达 $\pm 0.95 r$ 边界的帧比例）；
- `action_jerk_l2`（$\|a_t - a_{t-1}\|_2$ 均值，验证不引起机械臂抖动）；
- `wall_clock_ms_per_step`（真实端到端控制耗时）。

---

## 5. 核心学习阶段（PS-2）：8 维相位共生在线残差 TD3（`PS-ResRL`）

无论 `PS-0 / PS-1` 的启发式提升幅度大小，只要 `PS-MVP` 验证了接口与契约有效，即推进预注册的核心学习方法 **`P1: PS-ResRL`**，检验**端到端在线 RL 联合优化 $(\delta_t, \nu_t)$** 能否超越启发式 `B2 (PS-Rule)`、固定时钟 `B1 (ResFiT)` 与离散调度 `B3 (CF-ResRL)`。

### 5.1 最小手术式架构改造（基于上游 `resfit/rl_finetuning`）
为了保持与 `B1 (ResFiT)` 100% 公平可比，通过独立补丁文件 `patches/resfit/ps_resrl_td3.patch`（记录 SHA256，运行结束自动还原，严格沿用 [run_resfit_training.py](file:///media/midea/work/CF-ResRL/residual-rl-sim/scripts/run_resfit_training.py) 的干净工作树契约）做以下最小改造：

1. **环境封装器（`PhaseSymbioticBasePolicyVecEnvWrapper`，派生自 `BasePolicyVecEnvWrapper`）**：
   - 替换原有的 `self.base_policy.select_action(raw_obs)` 离散弹出逻辑：当某环境 $i$ 需要新块（初始、reset、$\phi_i \ge H$ 或 $w_i \ge W_{\max}$）时，调用 ACT 内部动作块预测接口缓存完整块 $B_i \in \mathbb{R}^{k \times 7}$，置 $\phi_i = 0, w_i = 0$；
   - 按 3.1 节混合插值计算 $\tilde{b}_i(\phi_i)$，并用 `action_scaler.scale` 得到 `base_naction`；
   - 新增部署可见低维特征拼接到 `observation.phase_context`（仅 9 维，不改动视觉编码器）：
     $$\text{phase\_context}_t = \Big[\frac{\phi_t}{H},\; \frac{w_t}{W_{\max}},\; \tilde{b}^{\mathrm{norm}}(\min(\phi_t+1, k-1)) - \tilde{b}^{\mathrm{norm}}(\phi_t)\Big] \in \mathbb{R}^9$$
     （即当前相位进度、墙钟占比 + 未来一步基座局部切向差分，使策略能“看见”下一拍 ACT 即将下压或松爪！）；
   - `step(action_8d)` 接收 8 维动作：
     - 前 7 维 `residual_naction = action_8d[:, :7]` 按 ResFiT 原逻辑与 `base_naction` 相加、裁剪、反归一化后送入 `vec_env.step(env_action)`；
     - 第 8 维 `phase_delta = action_8d[:, 7]` 映射为相位速度：
       $$\nu_t = \operatorname{clip}\!\left(1.0 + \text{phase\_delta},\; \nu_{\min},\; \nu_{\max}\right)$$
       更新 $\phi_{t+1} = \phi_t + \nu_t$；
     - 在 `info` 中精确记录 `proposed_phase_velocity`、`applied_phase_velocity`、`phase_cursor`、`triggered_stall_cap`、`act_call_occurred`。

2. **Actor 与 Critic（`action_dim = 7 + 1 = 8`）**：
   - **Actor 输出缩放分离**：前 7 维残差继续乘以 `action_scale = 0.1`（与 `B1` 完全一致）；第 8 维相位增量乘以 `phase_action_scale = 1.0`（使 $\Delta\nu \in [-1, 1] \Rightarrow \nu_t \in [0, 2]$），输出层保持 `actor_last_layer_init_scale = 0.0`，确保第 0 步严格以 $(\delta=0, \nu=1.0)$ 退化为标准 ACT。
   - **Critic 评估完整动作 + 相位速度**：Critic 接收 `(feat, state, phase_context, full_action_7d, phase_delta_1d)`。
   - **50/50 离线示教 Buffer 零篡改适配**：240 条 Square 示教轨迹中，每条轨迹天然对应 `base_action = ACT(o_t)`、`full_action = demo_action`（或原 ResFiT 离线标定逻辑），其相位动作维固定填入 `phase_delta = 0.0`（即 $\nu = 1.0$），同时填充真实的块内 `phase_context`。**零伪造标签，100% 兼容对称采样与 `n_step = 3`**。
   - **Warmup 阶段探索噪声**：前 10,000 步 warmup 中，前 7 维沿用 `random_action_noise_scale = 0.2`，第 8 维相位探索噪声默认采用 `phase_warmup_noise_scale = 0.5`（即 $\nu \sim \mathcal{U}(0.5, 1.5)$，避免初始阶段长时间卡死在 $\nu=0$），上线后 `stddev` 固定为 `0.05`（与 `B1` 一致，或做独立缩放配置记录）。

### 5.2 公平训练协议与对照组矩阵
- **总环境步预算**：严格与 `B1 (ResFiT)` 一致：**`10,000 warmup + 290,000 online = 300,000 env steps`**，每 10,000 步保存固定 checkpoint 并评估 100 回合，**禁止按测试集挑 best checkpoint**（统一对比固定步数 `100k / 200k / 300k` 曲线及最终 `300k` checkpoint）。
- **初筛与正式种子**：
  - 首轮筛查：固定 seed `20261124`（与 M0 ResFiT 同种子，实现最干净的同种子配对对比）；
  - 正式发表统计：扩展至 3 个（资源允许则 5 个）独立 RL seeds，报告均值与 95% CI。
- **核心消融（同 300k 预算）**：
  - `B1`：标准 ResFiT 7-D（$\nu \equiv 1.0$）；
  - `B1-ctx`：标准 ResFiT 7-D + 相同的 9 维 `phase_context` 输入（**关键消融！排除收益仅仅来自“多给了块内进度特征”而不是“主动调制 $\nu_t$”**）；
  - `P1-hold-only`：仅允许减速/挂起（$\nu_t \in [0, 1]$）；
  - `P1-full`：允许减速与快进（$\nu_t \in [0, 2]$）。

> **特别注意 `B1-ctx` 基线**：这是堵死审稿人质疑的杀手锏——只有 `P1-full > B1-ctx`，才能无可辩驳地证明收益来自**主动块时钟控制**，而不仅仅是给网络多看了当前 `chunk_index`！

---

## 6. 终版扩展矩阵（PX1 → PX2 → PX3）

完成 `PS-MVP` 与 `PS-2` 后，按预注册顺序推进扩展（结果无论正负均完整记录边界）：

### 6.1 PX1：异步延迟鲁棒性 + 在途丢弃前缀对齐（融合创新点 2）+ 混合重规划
在 Square 上测试异步基座延迟 $\tau \in \{0, 2, 4\}$ 控制步，重点评估两种低成本扩展：
1. **在途丢弃前缀漂移补偿（`Dead-Reckoned Prefix Alignment`, 创新点 2）**：
   - 当新块 $B^{\mathrm{new}}_{0:k}$ 经历 $\tau$ 步延迟到达时，不再白白丢弃 $B^{\mathrm{new}}_{0:\tau}$，而是利用账本中的实际执行动作计算零成本漂移向量：
     $$\Delta E_{t+\tau} = \sum_{j=0}^{\tau-1} \big(a_{t+j}^{\mathrm{applied}} - b_j^{\mathrm{new}}\big)$$
   - 测试将其作为特征喂入策略，以及一阶衰减位置回拉 $b'_{j} = b_j^{\mathrm{new}} - \eta^{j-\tau}\Delta E_{t+\tau}$ 在 $\tau=2, 4$ 下对“切换震荡（Refresh Shock）”的缓解效果。
2. **双层融合控制（`PS-ResRL + CF-ResRL Refresh`）**：
   - 内层由 `PS-ResRL` 通过 $(\delta_t, \nu_t)$ 以 0 推理成本处理常规对准与相位超前；
   - 外层叠加 `CF-ResRL` 的调度器仅在极端拓扑偏离时触发 `REFRESH`；
   - 绘制完整的 **`Success Rate vs. ACT Calls / Episode` Pareto 前沿曲线**，直接叠加展示 `ACT`, `ResFiT`, `Fixed-Freq`, `CF-ResRL (B5/B6/P1)`, `PS-Rule`, `PS-ResRL`, `PS-ResRL + CF-ResRL`。

### 6.2 PX2：跨任务泛化（Robomimic Can-MH）
使用 `ankile/robomimic-mh-can-image`（300 episodes, 20 Hz, 7-D），严格复用同一代码流程：
- 验证在非精密插接、偏高速抓取搬运的 Can 任务上，`PS-ResRL` 的快进区间（$\nu_t > 1$）能否在维持或提升成功率的同时**显著缩短回合执行步数（Time-to-Success）并降低 ACT 调用次数**，且不产生退化。

### 6.3 PX3（算力允许时的顶会加分项）：双臂 24-DoF 精密协作（DexMimicGen `BoxCleanup` 或 `Coffee`）
在 24-DoF 双臂任务中测试“单标量全局相位 $\nu_t$” vs “左右臂解耦双相位 $(\nu_t^{L}, \nu_t^{R})$”，验证双臂协同对准时的相位等待（一只手提前到位等待另一只手）能否成为高维真机场景的杀手级展示案例。

---

## 7. 目标代码结构与结果留证规范

在新 worktree `/media/midea/work/CF-ResRL/residual-rl-sim-ps`（分支 `research/ps-resrl-phase-symbiotic`）中新增以下模块，**不修改 `cf_resrl/` 已有核心契约的语义**：

```text
ps_resrl/
  __init__.py
  phase_scheduler.py         # ContinuousPhaseScheduler（混合插值、防死锁、零调制等价）
  frenet.py                  # 切向/法向分解与 PS-Rule 解析门控
  phase_snapshot.py          # 扩展 CompositeSnapshot 以无损保存/恢复 (phase_cursor, chunk_wall_steps)
  fifth_branch_runner.py     # 在 CF-ResRL 同源快照上运行 V10-PhaseHold / V10-PhaseFreeze1
  prefix_alignment.py        # PX1: 在途丢弃前缀漂移 ΔE_{t+τ} 计算与对齐
patches/resfit/
  ps_resrl_td3.patch         # 最小侵入式 8-D 相位共生封装器与 TD3 扩展补丁（记录 SHA256）
configs/ps_resrl/
  ps0_square_fifth_branch.json
  ps1_square_rule_eval.json
  ps2_square_td3_300k.json
  ps2_square_b1_ctx_300k.json
  px1_square_async_prefix.json
  px2_can_td3_300k.json
scripts/
  smoke_phase_scheduler.py
  run_ps0_fifth_branch.py
  eval_ps1_rule_closed_loop.py
  run_ps2_training.py        # 沿用 run_resfit_training.py 的不可变 attempt + 补丁自动清理规范
  summarize_ps_resrl.py      # 自动生成对比表、Pareto 图与相位速度分布图
tests/
  test_ps_resrl_contracts.py # 3.4 节全部 5 项硬契约单测
results/ps_resrl/
  ps0_fifth_branch/
  ps1_rule_closed_loop/
  ps2_training/
  px1_async_hybrid/
  px2_can/
```

每次真实实验落盘必须包含：`resolved_config.json`、`status.json`、`stdout_stderr.log`、git commit、上游子模块 commit、补丁 SHA256、权重校验值、逐回合/逐分支原始 CSV、汇总 JSON/图表；大权重不进普通 Git，保留路径/校验值/恢复命令。

---

## 8. 可直接粘贴给 Codex 的执行启动指令

> 等当前 `research/cf-resrl-factorial-supervision` 分支跑完 M0（ResFiT 300k）和 `CF-ResRL` Square MVP 并干净提交推送后，将下面整段发给 Codex 即可无缝接棒：

```text
请先确认 /media/midea/work/CF-ResRL/residual-rl-sim 当前分支 research/cf-resrl-factorial-supervision
已处于干净状态（git status 为空）且 M0 (ACT 50k + ResFiT 300k) 与 CF-ResRL Square MVP 已完成提交。

然后不要修改原目录或原分支，使用 git worktree 创建隔离工作区与新分支：
git -C /media/midea/work/CF-ResRL/residual-rl-sim worktree add \
  /media/midea/work/CF-ResRL/residual-rl-sim-ps \
  -b research/ps-resrl-phase-symbiotic

在新工作区 /media/midea/work/CF-ResRL/residual-rl-sim-ps 内，完整阅读
/media/midea/work/CF-ResRL/PS_RESRL_CODEX_TASK.md 并严格按以下顺序实施：

1. 实现 ps_resrl/phase_scheduler.py、ps_resrl/frenet.py、ps_resrl/phase_snapshot.py
   及 tests/test_ps_resrl_contracts.py，完成并记录全部 5 项硬契约测试（含真实 Square MuJoCo 零误差回放与零调制比特等价测试）。
2. 运行 PS-0（第五分支因果诊断）：只读复用 CF-ResRL MVP 的 200 natural + 200 enriched 快照，
   追加 V10-PhaseHold 与 V10-PhaseFreeze1，输出 phase_recovery_rate、净价值增益、残差饱和率及 episode-grouped 95% CI。
3. 运行 PS-1（零训练闭环启发式评测 B2: PS-Rule）：仅在 30 个 validation 场景选定 κ，
   在锁定测试集上与 ACT、ResFiT、CF-ResRL MVP 最优调度器和固定高频重规划做完整闭环对比。
4. 实现 patches/resfit/ps_resrl_td3.patch 与 scripts/run_ps2_training.py（沿用不可变 attempt 目录与运行后自动还原补丁规范），
   先完成 1-step / 2k-step 训练冒烟测试（验证零初始化退化、50/50 离线数据对齐、夹爪离散性和防死锁），
   再在固定 300k 步预算（同 seed 20261124）下完整训练：
   - B1-ctx（7-D ResFiT + 相同 9-D phase_context 消融）
   - P1-full（8-D 相位共生残差 TD3，ν ∈ [0, 2]，原始稀疏奖励零篡改）
   - P1-hold-only（ν ∈ [0, 1] 消融）
5. 无论结果正负，如实记录全部指标（成功率、ACT 调用数、回合步数、残差饱和率、动作平滑度、ν 阶段分布），
   按阶段提交并推送到新分支 research/ps-resrl-phase-symbiotic，随后按 PX1 → PX2 推进扩展矩阵。

纪律提醒：
- 禁止修改或 force-push research/cf-resrl-factorial-supervision 分支；
- 禁止伪造日志/CSV/曲线；
- 禁止在主方法 P1 中篡改稀疏任务奖励或向策略泄漏 privileged object state；
- 若出现 GPU/依赖/文件阻塞，写入 BLOCKED.md 并保留已完成阶段的真实证据。
```
