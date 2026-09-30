# 总待办清单（TODO）—— 按优先级

**文档职责**：这是**唯一**的"未完成任务"清单。已完成的事情在 `DONE_zh.md`，
每个缺陷/特性的来龙去脉在 `DEFECT_LOG_zh.md`。旧的分主题清单
（`high_level_todo.md` / `known_issues.md` / `history_low_level_policy_todo.md` /
`bad_orientation_analysis_zh.md` / `progress_summary_zh.md` / `todo_master_zh.md`）
已并入这两份文档，原文可在分支上查到（见文末"旧文档去哪了"）。

**维护约定**：见 `templates/DOC_TEMPLATE_zh.md`。每次更新在下面"更新记录"加一行；
条目用 `- [ ]`/`- [x]`；完成即迁到 `DONE_zh.md`。

## 更新记录

| 日期 | 更新内容 | 相关 commit / 分支 |
|---|---|---|
| 2026-09-20 | 初版：把 6 份旧清单合并成 TODO/DONE/DEFECT_LOG 三份；低层内容并入 `main`，P0 变成"高层链合并 + 导出流程固化" | `main @ 7ff5b86` |
| 2026-09-20 | **P0 清空**：高层链合并进 `main`（3 个 merge commit）+ 导出部署态策略固化（脚本默认目录/训练收尾提示/训练说明）；新增 DEF-018/019；P3"文档收尾"整条完成（旧 `docs/review/*.md` 已删 + 全部悬空引用改指新文档） | `codex/hl-merge-p0`（`129848e`/`af4602d`/`07601e9`/`30d5411`/`0772757`） |
| 2026-09-20 | 导出增加 **ONNX**（`--onnx/--opset`，含 onnxruntime 自检；DEF-020）；run `2026-09-20_00-50-31` 用最新 `model_19999.pt` 重新导出 | `708ca53` |
| 2026-09-20 | 新增**部署交接**：`docs/deploy_sim2sim_sim2real_zh.md` + `probe_deploy_layout.py`（DEF-021）；P1/P2/P3 待办不变 | `7458672` |
| 2026-09-20 | **部署基线固化**（`main @ 2d49f47` ↔ run `2026-09-20_00-50-31`，含产物 sha256 + tag `deploy-baseline-2026-09-20`，DONE 第六节 / DEF-022）；新增 `summarize_run.py`（P3 提前做，DEF-023）；**P1-1 改为"口径 + 配置"两条修法**、P1-2 补臂相关证据 | 基线 `2d49f47`；工具 `dc3a5b9`；文档 `eb22401` |
| 2026-09-20 | P1-1 进入实测：新增**探索噪声上界** `max_noise_std`（默认 0 = 不限制，投影梯度实现，DEF-024），并启动 4000-iter 的 cap=1.2 / entropy_coef=0.002 A/B（结果待回填） | `docs: P1-1 A/B` 提交 |
| 2026-09-29 | 新分支 `codex/ll-train-detail-fix`：**静止伫立专项**（DEF-026，两项惩罚 + 三条课程）、**镜像符号修复**（DEF-027，"右后腿往右前方撇"）、**扰动加强**（DEF-028）、**多地形任务**（DEF-029）、**遥操 history 任务**（DEF-030）；新增 P1-1'~P1-4 四条待办 | `codex/ll-train-detail-fix` |
| 2026-09-30 | **本机收尾批（DEF-034）**：回归矩阵脚本化（P1-4/P3 完成，11 OK / 1 SKIP / 0 FAIL）+ known_issues ⑧⑩⑪⑫⑱ + 三处工程债；11 条能跑的任务全部 EXIT=0 | `codex/ll-train-detail-fix` |
| 2026-09-30 | 第二批（按用户答复）：**cusrl 全删**（9 处注册字段 + setup 依赖）、**视觉编码器本地权重优先/默认不联网**（新增 `RL_TRAINING_ENCODER_DIR` / `RL_TRAINING_ALLOW_ENCODER_DOWNLOAD`）、**openvla 分支删除**（含 cfg）、**vr_extented 模块级 print 改成调试图**；sim2sim 因"已在另一个仓库实现"**移出待办**；P2 action term 收敛成一条带做法的待办 | `codex/ll-train-detail-fix` |

**优先级定义**：P0 = 挡在"能部署/能继续训练"前面；P1 = 决定训练质量上限；
P2 = 高层 replay 与工程债；P3 = 验证工具与文档。

---

## P0 —— 挡在主线前面

**已清空**（2026-09-20）。原先两条都已完成，见 `DONE_zh.md` 第五节：

- 高层 P0/P1 链合并 → 已并入 `main`（`codex/hl-merge-p0`：`129848e`/`af4602d`/`07601e9`），
  4 个高层任务 2 iter 全 EXIT=0，启动打印 obs 维度与 checkpoint 一致；
- "训练完必须导出部署态策略" → 已固化（`export_deploy_policy.py` 默认 `exported_deploy/` +
  训练收尾打印命令 + `docs/train_history_flat_zh.md` 补章节），验收实测自检 0.000e+00。

> 仍**不要合** `codex/docs-review`（旧清单来源，内容已并入三份新文档）。

---

## P1 —— 训练质量（决定上限）

- [x] **P1-1' 静止伫立专项：软化版权重的 A/B（两轮都跑完，软化版胜出）** → 结果见 DONE 第七节；
      剩余"全长 20k 定稿"另立 P1-1'''（下面）
  - 依据：DEFECT_LOG_zh.md DEF-026。基线（旧代码 20k）命令 (0,0,0) 时 `err_vel_xy = 0.148 m/s`、摔倒 0.133。
  - 已做：`stand_still_vel_l2` / `stand_still_wheel_vel_l2` 两项惩罚 + `rel_standing_envs` 0.02→0.15，
    三条 25k 步爬升课程（DEF-026 §3）。
  - **第一轮 A/B（权重 −8.0 / −0.01，1500 iter，与同代旧代码 `model_1500` 对比）已完成**：
    目标指标修好 —— 漂移**两档难度都降 22~23%**（s0 0.1153→0.0886；play 0.1475→0.1157）；
    但同代摔倒率 s0 从 0.178 涨到 **0.708**（终止几乎全是翻倒）⇒ 惩罚量级 = 总回报的 122%，太重。
    数字/复现命令见 `DONE_zh.md` 第七节，机理见 DEF-026 §4。
  - **软化版 run 也跑完并验收通过**（`2026-09-29_21-38-50_stand_still_soft`，1000 iter / seed 42）：
    同代三方对比（旧代码 / 第一版 −8/−0.01 / 软化版 −2/−5e-4）命令 (0,0,0)：
    `err_vel_xy` 0.1030 / 0.0860 / **0.0857**；摔倒率 0.2481 / 0.3401 / **0.1065**；
    回合长度 890.0 / 845.6 / **920.1**；回报 40.83 / 38.44 / **46.28** ⇒ **软化版三项全赢**。
  - 步态对称性（同代 1000 iter / 命令 1.0,0,0）：后腿左右不对称 −0.0702 / +0.0287 / **−0.0099 m**；
    前/后轮距 0.503/0.462 → 0.459/**0.458**；`hl~hr` 镜像 RMS（hipx/hipy/knee）
    0.252/0.154/0.522 → **0.172/0.115/0.325** ⇒ "右后腿往右前方撇"基本消失。
  - **唯一待观察**：0.5 / 1.0 m/s 两档的跟踪误差比旧代码略差（0.16~0.19 vs 0.12~0.15），
    1000 iter 还早，需全长 run 判断。
  - **下一步（P1-1'''）**：与 P1-1''（`max_noise_std=1.2`）一起跑**全长 20k**
    （4 台 autodl 3090 并行：① 软化版 20k ② 第一版/重惩罚 20k 对照
    ③ `Rough-Slopes-*` 多地形多点 ④ 分量消融/基线复核）。判据沿用上面的三方口径。

- [ ] **P1-1''' 全长 20k 定稿（**已上云开跑**，见 DEFECT_LOG_zh.md DEF-032 §4）**
  - 云端口径：4096 envs / seed 42 / 20k iter（与历史基线一致），阈值与判据沿用 DONE 第七节。
  - **已启动 ①**：`bbc64d91a6-99f1820e`（4UGPU 3090，ssh 10.60.144.11:1237）跑
    `History-Adaptation-Deeprobotics-M20-v0` + 软化版静止惩罚，3.2 s/iter ≈ 18 h，
    `--run_name cloud_soft20k`，日志 `/root/run_soft20k.log`。
  - **已启动 ②**：`686346b9c6-b16aa8d9`（planner 3090，ssh 10.60.144.11:291）跑
    `Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0`（**多地形，本机跑不了**），
    6.3 s/iter ≈ 35 h，`--run_name cloud_roughslopes20k`，日志 `/root/run_roughslopes.log`。
  - **中途验收（iter=10000）已完成并回填**（2026-09-30）：run 目录已 `scp` 回本机
    （`logs/rsl_rl/history_adaptation/2026-09-30_00-09-25_cloud_soft20k`），
    固定命令 eval 三档的 err_vel_xy 比同代旧代码好 −4%~−15%、步态"右后腿撇"基本消失；
    训练期 `root_height_below_minimum` 偏高（加强 push + 15% 站姿样本所致），
    详见 `DONE_zh.md` 第七节 §5。
  - **四条 run 已排进云端队列（2026-09-30，DEF-033）**：
    ① `cloud_soft20k`（跑着，今天 ~19:30 完）② `cloud_cap12_20k`（**排队**：等 ① 结束后自动开跑，
    验证 P1-1''）③ `cloud_roughslopes20k`（多地形，明天 ~12:30）④ `abl_pushonly_10k` +
    `abl_rewardonly_10k`（新实例，链式排队，分别 ~19:00 / 明天 ~02:40）。
  - 待做：① 每条跑完后 `scp` 回来（或用 Jupyter API 读日志/取 run）对比并定稿；
    ② 用 `probe_gait_symmetry.py` 复核云端策略的步态对称性；
    ③ 视情况用第三台（`c71a49a292`）跑 `max_noise_std=1.2` 的 20k 对照（P1-1''）；
    ④ **用完记得关机**（DEF-032 §5）。

- [ ] **P1-1'' 把 P1-1 的 `max_noise_std=1.2` 落成 cfg 默认值**
  - 现状：DEF-024 §4 已证明 cap=1.2 在 4000 iter 上全面更好，但 `rsl_rl_ppo_cfg.py` 的
    `RslRlPpoActorCriticHistoryCfg.max_noise_std` 仍是 0（不限制），只能命令行覆盖。
  - 依据（DEF-024 的遗留条件）：先在**全长 20k / 同 seed / 4096 envs** 上再验一次，
    通过后把默认值改成 1.2。
  - 验收：全长 run 的 `Policy/mean_noise_std` 平台 ≤1.2，`Train/mean_reward` 不劣于基线。
  - 注：本次 2000-iter 的静止专项 run 仍是 `max_noise_std=0`，与基线口径一致，便于对比。

- [ ] **P1-3 多地形任务的端到端验收（被 DEF-031 挡住）**
  - 现状：`Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0` 只做到 cfg 级验证；
    本机（Windows + A4000）跑 `terrain_type="generator"` 的任务会在 env 创建期死锁
    （DEF-031，原始代码同样复现）。
  - 要做什么：换到能跑生成地形的机器后 ① `--num_envs 64 --max_iterations 2` 冒烟；
    ② 短训（≥2000 iter）看地形通过率与 `root_height`/`bad_orientation_2` 合计摔倒；
    ③ 把数字回填 DEF-029。
  - 同批要补的还有 `Rough-*` / `Rough-WO-Stairs-*` 三个老任务 —— 它们在本机也从未跑过。

- [x] **P1-4 把"本机跑不了生成地形任务"钉进回归脚本** → **已完成（2026-09-30，DEF-034 §1）**
  - `scripts/reinforcement_learning/rsl_rl/smoke_regression.py`：逐任务独立进程 + 日志落盘 +
    **"日志 N 秒无增长 ⇒ 判 SKIP 并杀进程树"**（本机 `Rough-*` 稳定 SKIP，不再无限等待）
    + Markdown 汇总表 + 有 FAIL 才非零退出。实测 11 OK / 1 SKIP / 0 FAIL。


- [ ] **探明"探索噪声平台 ~1.5"（原"`noise_std`/`error_vel_xy` 长期退化"已归因，见 DEF-023）**
  - **已归因（2026-09-20，用 `summarize_run.py` 做阶段聚合）**：
    ① `Policy/mean_noise_std` **不是发散而是有界平台**：新 run 0.973/1.009/1.132/**1.473**（s0~s3），
    iter≈5000 起就在 1.43~1.49 抖动；旧 run（无课程）1.236→**1.505**，Δ末 = −0.035。
    机制：`log_std` 无上界 + `loss = surrogate + value_loss − entropy_coef*entropy`（`entropy_coef=0.01`）
    持续给熵正奖励，同时 `schedule="adaptive"`/`desired_kl=0.01` 把 `Loss/learning_rate` 压到地板
    （最低 1e-5）⇒ 高熵 + 低学习率，精度上界被压住（但 reward/ep_len 不降，训练没崩）。
    ② `Metrics/base_velocity/error_vel_xy` 的上升**与命令课程同形**（旧 run 0.15→0.77 同样升）
    ⇒ 是"命令范围放宽到 vx ±5 m/s"后的**口径产物**，不是策略退化：新 run 末 1000 的
    reward **23.6 vs 17.8**、ep_len **917 vs 768**、合计摔倒 **0.122 vs 0.331** 全面更好。
  - **A/B 已收尾（2026-09-22，数字与判定见 DEF-024 §4）**：开关 `max_noise_std` 已实现
    （默认 0 = 不限制）。**统一窗口**（iter 3125–3999，因为基线是 20k iter、阶段均值不可比）实测：
    - `max_noise_std=1.2`（run `2026-09-20_18-54-34_cap_noise_std`）**通过全部口径、建议作为默认**：
      噪声 1.405→**1.052**、reward 23.93→**38.52**、ep_len 858→**905**、s3 合计摔倒 0.194→**0.132**。
    - `entropy_coef=0.002`（run `2026-09-20_22-13-37_ent_coef_low`）也通过、但三项都略逊
      （reward 36.9、ep_len 878、摔倒 0.183）。
    - 意外点 `entropy_coef=0`（run `2026-09-20_19-30-43_ent_coef_low`）**只挂摔倒**（0.222 > 0.194）
      ⇒ **噪声不是越小越好**，存在中间最优区。
  - **剩余（本项未完全关闭）**：① `Loss/learning_rate` 在 s3 仍被 adaptive 调度压到 3e-5~2e-4
    （cap 甚至低于基线）⇒ "LR 地板"机制**未解**，候选：非 adaptive 调度 / 调 `desired_kl`；
    ② cap 的收益只在 4000 iter 上验过 ⇒ **部署前跑一次 20k 全长**（同 seed=42 / 4096 envs /
    `agent.policy.max_noise_std=1.2`）＋固定命令 eval，再改 cfg 默认值。
  - **自动化提醒（DEF-025）**：本环境桌面版 automation **不可用**（唤醒投递缺 `call_id` ⇒ 422
    且把线程写死），两条 automation 已 `PAUSED` ⇒ 巡检/收尾一律手动，命令见 DEF-024 §4 末尾。
  - 当时的修法（**已实现并实测，结论见上**）：① 给 `log_std` 加上界（`max_noise_std`/clamp，
    目标平台 ≤1.2）—— **采用**；② `entropy_coef` 0.01 → 0.005/0.002 —— **通过但未采用**（不如 ①）。落点：
    `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py:HistoryAdaptationPPORunnerCfg`
    （`policy.init_noise_std` / `noise_std_type="log"` / `algorithm.entropy_coef`）
    + `rsl_rl/rsl_rl/modules/actor_critic_history.py`（`log_std` 的取用处）。
    原候选"调 `body_*_rew_s3` 的 `num_steps` / 查 `track_lin_vel_xy_exp` 权重"**已降级**：
    实测 `Curriculum/body_pitch_rew_s3|body_roll_rew_s3` 在 iter≈3125 就到终值 0.8、
    `body_height_rew_s2` 在 s1 就到 0.8，与 `noise_std` 平台**不同期**，不构成解释。
  - 验收（**口径已改**）：(a) `noise_std` 平台 ≤1.2 且 `Loss/learning_rate` 不再长期贴地板（≥1e-4）；
    (b) 固定命令 eval（play/eval 探针，固定 vx/vy/yaw）下的 reward 与速度误差不劣于基线
    `2026-09-20_00-50-31`。跨阶段/跨 run 一律用 `summarize_run.py` 的**阶段均值**，不看单点。
  - 成本：`noise_std` 在 iter≈5000 已饱和 ⇒ A/B 可只跑 ~5k iter（比全量 20k 省 3/4），要更稳再补全量。

- [ ] **s3（完整任务）阶段的臂扰动鲁棒性**
  - 现象（2026-09-20 用阶段均值复核）：`root_height_below_minimum` s0/s1/s2 = 0.0258 / 0.0208 / 0.0275，
    **s3 = 0.1181**（末 1000 0.1151）；同期 `Metrics/ee_pose/orientation_error` 从 0.32 抬到 **0.85**
    （s3 才放开臂的大范围摆动），而 `height_error_bias_steady` 全程只有 1.3~1.8 cm
    ⇒ 剩余摔倒 = **臂摆动时倾覆**，不是高度控制失效。
  - 候选：① 收紧 EE 区间上界（`p_pitch` 上界 +36° ≈ 让臂举高、重心上移）；
    ② 延长 s1/s2 停留步数（现在各 25k）；③ 再评估 `root_height_below_minimum` 0.30 → 0.26。
  - 注意：**单独降阈值没用**（阈值反事实 0.30→0.26 只把 20s 触发率从 25.8% 降到 24.4%）；
    且 `bad_orientation_2` 在 s3 是**下降**的（0.0756→0.0144）⇒ 两个终止项必须看合计（s3 = 0.1325）。

- [ ] **低层 `known_issues` 剩余条目**（原编号）
  - [x] ① `joint_mirror` 用平方差做镜像惩罚（左右关节符号约定可能相反）
    → **2026-09-29 已修**：确认符号确实相反（对角对要求 θ_fl=θ_hr，真实关系是 −θ），
    新增 `joint_mirror_signed` 并扩到 4 对（含左右对），见 `DEFECT_LOG_zh.md` DEF-027；
  - [x] ⑧ `action_mirror`/`action_sync` 用 articulation 关节 id 索引动作向量 + 关节名不存在
    → **2026-09-30 删除**（DEF-034 §3：打开就炸的埋雷，且状态层 `joint_mirror_signed` 已覆盖）
  - [x] ⑩ `arm_rewards.py` 的 `grasp_success`/`ee_approach_object` 依赖不存在的 `object`
    → **2026-09-30 加明确报错**（DEF-034 §4：保留函数，但接错场景会给出可读提示）
  - [x] ⑪ `HeightInvariantEECommand._update_command` 覆盖父类没调 `super()`
    → **2026-09-30 实测澄清**（DEF-034 §2）：父类该方法本来 `pass`，`pose_command_w` 由
    `_update_metrics()` 每步更新（只比 `pose_command_b` 晚一拍），"永不更新"不成立
  - [x] ⑫ reset 后第一帧 `pose_command_b` 全 0（观测/奖励看到零位姿目标）
    → **2026-09-30 实测确认并修复**（DEF-034 §2）：加 `reset()` 覆写，修后与真实 EE 位姿差 0.000e+00
  - ⑬ `UniformThresholdVelocityCommand._resample_command` 的 `> 0.0` 是空操作；
  - ⑭ EE 目标碰撞检查静默降级 + 硬编码 AABB；⑮ 硬编码常数（0.513 / 0.09 / 0.135 / action scale）；
  - [x] ⑱ 接触传感器与 articulation body 顺序不同（归因陷阱）
    → **2026-09-30 加启动期打印 + 名字可归因检查**（DEF-034 §5）
  - ⑬ 复核结论（2026-09-30）：当前 `UniformThresholdVelocityCommand._resample_command` 用的是
    `> 0.1`（不是 `> 0.0`），语义正确 ⇒ **该条已过时**，下次清理时可直接删掉这一行。

- [ ] `[可选]` **执行器刚度课程 / 刚度标定**
  - 依据：`piper_arm` 现在 `DelayedPD(300/20)`，注释里的目标区间是 60~100 / 0~20；
    实测软臂（40/8）臂速 0.95 vs 硬臂（300/20）1.74 rad/s。
  - 做法：刚度做成课程（前段 40/8 → 后期 300/20），或终值降到 60~100 后重新标定。
    现在 `root_height` 已降到 0.12，优先级可往后放。

---

## P2 —— 高层 replay 与工程债（"高层修改先暂放"期间不动）

- [ ] **把剩下 3 个高层 action term 收进 `LowLevelPolicyActionBase`**（用户要求：尽量一个基类）
  - 现状：`PreTrainedNavAction` 已迁移（清单 ⑤ 的 R1）；**`PreTrainedPickAction`（29.9 KB）、
    `PreTrainedPickWBCAction`（28.7 KB）、`TeleopLLAction`（27 KB）仍各自抄了一份**
    "载入策略 / 三个低层 action term / 布局解析 / 低层观测组 / history 窗口 / 低层 tick 循环"
    （每个文件约 200 行重复代码；基类 `low_level_policy_action.py` 已经把这块抽好了）。
  - **openvla 已删除**（2026-09-30，用户确认废弃；DEF-034 §7）——`PreTrainedPolicyAction`
    没被任何 task 注册，可以顺手一起迁或删（**建议删**，与 openvla 同理）。
  - 做法（已验证可行的路线）：① 基类补一个 `_on_reset(env_ids)` 钩子（pick 用来
    `_reset_target_to_current_ee`、teleop 用来 `recalibrate`/`_reset_default_body_pose`）；
    ② 三个类改成 `class X(LowLevelPolicyActionBase)`，`__init__` 只留"分配 `_raw_actions`
    （必须在 `super().__init__` 之前）→ `super().__init__(cfg, env)` → 任务专属状态"；
    ③ 删掉各自的 `apply_actions`（基类已实现 tick 循环），需要额外动作的写进
    `_on_low_level_tick()`（teleop 的 `push_ee_target_to_ik`）；④ `process_actions` /
    properties / 调试可视化保持原样。
  - 验收：`smoke_regression.py` 覆盖到全部三个类（Pick-Flat / Pick-WBC-Flat / Teleop /
    Teleop-History 各 2 iter）+ 对比改动前后启动打印的"低层 obs 维度 == checkpoint 期望"。

- [ ] **高层 `high_level_todo.md` 第 10 条的"待确认"**
  - `HLFlatPickTerminationsCfg_PLAY` 里 `lift_object` 与 `pick_success` 语义/命名重复；
  - [x] `highlevel/mdp/encoder.py` 的 `_frozen_encoders` 以 name 为 key 全局缓存（换 device
    会拿到旧设备上的模型）→ **2026-09-30 已修**（key 改成 `(name, device)`，DEF-034 §5）；
    仍待办：`torch.hub.load("facebookresearch/dinov2", ...)` 需要联网（离线机器上会失败），
    建议改成"本地权重优先，缺失再联网"。

- [ ] **工程性清理**（原 `known_issues.md` 二、1-8）
  - `mdp/__init__.py` 星号导入造成同名遮蔽（`ROUGH_TERRAINS_CFG` /
    `randomize_rigid_body_inertia` / `randomize_com_positions` 覆盖官方实现）；
    **复核结论（2026-09-30）**：现在 `velocity_env_cfg.py` 显式
    `from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG`，遮蔽只影响
    `mdp.rough_terrains_cfg` 这类间接引用 ⇒ 影响面小，留待下次一起清；
  - [x] `setup.py` 的 `packages` 只列顶层包 → **2026-09-30 改成 `find_packages`**（DEF-034 §5）；
    **仍待办**：`cusrl_cfg_entry_point` 有 **9 处**指向不存在的 `agents/cusrl_ppo_cfg.py`
    （要决定"删注册字段"还是"补模块"，前者更干净）；
  - 依赖被本地魔改：`IsaacLab-5.1.0/.../task_space_actions.py`（`[IK DEBUG]` 打印 + 私有成员）；
  - `devices/vr_extented.py` 模块级 print（第 59~63 行）+ 无超时线程（线程是 `daemon=True`，
    但网络操作没有超时）；
  - [x] `scripts/utils/mp4-png-composition.py` 4 处裸 `except:` → **2026-09-30 改成
    `except Exception:`**（DEF-034 §5）；
  - `logs/` 每个 run 50 个 `model_*.pt`（~5 MB/个，注意磁盘）；
  - 中文/英文注释混排、大段注释掉的代码。

---

## P3 —— 验证工具与文档

- [x] **回归矩阵脚本化** → **2026-09-30 完成**（DEF-034 §1/§6）：
  `scripts/reinforcement_learning/rsl_rl/smoke_regression.py`，13 任务实测
  **11 OK / 1 SKIP（生成地形，本机，DEF-031）/ 0 FAIL**；带卡死检测、Markdown 汇总、
  FAIL 才非零退出。**仍待办**：把高层四条的低层 checkpoint 路径
  （`RL_TRAINING_LOW_LEVEL_POLICY_*`）做成脚本参数透传（现在走各自 cfg 默认值）。
  （矩阵本身 2026-09-20 曾手工整跑一遍：8/8 EXIT=0，见 `DONE_zh.md` 第六节）
  - 低层（main 已有）：`History-Adaptation-Deeprobotics-M20-v0`、
    `Flat-Deeprobotics-M20-Piper-WBC-v0`、`Flat-Deeprobotics-M20-Piper-v0`、
    `Flat-Deeprobotics-M20-Piper-Arm-v0` —— 各 `--headless --num_envs 64 --max_iterations 2`。
  - 高层（合并高层链后）：Pick-Flat / Pick-WBC-Flat / Teleop / Nav-Flat-Teacher 同上。
  - 还缺的：把 8 条命令收成一个脚本（顺序跑、逐条记 EXIT、失败即非零退出、日志落到
    `logs/smoke/<日期>_<task>.log`），省得每次手敲；高层四条要能透传低层 checkpoint 路径
    （`RL_TRAINING_LOW_LEVEL_POLICY_*`）。预估 0.5~1 h（跑一次 ~6 min）。

- [x] **训练曲线自动分析脚本**（`scripts/reinforcement_learning/rsl_rl/summarize_run.py`）—— 2026-09-20 完成，见 `DONE_zh.md` 第四节
  - 依据：现在每次手抠 tensorboard（57 MB events，读一次 30~60 s）。
  - 内容：输入 run 目录 → 输出"关键指标 × 迭代"表（终止构成、ep_len、reward、
    `error_vel_xy`、`noise_std`、`height_error_*`、课程权重），支持两 run 对比。
  - 实现：阶段均值表（按课程阶段 s0~s3，边界从 `params/agent.yaml` 的 `num_steps_per_env` 推）
    + 采样网格表 + 两 run 对比（`--derive` 可把终止项求和，如"合计摔倒"）；
    首次解析 70 MB 事件文件 30~50 s，之后走 `<run>/.summary_cache.npz`（<1 s）。
  - 已用它完成 DEF-023（P1-1 归因 + P1-2 新证据）。

- [x] ~~sim2sim(MuJoCo) 落地~~ → **不做**（2026-09-30 用户确认：sim2sim 已经在**另一个
  仓库**里实现了；本仓库只保留 DEF-021 的"接口契约 + 部署态导出 + 探针"，不再自己搭 MuJoCo 脚本）。

- [ ] **把"EE 锚点 4 组对照"固化成一键脚本**
  - **"EE 锚点"是什么**：训练早期（课程 s0）把机械臂的目标位姿**锁死在一个固定点**上，
    让机械臂先别动、底盘专心学平衡，之后再逐步放开到完整工作空间（课程 s1→s3）。
    这个"锁死的固定点"就叫锚点。仓库里有 3 种候选锚点：
    `none`（不锁，= 无课程）/ `default`（锁在**默认姿态**，即机械臂举起）/ `low`（锁在
    工作空间中心的**低位**前伸位姿）。DEF-006 实测过 20 s 内 `root_z<0.30` 的触发率：
    none **25.8%** / default **55.5%**（更差！因为举臂抬高重心）/ low **1.0%** ⇒ 所以最终
    选了 `low`。探针：`probe_root_height_termination.py --freeze_ee_preset {none,default,low}`。
  - 要做的：把这几组对照（512 envs × 20 s ≈ 2.5 min/组）收成一个脚本，一次跑完并输出对比表，
    以后改 EE 区间/锚点前先跑一遍。

- [x] **文档收尾**（2026-09-20 完成）
  - [x] 合并高层分支时删掉它们带来的旧 `docs/review/*.md`（`30d5411`、`0772757`）：
    `docs/review/` 现在只剩 TODO/DONE/DEFECT_LOG/NEXT_SESSION_PROMPT + `templates/`。
  - [x] 代码/注释/探针里指向旧文档的 9 处引用改指新文档（`bad_orientation_analysis_zh`
    → `DEFECT_LOG_zh.md` DEF-006、`history_low_level_policy_todo` → DEF-014、
    `known_issues #19` → `TODO_zh.md` P1-1）；`grep` 复核：仓库内（除三份文档自身的
    "旧文档去哪了"表）已无悬空引用。

---

## 旧文档去哪了

| 旧文档 | 现在读什么 | 原文在哪（git） |
|---|---|---|
| `high_level_todo.md` | 本文件 P0/P2 + `DONE_zh.md` 第二节 + `DEFECT_LOG_zh.md` DEF-004~007 | `codex/hl-fix-ee-command @ 0389253` |
| `known_issues.md` | 本文件 P1/P2 + `DONE_zh.md` 第三节 + `DEFECT_LOG_zh.md` | 同上 |
| `history_low_level_policy_todo.md` | `DONE_zh.md` 第四节 + `DEFECT_LOG_zh.md` DEF-014 | 同上 |
| `bad_orientation_analysis_zh.md` | `DONE_zh.md` 第一节 + `DEFECT_LOG_zh.md` DEF-001~003 | `codex/ll-height-stability @ 96e1b66` |
| `progress_summary_zh.md` | `DONE_zh.md` 全文 + 本文件"更新记录" | 同上 |
| `todo_master_zh.md` / `next_session_prompt.md`（旧） | 本文件 + `NEXT_SESSION_PROMPT.md` | `codex/docs-next-session @ bf9c5d3` |
