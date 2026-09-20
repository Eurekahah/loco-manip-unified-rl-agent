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

**优先级定义**：P0 = 挡在"能部署/能继续训练"前面；P1 = 决定训练质量上限；
P2 = 高层 replay 与工程债；P3 = 验证工具与文档。

---

## P0 —— 挡在主线前面

- [ ] **高层 P0/P1 链合并**（现在 main 上的高层任务仍是旧代码）
  - 现象：main 上跑 `Isaac-Deeprobotics-High-Level-Pick-Flat-Teacher-v0` 会在第一次
    `env.step()` 崩（`PreTrainedPickAction` 没有 `ll_command`）、IK 目标写进死字段
    `pose_command_w`、replay 的 `ee_goal` 喂的是世界系 —— 三条 P0 加"actions 观测少 7 维"
    都只在分支上修好了。
  - 要做什么：按下面顺序合并（低层已在 main）：
    `codex/hl-replay-layout`(`dc45d0e`) → `codex/hl-replay-l2`(`1f56b6e`) →
    `codex/hl-fix-ll-command`(`e064bc6`) → `codex/hl-fix-ee-command`(`0389253`) →
    `codex/hl-replay-history`(`47519b7`) → `codex/hl-ckpt-params`(`e9edc34`) →
    `codex/hl-replay-base-class`(`7ec8b4c`)
  - 已知冲突：`hl-ckpt-params`(⑦⑧，改了 6 个 action term 的载入段) 与
    `hl-replay-base-class`(R1，重写其中 3 个) 在同一批 `__init__` 上冲突 —— 先合 ⑦⑧，
    R1 里保留基类写法并把 `torch.jit.load(...)` 换成 `load_low_level_policy(...)`。
  - 验收：4 个高层任务 `--headless --num_envs 64 --max_iterations 2` 全 exit 0，
    且启动打印的 obs 维度与 checkpoint 一致（`ll-replay` 打印）。
  - **不要合** `codex/docs-review`（旧清单来源）；合并高层分支时注意它们带着旧
    `docs/review/*.md`，会和新文档重复（合并时删掉旧的）。

- [ ] **固化"训练完必须导出部署态策略"这一步**
  - 依据：`play.py` 导出的 `exported/policy.pt` 是 actor-only（输入 115 = 83 + 32 latent，
    latent 没来源），拿去 sim2sim 是错的。`2026-09-20_00-50-31/model_15000.pt` 我已经用
    `export_deploy_policy.py` 导出好放在 `exported_deploy/`（自检 0.000e+00）。
  - 要做什么：把导出步骤写进 `docs/train_history_flat_zh.md`（或训练脚本的收尾提示），
    并在 `local_export` 之外的每次训练都执行。
  - 验收：新 run 目录里 `exported_deploy/{policy.pt,policy_layout.json}` 存在且自检通过。

---

## P1 —— 训练质量（决定上限）

- [ ] **训练稳定性：`mean_noise_std` 与 `error_vel_xy` 的长期退化**
  - 现象：新旧两个 run 都出现 `Policy/mean_noise_std` 1.0 → ~1.49、
    `Metrics/base_velocity/error_vel_xy` 0.38 → ~0.89（**与 EE 课程/root_height 改动无关**）。
  - 候选改法（要 A/B）：① 约束 `init_noise_std` / 噪声上限；②
    `body_pitch_rew_s3`、`body_roll_rew_s3` 的 `num_steps` 50k → 25k，让奖励权重跟上难度；
    ③ 检查 `track_lin_vel_xy_exp` 权重与课程速度上界（现在课程会推到 ±5 m/s）是否匹配。
  - 验收：`error_vel_xy` 随迭代单调下降、`noise_std` 稳定 ≤1.2。

- [ ] **s3（完整任务）阶段的臂扰动鲁棒性**
  - 现象：`2026-09-20_00-50-31` 里 `root_height_below_minimum` 在 s3 后稳定 0.09~0.15
    （s0~s2 阶段是 0.01~0.04），说明剩下的摔倒集中在"臂大范围摆动"时。
  - 候选：① 收紧 EE 区间上界（`p_pitch` 上界 +36° ≈ 让臂举高、重心上移）；
    ② 延长 s1/s2 停留步数（现在各 25k）；③ 再评估 `root_height_below_minimum` 0.30 → 0.26。
  - 注意：**单独降阈值没用**（阈值反事实 0.30→0.26 只把 20s 触发率从 25.8% 降到 24.4%）。

- [ ] **低层 `known_issues` 剩余条目**（原编号）
  - ① `joint_mirror` 用平方差做镜像惩罚（左右关节符号约定可能相反）；
  - ⑧ `action_mirror`/`action_sync` 用 articulation 关节 id 索引动作向量 + 关节名不存在
    （当前 weight=0，属埋雷）；
  - ⑩ `arm_rewards.py` 的 `grasp_success`/`ee_approach_object` 依赖不存在的 `object`（dead code）；
  - ⑪ `HeightInvariantEECommand._update_command` 覆盖父类没调 `super()`（`pose_command_w` 永不更新）；
  - ⑫ reset 后第一帧 `pose_command_b` 全 0（观测/奖励看到零位姿目标）；
  - ⑬ `UniformThresholdVelocityCommand._resample_command` 的 `> 0.0` 是空操作；
  - ⑭ EE 目标碰撞检查静默降级 + 硬编码 AABB；⑮ 硬编码常数（0.513 / 0.09 / 0.135 / action scale）；
  - ⑱ 接触传感器与 articulation body 顺序不同（归因陷阱，建议加断言）。

- [ ] `[可选]` **执行器刚度课程 / 刚度标定**
  - 依据：`piper_arm` 现在 `DelayedPD(300/20)`，注释里的目标区间是 60~100 / 0~20；
    实测软臂（40/8）臂速 0.95 vs 硬臂（300/20）1.74 rad/s。
  - 做法：刚度做成课程（前段 40/8 → 后期 300/20），或终值降到 60~100 后重新标定。
    现在 `root_height` 已降到 0.12，优先级可往后放。

---

## P2 —— 高层 replay 与工程债（"高层修改先暂放"期间不动）

- [ ] **`pre_trained_policy_action` / `openvla_pick_action` 迁移到 `LowLevelPolicyActionBase`**
  - 现状：只补了 `ll_command`/`ll_command_w` 接口；本体仍是"就地改 cfg + 自己的
    last_action 闭包 + 无布局校验/history"；`openvla_pick_action` 的 `ee_goal` 仍是**世界系**
    （③ 只统一了 3 个 term）。
  - 限制：前者没被任何 task 注册、后者要 OpenVLA 7B 模型 → 只能做 import + 维度级验证。

- [ ] **高层 `high_level_todo.md` 第 10 条的"待确认"**
  - `HLFlatPickTerminationsCfg_PLAY` 里 `lift_object` 与 `pick_success` 语义/命名重复；
  - `highlevel/mdp/encoder.py`：`torch.hub.load` 需要联网；`_frozen_encoders` 以 name 为 key
    全局缓存（换 device 会拿到旧设备上的模型）。

- [ ] **工程性清理**（原 `known_issues.md` 二、1-8）
  - `mdp/__init__.py` 星号导入造成同名遮蔽（`ROUGH_TERRAINS_CFG` /
    `randomize_rigid_body_inertia` / `randomize_com_positions` 覆盖官方实现）；
  - `setup.py` 的 `packages` 只列顶层包、`cusrl_cfg_entry_point` 指向不存在的模块；
  - 依赖被本地魔改：`IsaacLab-5.1.0/.../task_space_actions.py`（`[IK DEBUG]` 打印 + 私有成员）；
  - `devices/vr_extented.py` 模块级 print + 无超时线程；`scripts/utils/mp4-png-composition.py`
    4 处裸 `except:`；
  - `logs/` 每个 run 50 个 `model_*.pt`（~5 MB/个，注意磁盘）；
  - 中文/英文注释混排、大段注释掉的代码。

---

## P3 —— 验证工具与文档

- [ ] **回归矩阵**（每次改动后跑一遍）
  - 低层（main 已有）：`History-Adaptation-Deeprobotics-M20-v0`、
    `Flat-Deeprobotics-M20-Piper-WBC-v0`、`Flat-Deeprobotics-M20-Piper-v0`、
    `Flat-Deeprobotics-M20-Piper-Arm-v0` —— 各 `--headless --num_envs 64 --max_iterations 2`。
  - 高层（合并高层链后）：Pick-Flat / Pick-WBC-Flat / Teleop / Nav-Flat-Teacher 同上。

- [ ] **训练曲线自动分析脚本**（`scripts/reinforcement_learning/rsl_rl/summarize_run.py`）
  - 依据：现在每次手抠 tensorboard（57 MB events，读一次 30~60 s）。
  - 内容：输入 run 目录 → 输出"关键指标 × 迭代"表（终止构成、ep_len、reward、
    `error_vel_xy`、`noise_std`、`height_error_*`、课程权重），支持两 run 对比。

- [ ] **把"EE 锚点 4 组对照"固化成一键脚本**
  - 现在靠 `probe_root_height_termination.py --freeze_ee_preset {none,default,low}` 手工跑
    （512 envs × 20 s ≈ 2.5 min）。以后改 EE 区间/锚点前先跑一遍。

- [ ] **文档收尾**
  - 代码/文档里指向旧文档的路径（`bad_orientation_analysis_zh.md`、`known_issues.md`、
    `progress_summary_zh.md`、`high_level_todo.md`、`todo_master_zh.md`）要更新为
    `TODO_zh.md` / `DONE_zh.md` / `DEFECT_LOG_zh.md`；
  - 合并高层分支时删掉它们带来的旧 `docs/review/*.md`，避免与新文档重复。

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
