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

- [ ] **回归矩阵脚本化**（矩阵本身 2026-09-20 已手工整跑一遍：8/8 EXIT=0，见 `DONE_zh.md` 第六节）
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

- [ ] **sim2sim(MuJoCo) 落地**（承接 DEF-021 的部署文档）
  - 现在只有"接口契约 + 探针 + 文档"，**还没有可运行的 MuJoCo 部署脚本**。
  - 下一步：用 `deep_robotics_model/M20_Piper_own/mjcf/M20_Piper_own.xml` 按文档第 8 节的
    七步顺序搭（零动作站立 → 零位移命令 → 开 IK → 小步进 → 速度命令）；
    数值验收用第 8 节的"ONNX vs TorchScript 相对误差 ~1e-6"。
  - 已知待解：MJCF 的 `timestep=0.002` vs Isaac `0.005`；IK 需要自己实现（DLS λ=0.01）。

- [ ] **把"EE 锚点 4 组对照"固化成一键脚本**
  - 现在靠 `probe_root_height_termination.py --freeze_ee_preset {none,default,low}` 手工跑
    （512 envs × 20 s ≈ 2.5 min）。以后改 EE 区间/锚点前先跑一遍。

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
