# 缺陷 / 特性记录（DEFECT LOG）

**文档职责**：**每条**缺陷/特性的"现象 → 根因 → 修正 → 结果"都记在这里，
方便日后回溯（为什么这么改、当时的数据是什么、被否掉的方案是什么）。

**维护约定**：新条目加在"记录"区**最上面**（时间倒序），格式见
`templates/DEFECT_ENTRY_TEMPLATE_zh.md`；未修完的在"状态"里写清并指到
`TODO_zh.md` 的优先级。

## 更新记录

| 日期 | 更新内容 | 相关 commit / 分支 |
|---|---|---|
| 2026-09-20 | 初版：把 6 份旧文档里散落的修复记录归一成 DEF-001~017 | `main @ 7ff5b86` |
| 2026-09-20 | 新增 DEF-019（⑦⑧ × R1 同段冲突的合并解法）、DEF-018（Windows 大小写路径冲突）；高层链并入 main | `codex/hl-merge-p0`（`af4602d`/`07601e9`/`30d5411`/`0772757`） |
| 2026-09-20 | 新增 DEF-020：导出部署态策略时增加 ONNX（含"绝对误差阈值误判 fp32 舍入"的教训）；run `2026-09-20_00-50-31` 用 iter=19999 重新导出 | `708ca53` |
| 2026-09-20 | 新增 DEF-021：sim2sim/sim2real 部署参考文档 + 部署规格探针（实测出"Isaac 原生关节序 ≠ MuJoCo 关节序"等关键事实） | `7458672` |
| 2026-09-20 | 新增 DEF-022（部署基线固化：`main @ 2d49f47` + 产物 sha256 + 训练代码核对 + tag `deploy-baseline-2026-09-20`）、DEF-023（P1-1 归因：`noise_std` 是饱和平台、`error_vel_xy` 是命令课程口径产物；含 P1-2 新证据） | 工具 `dc3a5b9` / 文档 `eb22401` |
| 2026-09-20 | 新增 DEF-024：探索噪声上界 `max_noise_std`（默认 0 = 不限制）+ 投影梯度实现 + hydra 覆盖踩坑；P1-1 的 A/B（cap 1.2 / entropy_coef 0.002）已启动，结果待回填 | 代码 + `docs: P1-1 A/B` 提交 |
| 2026-09-20 | 新增 DEF-025：桌面版自动化（heartbeat / cron）的唤醒投递条目缺 `call_id`，被 deepseek `/responses` 一律 422 拒绝、并**永久污染所在线程**；两条 automation 已 `PAUSED`，P1-1 收尾改回手动 | 本次（docs-only） |
| 2026-09-22 | 回填 **DEF-024 §4**：P1-1 A/B 实测收尾（cap=1.2 通过、**建议作为默认**；`entropy_coef=0.002` 通过但略逊；意外点 `entropy_coef=0` 反而 +15% s3 摔倒）+ **统一窗口口径修正** | 本次（docs-only） |
| 2026-09-29 | 新增 DEF-026~DEF-031：静止伫立专项（零速漂移 + 站姿占比课程）、`joint_mirror` 镜像符号 bug（"右后腿往右前方撇"的根因）、扰动加强 + 课程拉长、多地形任务（粗糙 0.01~0.05 + 正反斜坡 + 平地）、遥操 history 任务、以及**本机跑不了生成地形任务**的平台问题 | 分支 `codex/ll-train-detail-fix` |
| 2026-09-30 | 新增 **DEF-035**：剩下 3 个高层 action term（`PreTrainedPickAction` / `PreTrainedPickWBCAction` / `TeleopLLAction`）收进 `LowLevelPolicyActionBase`，删掉各自的 ~200 行重复机械；顺带删除无人注册的 `PreTrainedPolicyAction`（openvla 时代遗留）；新增 `probe_reset_anchor_timing.py` 证明复位钩子读到的是**复位后**状态 | 分支 `codex/ll-train-detail-fix` |

---

# 记录（新→旧）

### DEF-035 `2026-09-30` P2：3 个高层 action term 收进 `LowLevelPolicyActionBase`（用户要求"尽量一个基类"）

| 项 | 内容 |
|---|---|
| 类型 | 重构 / 工程债（**不需要训练**，本机可验证） |
| 状态 | 已完成并实测（同批回归 11 OK / 1 SKIP / 0 FAIL） |
| 关联 | `codex/ll-train-detail-fix`；`highlevel/mdp/{pre_trained_pick_action,pre_trained_pick_wbc_action,teleop_ll_action,low_level_policy_action,__init__}.py`、`highlevel/high_level_env_cfg.py`；新探针 `scripts/reinforcement_learning/rsl_rl/probe_reset_anchor_timing.py`；`TODO_zh.md` P2 |

**1. 现象（重构前）**

`PreTrainedNavAction` 早就继承 `LowLevelPolicyActionBase`（清单 ⑤ 的 R1），但另外三个
高层 action term 各自抄了一份**完全同构**的机械代码（每个文件约 200 行）：

* `self.robot = env.scene[cfg.asset_name]`、`load_low_level_policy(cfg.policy_path, ...)`；
* 手搓三个低层 action term（`low_level_leg_actions` / `..._wheel_actions` / `..._ee_actions`）；
* `resolve_layout` + `check_low_level_action_cfgs` + `_joint_pos_dim/_wheel_vel_dim/_ee_ik_dim`；
* 三个 `torch.zeros` 低层动作缓存 + `last_action()` 闭包；
* `build_low_level_observation_group` + `expected_policy_obs_dim` + `build_low_level_obs_manager`
  + `verify_low_level_layout` + `build_history_window`；
* `apply_actions()` 里的 `counter % low_level_decimation` tick 循环 + 策略输出切分 +
  三个低层 action term 的 `process_actions/apply_actions`。

⇒ 清单 ②③④⑥⑨⑯ 的每一条都要在 3~4 个文件里各改一遍，而且"训练用什么、回放就用什么"
没有任何结构性保证（历史上 DEF-008/009/010/011 都是这么来的）。

**2. 修正**

三个类改成 `class X(LowLevelPolicyActionBase)`，`__init__` 只保留三段：
① `_raw_actions` 分配（**必须在 `super().__init__` 之前**，因为基类构造低层观测组时会用到它）
→ ② `super().__init__(cfg, env)` → ③ 任务专属状态。删掉各自的 `apply_actions`，
改用基类的 `_on_low_level_tick()` / `_on_reset(env_ids)` / `_extra_cache_tensors()` 三个钩子：

| 类 | `_build_low_level_obs_cfg` | `_extra_cache_tensors` | `_on_low_level_tick` | `_on_reset` |
|---|---|---|---|---|
| `PreTrainedPickWBCAction` | 用基类默认（13 宽 `_ll_command`） | `[]`（与迁移前一致，只清三个低层动作缓存） | `push_ee_target_to_ik` | `_target_initialized[ids] = False` |
| `PreTrainedPickAction` | 覆写：`velocity_commands` 取 `_raw_actions[:, :3]`（**必须**：本类复位时 `_raw_actions` 被清零、`_ll_command` 不会） | `[self._raw_actions]`（与迁移前一致） | `push_ee_target_to_ik` | 同上 |
| `TeleopLLAction` | 用基类默认（13 宽 `_ll_command`） | `[]` | `push_ee_target_to_ik` | `absolute_commands` ? `recalibrate(ids)` : `_capture_default_ee_pose(ids)` + `_reset_default_body_pose(ids)` |

同时删掉**无人注册**的 `PreTrainedPolicyAction`（`pre_trained_policy_action.py`，371 行：
仍在本类里就地改 `cfg.low_level_observations`、硬编码 `scale=0.125`、引用
`mdp.joint_pos_rel_without_wheel`）——它与已删除的 `openvla_pick_action.py` 同属 openvla 时代，
只在一段 `#` 注释里被提到过。

**3. 验收（本机，`--headless --num_envs 64 --max_iterations 2`）**

* **回归矩阵**：12 任务 **11 OK / 1 SKIP（生成地形，DEF-031）/ 0 FAIL**
  （`logs/smoke/2026-09-30_regression_p2.md`）。
* **低层布局打印逐字节一致**：把改动前那批日志（`logs/smoke/2026-09-30_after_cleanup_*.log`）
  与改动后的（`logs/smoke/2026-09-30_*.log`）只比 `[ll-replay:*]` 行 ——
  Pick-Flat / Pick-WBC-Flat / Teleop / Teleop-History / Nav-Flat **全部完全一致**：

  | 任务 | 低层 policy 动作维度 | 低层 obs 维度（期望） | 备注 |
  |---|---|---|---|
  | `...Pick-Flat-Teacher-v0` | 23（leg 12 + wheel 4 + ee_ik 7） | **83（83）** | L1 显式 `ee_action_dim=7` |
  | `...Pick-WBC-Flat-Teacher-v0` | 16 | **76（76）** | 观测含 `body_pose_cmd(3,)` |
  | `Isaac-M20-Piper-Teleop-v0` | 16 | **76（76）** | 同上 |
  | `Isaac-M20-Piper-Teleop-History-v0` | 16 | **83（83）** + history 10×70 | 双输入 forward |
  | `...Nav-Flat-Teacher-v0` | 16 | 76（76） | 迁移前就在基类上 |

**4. 复位钩子的时机（新增证据，避免"看着对、其实晚一步"）**

基类的 `reset()` 会被 `ActionManager.reset`（在 `ManagerBasedRLEnv._reset_idx` 里）调到，
钩子里直接读 `robot.data` 是否已经是**复位后**状态，值得单独验一次（旧实现在
`apply_actions` 里用 `episode_length_buf == 0` 检测，等于**晚一个 env step**）。

新探针 `probe_reset_anchor_timing.py`（`--action_scale 3.0` 把机器人推翻后强制复位）实测：

```
[post_step38]                        root_z=+0.5371 ee_z=+0.9127   ← 复位前（终止那一刻）
[_on_reset(before)]  ep_buf[0]= 40   root_z=+0.5500 ee_z=+0.9827   ← 已经换成复位位姿
[post_step39]        ep_buf[0]=  0   root_z=+0.5500 ee_z=+0.9827
[_reset_target_to_current_ee(before)] root_z=+0.5500 ee_z=+0.9827
[_reset_target_to_current_ee(after)]  target_ee_pos_b[0] = [0.34923, 0.0, 0.43266]
```

⇒ `_on_reset` 里读到的 `root_z/ee_z`（0.5500 / 0.9827）**是复位位姿**，不是终止的摔倒位姿
（IsaacLab 的 `write_root_pose_to_sim` / `write_joint_state_to_sim` 会把
`_body_link_pose_w.timestamp` 置 `-1`，下一次访问 `robot.data` 时重新从 PhysX 取）。两次 episode
重锚出来的 `target_ee_pos_b` 两次一致到 **1e-7**。

**5. 唯一的（有利的）行为差异**

Pick/WBC 的"重锚 EE 目标"从**复位后第 2 个 env step**（= 新 episode 已跑了 40 个 sim step、
0.2 s）提前到**第 1 个 env step**（复位当下）。Teleop 的 `recalibrate` /
`_capture_default_ee_pose` 同理提前，且因为 `_on_reset` 已经能看到复位状态，二者等价。
⇒ 语义更贴合"把目标锚在 episode 起点位姿"，不是回归。

**6. 本批**没做**的（仍在 TODO P2）**：`mdp/__init__.py` 星号导入遮蔽（影响面小，见 TODO）；
四个文件里遗留的未使用 import（`Articulation` / `ObservationManager` / `check_file_path` /
`read_file`，重构前就没有）—— 留到统一的"清理注释与 import"那一批。

**7. 重构暴露的一个真实缺陷：`probe_history_window.py` 的猴子补丁失效**

`probe_history_window.py` 为了核对"history 窗口最后一帧 == 低层训练函数独立复算"，
把 `teleop_ll_action.run_low_level_policy` 换成了记录用的 wrapper —— 它依赖各 action term
模块里 `from low_level_replay import run_low_level_policy` 留下的**模块级全局**。
迁移后调用点移到了基类（`low_level_policy_action.apply_actions`），于是补丁打在了一个
不再被读的全局上：

```
AttributeError: module '...highlevel.mdp.teleop_ll_action' has no attribute 'run_low_level_policy'
```

**修法**：改成对**基类模块**（`low_level_policy_action`）装补丁，并把"还有该全局的模块"
顺手都装上；`_wrapper` 内部调用**事先保存**的原始实现（不能闭包引用单个 `original`，
否则多模块同时装同一个函数时会 `NameError`）。

**修后实测**（`--task Isaac-M20-Piper-Teleop-History-v0 --num_envs 8 --steps 40`，EXIT=0）：

```
[probe] (2) 最后一帧 vs 训练函数独立复算（40 次低层 tick） max|...| = 0.000e+00
[probe] (3) 整窗顺序核对（含复位填充，72 次带复位填充） max|窗口 − 期望拼接| = 0.000e+00
[probe] 结论: history 窗口与低层训练口径一致（逐位相同）
```

⇒ 教训：**探针依赖"模块级全局"时，重构要一起改**；本仓库的探针属于"验收口径"的一部分，
所以每批改动都要把相关探针也跑一遍（本轮除回归矩阵外，另外跑了
`probe_history_window.py` / `probe_reset_anchor_timing.py`）。

### DEF-034 `2026-09-30` 本机可做的收尾批：回归矩阵脚本化 + known_issues ⑧⑩⑪⑫⑱ + 三处工程债

| 项 | 内容 |
|---|---|
| 类型 | 工具/缺陷/清理（**不需要训练**，本机可验证） |
| 状态 | 已修并实测（回归矩阵 12 任务：见 §6） |
| 关联 | `codex/ll-train-detail-fix`；`scripts/reinforcement_learning/rsl_rl/smoke_regression.py`、`.../probe_ee_command_init.py`、`velocity/mdp/{rewards,observations,arm_rewards}.py`、`velocity_env_cfg.py`、`highlevel/mdp/encoder.py`、`source/rl_training/setup.py`、`scripts/utils/mp4-png-composition.py`；`TODO_zh.md` P1-4 / P3 |

**1. 回归矩阵脚本化（TODO P1-4 + P3）**

* 新增 `scripts/reinforcement_learning/rsl_rl/smoke_regression.py`：逐任务独立进程跑
  `train.py --headless --num_envs 64 --max_iterations 2`，日志落
  `logs/smoke/<日期>_<task>.log`（沿用历史命名），结尾打 Markdown 表 + `--out` 落盘，
  **有 FAIL 才非零退出**（可直接接脚本链/CI）。
* **卡死检测**（P1-4）：日志文件连续 `--stall-timeout` 秒不增长 ⇒ 判 **SKIP** 并杀进程树
  （Windows `taskkill /T /F`、POSIX `killpg`）。本机跑生成地形任务就是这个表现（DEF-031），
  所以本机上 `Rough-*` 会稳定落进 SKIP、**不再无限等待**；换到能跑生成地形的机器上同一份
  脚本会自动变成 OK。
* 自测（3 任务）：`History-Adaptation` OK(47s)、`Flat-...-WBC` OK(78s)、
  `Rough-Slopes` **SKIP(102s，70s 无输出被杀)** ⇒ OK 2 / SKIP 1 / FAIL 0，退出码 0。

**2. known_issues ⑫ 修复 + ⑪ 实测澄清（EE 位姿命令的"reset 首帧"）**

* 新增 `scripts/reinforcement_learning/rsl_rl/probe_ee_command_init.py`：建环境 → reset →
  打印 `pose_command_b / pose_command_w / 与真实 EE 位姿(root 系)的偏差`。
* 实测（修复前，`Flat-Deeprobotics-M20-Piper-WBC-v0`）：

  | 时刻 | `pose_command_b[0]` | 与真实 EE 位姿差 | `pose_command_w[0]` |
  |---|---|---|---|
  | `gym.make` 之后（未 reset） | `(0,0,0, 1,0,0,0)` | **0.2997 m** | 全 0 |
  | **`env.reset()` 之后**（= 训练循环第一步的观测） | 仍是 `(0,0,0, 1,0,0,0)` | **0.4327 m** | 全 0 |
  | step≥1 | `(0.3523, 0.0001, 0.4301, …)` | 2~6 cm（正常插值追赶） | 正常 |

  ⇒ **⑫ 成立**：复位之后那一步，观测（`ee_goal`）与奖励看到的仍是父类初值
  "目标 = 底盘原点 + 单位姿态"。原因是 `CommandTerm.reset()` 只调 `_resample_command()`
  （本命令项只写 `pose_start_b/pose_end_b`），而 `pose_command_b` 要等下一次 `compute()`
  里的 `_update_command()` 才赋值。
* **⑪ 澄清**：父类 `UniformPoseCommand._update_command()` 本来就是 `pass`，所以
  "漏调 super() 丢逻辑"不成立；`pose_command_w` 也不是永不更新（它在父类
  `_update_metrics()` 里算，每个 compute 都会更新，只是**比 `pose_command_b` 晚一拍**，
  实测 step≥1 时两者差 6e-3~1.7e-2，属父类既定顺序、不是缺陷）。
* 修法：给 `HeightInvariantEECommand` 加 `reset()` 覆写 —— `super().reset()` 之后再补一次
  "命令 ← 插值起点（= 复位那一刻真实 EE 位姿）"。修复后同一探针实测
  `after_reset`：`pose_command_b[0] = (0.3492, 0.0000, 0.4327, -0.7373, …)`（正是
  `gripper_base` 相对 base 的几何位置，与 DEF-021 的 `(0.3492, 0, 0.4326)` 一致）、
  **与真实 EE 位姿差 = 0.000e+00** ⇒ 复位首帧机械臂不需要动、也不会把底盘拽一下。

**3. known_issues ⑧：删掉 `action_mirror` / `action_sync`（"打开就炸"的埋雷）**

* 两者用 `asset.find_joints(...)`（**articulation 关节 id**）去索引
  `env.action_manager.action`，而动作向量是按**动作项自己的列序**（12 腿 fl,fr,hl,hr + 4 轮）排的
  ⇒ 下标不一致，weight 一开就算错；
* 配置里的关节名是 Go1 风格（`FR_hip_joint` / `RL_thigh_joint` …），本机型不存在
  （M20 用 `fl_hipx_joint` …）⇒ 一打开就抛"找不到关节"；
* 同样的目的已由**状态层**的 `joint_mirror_signed`（带符号约定、4 对镜像，DEF-027）覆盖。
  ⇒ 删除 `RewardsCfg.action_mirror` / `.action_sync` 两个 term 与 `rewards.py` 里对应两个函数，
  原地留注释说明。

**4. known_issues ⑩：给 `grasp_success` / `ee_approach_object` 加明确报错**

* 这两项默认 `SceneEntityCfg("object")`，而**低层 velocity 场景没有 object**
  （只有高层 pick / openvla / nav 场景有）⇒ 以前误接上会得到一个难懂的 `KeyError`。
  现在先 `_require_scene_entity(...)`：报错里写明"本奖励需要 object / 当前场景只有哪些实体 /
  它是给高层用的"。（保留函数本身：它们对高层场景是有效工具，只是当前无人引用。）

**5. known_issues ⑱ + 三处工程债**

* ⑱ 接触传感器行序 ≠ articulation 行序（历史"臂/夹爪 90 N"误判的根源）：在启动期布局自检
  `mdp.check_policy_layout` 里增加打印 —— 传感器 body 数、是否与 articulation 顺序一致、
  前 6 个 `名字#行号`，并检查"传感器里的 body 名都能在 articulation 里找到"（找不到就警告，
  因为那就无法按名字归因了）。
* `source/rl_training/setup.py`：`packages=["rl_training"]` → `find_packages(include=["rl_training","rl_training.*"])`
  （原来子包不会被打进 wheel/sdist）。
* `highlevel/mdp/encoder.py`：`_frozen_encoders` 的 key 从 `name` 改成 `(name, device)`
  （原来同一进程里换 device 会拿到建在旧设备上的模型）；`register_trainable` 的冲突检查
  随之用 `_frozen_names()`。
* `scripts/utils/mp4-png-composition.py`：4 处裸 `except:` → `except Exception:`
  （不再吞 `KeyboardInterrupt` / `SystemExit`）。

**6. 回归验收（2026-09-30，`--num_envs 64 --max_iterations 2`，13 任务）**

| 任务 | 结果 | 耗时(s) |
|---|---|---|
| `History-Adaptation-Deeprobotics-M20-v0` | OK | 38 |
| `Flat-Deeprobotics-M20-Piper-WBC-v0` | OK | 38 |
| `Flat-Deeprobotics-M20-Piper-v0` | OK | 37 |
| `Flat-Deeprobotics-M20-Piper-Arm-v0` | OK | 36 |
| `Isaac-Deeprobotics-High-Level-Pick-Flat-Teacher-v0` | OK | 108 |
| `Isaac-Deeprobotics-High-Level-Pick-WBC-Flat-Teacher-v0` | OK | 106 |
| `Isaac-M20-Piper-Teleop-v0` | OK | 42 |
| `Isaac-M20-Piper-Teleop-History-v0` | OK | 42 |
| `Isaac-Deeprobotics-High-Level-Nav-Flat-Teacher-v0` | OK | 91 |
| `History-Ablation-PushOnly-Deeprobotics-M20-v0` | OK | 37 |
| `History-Ablation-RewardOnly-Deeprobotics-M20-v0` | OK | 49 |
| `Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0` | **SKIP** | 157（120s 无输出被杀，DEF-031） |

**OK 11 / SKIP 1 / FAIL 0**（退出码 0），完整表在 `logs/smoke/2026-09-30_regression.md`。
⇒ 本批改动（删两个 reward term、EE 命令 reset 覆写、启动期多打印、setup/encoder/mp4 清理）
**没有引入回归**：11 个能跑的任务全部 EXIT=0 且 `Learning iteration` 计数 = 2。

**7. 第二批（同日，按用户答复调整）**

* **cusrl 全部删除**（用户确认"本身没有使用 cusrl 训练"）：
  `.../deeprobotics_m20/__init__.py` 的 **7 处** + `.../deeprobotics_lite3/__init__.py` 的
  **2 处** `cusrl_cfg_entry_point` 注册字段，以及 `source/rl_training/setup.py` 里的
  `"cusrl[all]"` 依赖。⇒ 不再引用不存在的 `agents/cusrl_ppo_cfg.py`，装包时也不会去拉 cusrl。
* **视觉编码器"本地权重优先、默认不联网"**（`highlevel/mdp/encoder.py`）：
  新增两个环境变量 —— `RL_TRAINING_ENCODER_DIR`（默认 `~/.cache/rl_training/encoders`）、
  `RL_TRAINING_ALLOW_ENCODER_DOWNLOAD`（默认 **0**）。`dinov2_small/dinov2_base/clip_vit/cnn`
  一律先找 `<ENCODER_DIR>/<name>.pth`；找不到就**报错**并给出"权重该放哪 + 怎么自己导出一份"的
  说明；只有显式开开关才回退 `torch.hub` / `open_clip(pretrained='openai')` 联网下载。
  `UnfrozenResNet18` 也改成默认 `weights=None`（不再拉 ImageNet 权重）。
  另外把 `cnn` 分支里 `from my_project.models import ...`（**不存在的模块**，死代码）换成
  本文件自带的 `LightweightCNN` + 本地 checkpoint。实测：无本地权重时
  `get_encoder("dinov2_small")` 抛出可操作报错而不是默默联网。
* **openvla 分支删除**（用户确认"遗弃很久了"）：删除
  `mdp/openvla_pick_action.py`（22 KB）、其配置
  `config/high_level/hl_flat_openvla_env_cfg.py`、`mdp/__init__.py` 的 star-import，
  以及 `low_level_policy_action.py` / `low_level_replay.py` 里的两处提及。
  依据：该 cfg **没有被任何 task 注册引用**，`VLAPickAction` 也没有其它使用点 ⇒ 死分支。
* **`devices/vr_extented.py` 的模块级 print**：那 3 行是在**模块导入时**执行的（不在函数里），
  只要 import 就会往 stdout 写 XLeVR 路径 —— 与"VR 能不能连上"无关（连接由后台线程
  `_run_vr_services()` 负责）。现在改成 `RL_TRAINING_VR_DEBUG=1` 才打印。

**8. 本批**没做**的（仍在 TODO）**：`mdp/__init__.py` 星号导入遮蔽；
`pre_trained_pick_action` / `pre_trained_pick_wbc_action` / `teleop_ll_action` 迁移到
`LowLevelPolicyActionBase`（**用户要求：尽量收成一个基类**，基类已有
`_build_low_level_obs_cfg` / `_on_low_level_tick` / `_route_policy_output` 三个钩子，
迁移时还需要给基类补一个 `_on_reset(env_ids)` 钩子以承载 pick 的"重锚 EE 目标"与
teleop 的"重新标定"）；EE 锚点 4 组一键脚本。
> 其中 **`_on_reset(env_ids)` + `reset()` 基类钩子已经在本次预置好**（行为中性：
> 默认空实现，现有子类不覆盖 ⇒ 实测 Teleop-History / Pick-WBC / Nav 三个任务仍 OK），
> 剩下的三个类的迁移是纯机械改动（下次做）。

### DEF-032 `2026-09-30` 云端（autodl 私有云 TiEV）接力：环境复制方法 + 已启动的长跑

| 项 | 内容 |
|---|---|
| 类型 | 基础设施 / 交接 |
| 状态 | 进行中（长跑未结束） |
| 关联 | `codex/ll-train-detail-fix`；`TODO_zh.md` P1-1''' |

**1. 为什么要上云**：本机（RTX A4000 + Ryzen 7 2700X，8 核）4096 envs ≈ **5.5~6.5 s/iter**
⇒ 20k iter 要 **30+ 小时**，而且**跑不了生成地形的任务**（DEF-031）。云端（3090）实测
**3.2 s/iter**（History 平地）/ 6.3 s/iter（多地形），快一倍以上。

**2. 控制台事实（避免下次重新摸索）**

* 登录：`https://private.autodl.com/console/instance`（租户 TiEV-Tj / 用户 韩敬霄）。
* **现有 6 个历史实例**（每个 1×3090，均已关机）：其中
  `ultra CPU-5950-pc5-2GPU ffda41bd1f-38f1325f`（2026-09-18）**就是本项目那份环境**：
  `/root/autodl-tmp/IsaacLab`、`/root/autodl-tmp/loco-manip-unified-rl-agent`、
  conda env `/root/miniconda3/envs/env_isaaclab`（20 GB，Isaac Sim 在里面）。
  但它的主机 `ffda41bd1f` 的空闲 GPU 是 **0/2** ⇒ 只能 **无卡模式开机**（￥0.10/时）。
* **显卡驱动很关键**：Isaac Sim 5.1 只在 **驱动 580.x** 的主机（`ffda41bd1f` 580.173.02、
  `bbc64d91a6` 580.178.04）上干净启动；驱动 570.x / 535.x 的主机
  （`686346b9c6`、`c71a49a292`、`d54d48b2fa`、`1df740a715`）启动时打
  `vkCreateInstance failed. Vulkan 1.1 is not supported` + `Unable to get IGpuFoundation`，
  **但 headless 训练仍能跑**（实测 GPU 利用率 80%，只是渲染栈没起来，无相机场景无影响）。
* 磁盘：`bbc64d91a6` 最宽松（938G/59% 用）；`686346b9c6` 只有 ~18G 余量（放完 22G 环境后）；
  `c71a49a292` 系统盘 95% 用。**注意**：`/root/autodl-tmp` 是**实例私有**的数据盘，
  换实例/克隆实例都不会带过去 —— 这就是"换机器必须重新 clone 代码 + IsaacLab"的原因。
  幸运的是 **NFS `10.60.144.11:/home/autolab/Data/pub_data` 是共享的且可写**（挂到 `/root/tievnas`），
  需要跨实例传大文件时可以借它中转。

**3. 环境复制方法（可复用，比重新装快得多）**

* 控制台点"登录指令/密码"的**复制图标**即可拿到 `ssh -p <port> root@10.60.144.11` + 密码
  （注意：本机 `~/.ssh` 之前是空的，没有免密配置；密码是每个实例各一份）。
* 实例之间直接 `rsync`（20 GB env 在同一物理主机内几分钟就完了）：

  ```bash
  # 在目标实例上执行：把参考实例的 IsaacLab + 仓库 + conda env 拉过来
  SRC=root@10.60.144.11 ; SSH="ssh -p 635 -o StrictHostKeyChecking=no"
  rsync -a -e "$SSH" $SRC:/root/autodl-tmp/IsaacLab/                       /root/autodl-tmp/IsaacLab/
  rsync -a -e "$SSH" $SRC:/root/autodl-tmp/loco-manip-unified-rl-agent/   /root/autodl-tmp/loco-manip-unified-rl-agent/
  rsync -a -e "$SSH" $SRC:/root/miniconda3/envs/env_isaaclab/              /root/miniconda3/envs/env_isaaclab/
  ```

  前提：目标实例的 `~/.ssh/id_ed25519.pub` 已加到源实例的 `authorized_keys`（一次即可）。
  conda env 里是**绝对路径的 editable 安装**，所以两条路径必须与原实例一致。
* **`git fetch` 在实例上不一定通**（实测 `bbc64d91a6` 报 `GnuTLS recv error (-110)`），
  可靠做法是在本机 `git bundle create x.bundle <branch> ^<base>`，再 `scp` 过去
  `git fetch x.bundle 'branch:refs/heads/branch'`。（本次已把分支推到 GitHub
  `origin/codex/ll-train-detail-fix`，能连 GitHub 的机器可直接 pull。）
* 跑训练：`/root/miniconda3/envs/env_isaaclab/bin/python scripts/.../train.py --task ... --headless`
  （env 在 `/root/miniconda3/envs/env_isaaclab`；`--num_envs 4096 --seed 42` 与本地口径一致）。

**4. 已启动的长跑（本次）**

| 实例 | 主机 | 任务 | 配置 | 起始 | 实测速度 |
|---|---|---|---|---|---|
| `bbc64d91a6-99f1820e`（4UGPU） | 10.60.144.11:1237 | `History-Adaptation-Deeprobotics-M20-v0` | 4096 envs / seed 42 / 20k iter / 软化版静止惩罚 | 2026-09-30 ~00:10 | 3.2 s/iter（≈18 h） |
| `686346b9c6-b16aa8d9`（planner） | 10.60.144.11:291 | `Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0` | 4096 envs / seed 42 / 20k iter | 2026-09-30 ~00:20 | 6.3 s/iter（≈35 h） |
| `ffda41bd1f-38f1325f`（参考实例） | 10.60.144.11:635 | —— | **无卡模式**常开，作为文件源 | 2026-09-30 ~00:00 | —— |

> 还没做：`c71a49a292`（cvpr，1/3 空闲）本打算跑 `max_noise_std=1.2` 的 20k 对照（P1-1''），
> 但它的系统盘已用 95%，且驱动是 570.x；等上面两条跑完/有富余再决定。

**5. 运维提醒**

* 实例是**按小时计费**（GPU ￥0.01/时、无卡 ￥0.10/时，都很便宜），但**不用了要关机**；
  参考实例（无卡模式）如果不是为了当文件源，也可以关掉。
* 云端 run 目录在实例的 `/root/autodl-tmp/loco-manip-unified-rl-agent/logs/rsl_rl/...`
  （已 gitignore）；结果要拿回来就 `scp`（或先 `summarize_run.py` 出表）。
* 每个实例只跑了 **1** 个训练（用户习惯），符合"尽量别超过 4 台"。

### DEF-033 `2026-09-30` 第三台实例（planner 主机）+ 把"2×2 消融 + cap12 对照"排进队列

| 项 | 内容 |
|---|---|
| 类型 | 基础设施 / 实验编排 |
| 状态 | 进行中（四条 run 在跑/排队，结果待回填） |
| 关联 | `codex/ll-train-detail-fix`（消融提交 `6297af4`）；`TODO_zh.md` P1-1'' / P1-1''' / P1-3 |

**1. 为什么又建了一台**：用户指出 `0d5c409456`（good CPU-225-6000）上有一张空闲的
**RTX 6000D (83GB)**（UUID `GPU-c0c614a0-91fa-029e-cc73-1deaf02310c5`），可以再加一条训练。

**2. 踩到的坑（两次创建都失败，且卡被别人抢走）**

* 先试**克隆实例**（从 `ffda41bd1f-38f1325f` 的系统盘）：`创建失败`。
  原因基本可确定是**磁盘**：那道实例的系统盘已用 73.73%（≈675GB），而目标主机只剩 208GB
  （克隆默认还会连**数据盘**一起克隆，第一次弹窗里 `克隆数据盘` 默认是勾上的）。
* 再试**从系统镜像新建**（torch:cuda12.8-ubuntu22.04-py312）：同样 `创建失败`
  （按用户提示换了数据盘 `/data`、又换 `/SSD1` 都没成功；那台主机的 8 张卡里只剩 1 张空闲，
  很可能与主机磁盘池有关）。两次失败后我**释放了这两个失败实例**（它们从未启动、无数据）。
* 就在这几分钟里，那张 6000D 被**别的用户占走**了（占用详情显示
  `0d5c409456-9dc0d369 (炼丹师6927)`，开始时间 10:57:36），该主机变成 0/8。
* ⇒ 结论：**建实例要"先占坑再折腾"**；`0d5c409456` 那台目前不可用。
* 于是改在**有 2/8 空闲 GPU 的 planner 主机 `686346b9c6`** 上新建（￥0.01/时、驱动 570、
  与我已在跑的实例同款）→ 成功，实例 `686346b9c6-1dcaf819`。

**3. 新实例的布置（SSH 端口拿不到时的替代通道）**

* 这台新实例的**控制台"登录指令"复制按钮失效**（密码能复制、SSH 命令复制不到），
  端口扫描也没能确认 SSH 端口。于是改用 **JupyterLab 的 REST API**：
  `http://10.60.144.11:1471/jupyter/`（token 从页面的 `#jupyter-config-data` 里读），
  支持 `GET/PUT /api/contents/<path>`：**写脚本、读日志**都可以走 HTTP，
  执行只在 JupyterLab 的终端里敲**一条**命令（脚本后台跑、输出重定向到文件）。
* 环境复制沿用 DEF-032 §3 的 rsync：先把新实例的公钥加到源实例
  （这次源用 `686346b9c6-b16aa8d9`，同主机、最快），再 rsync
  `IsaacLab(199M) + 仓库(3.2G) + conda env(20G)`。
* 代码更新：`git fetch /root/ll-ablation.bundle 'codex/ll-train-detail-fix:refs/remotes/bundle/ablation'`
  + `git merge --ff-only`（**注意**：直接 `git fetch ... :refs/heads/<当前分支>` 会被 git 拒绝，
  必须先 fetch 到 `refs/remotes/...` 再快进）。
* 冒烟：`History-Ablation-PushOnly-...` 2-iter 在云端 **EXIT=0**、`Learning iteration` 计数 = 2；
  训练速度 **2.65 s/iter**（4096 envs）。

**4. 现在的四条 run（2026-09-30 11:20 状态）**

| # | 实例 | 任务 / run_name | 关键配置 | 预计 |
|---|---|---|---|---|
| 1 | `bbc64d91a6-99f1820e` | `History-Adaptation-*` / `cloud_soft20k` | 4096 envs / seed 42 / 20k / 软化版静止惩罚 | 今天 ~19:30 |
| 2 | `bbc64d91a6-99f1820e`（**排队**） | 同上 / `cloud_cap12_20k` | 再加 `agent.policy.max_noise_std=1.2` | 待 run 1 结束后自动开跑（~18 h） |
| 3 | `686346b9c6-b16aa8d9` | `Rough-Slopes-History-Adaptation-*` / `cloud_roughslopes20k` | 多地形 / 20k | 明天 ~12:30 |
| 4 | `686346b9c6-1dcaf819`（新） | `History-Ablation-PushOnly-*` / `abl_pushonly_10k` | 只加强 push / 10k | 今天 ~19:00 |
| 5 | `686346b9c6-1dcaf819`（**链式排队**） | `History-Ablation-RewardOnly-*` / `abl_rewardonly_10k` | 只改奖励 / 10k | 明天 ~02:40 |

> 排队方式：run 1 那台用 `while pgrep -f "run_name cloud_soft20k"; do sleep 60; done` 的守护脚本
> 等旧跑完再 `setsid nohup` 起新 run；新实例那台用同一个 bash 脚本里 `run_one; run_one` 串起来。
> 都是 `setsid nohup ... < /dev/null`，脱离 SSH/Jupyter 会话，并且**不要关机**（关机就全没了）。

**5. 判据**：四条都是 10k/20k + 4096 envs + seed 42，和已有的"旧代码 10k/20k"、
">完整包 10k" 同口径 ⇒ 固定命令 eval（`eval_fixed_command.py`）+ 步态探针
（`probe_gait_symmetry.py`）直接可比。

### DEF-031 `2026-09-29` 本机（A4000 / Windows）跑不了 `terrain_type="generator"` 的任务：env 创建期死锁

| 项 | 内容 |
|---|---|
| 类型 | 环境/平台问题（**不是本次改动引入**） |
| 状态 | 已定性（本机环境问题，非代码问题）——**云端 3090 上同一条命令 2-iter 冒烟通过**（见 DEF-032 §4），本机多地形验收改到云端做 |
| 关联 | 复现：`Rough-WO-Stairs-History-Adaptation-Deeprobotics-M20-v0`、`Rough-Slopes-...`、甚至**未改动的原始代码**（`git stash` 后同一条命令）；`TODO_zh.md` P1-4 |

**1. 现象**

* 任何用 `terrain_type="generator"` 的任务（`Rough-*` 系列、新加的 `Rough-Slopes-*`），
  `train.py --headless --num_envs 64 --max_iterations 2` 在 env 创建期**卡死**：
  日志停在 `[simulation_context.py] WARNING: The 'enable_external_forces_every_iteration' ...`
  之后再无输出，20 分钟不动。
* 同一条日志里必带一条 Isaac 断言：
  `ASSERTION FAILED (continuing execution but crash may occur): carb.tasking/SharedMutex.cpp(128) SharedMutex::lockExclusive ...`。
* 此时 `nvidia-smi` 利用率 4%、进程 CPU 时间 30 s / 20 min ⇒ **是死锁，不是"生成太慢"**。

**2. 排除实验（证明与本次改动无关）**

* 把本次全部改动 `git stash` 掉，用**原始代码**跑 `Rough-WO-Stairs-History-...`：
  在同一个位置以同样方式卡死（日志长度 6599 B，最后一行同样是那条 physics warning + 断言）。
* 用 `eval_fixed_command.py`（它把地形缩到 `num_rows=num_cols=5`、关地形课程）跑
  `Rough-Slopes-...`：依旧卡死 ⇒ 与网格规模、与地形课程开关无关。
* 对照组：**所有平地任务**（`History-Adaptation-*`、`Flat-*-WBC-v0`、`Isaac-M20-Piper-Teleop*`）
  `--num_envs 64 --max_iterations 2` 全部 **EXIT=0** ⇒ Isaac Sim 本身、本仓库的 MDP 组装都正常，
  只有"要生成地形网格"这条路挂住。

**3. 影响 / 处理**

* 本机**无法**对多地形任务做端到端冒烟或训练；历史 run 目录里也没有任何 `Rough-*` 的 smoke 日志
  （2026-09-20 那轮"8 任务回归"全是平地 + 高层）。
* 本次对多地形任务只做到：① `__post_init__` 成功执行（`train.py` 打印
  `Parsing configuration from: ...RoughSlopesEnvWBCConfig` 之后才卡，说明 cfg 组装 OK）；
  ② 地形组成 / 噪声区间 / 课程项与 `RoughWOStairsEnvWBCConfig` 逐项对照；
  ③ 共享的奖励/课程类由平地任务（同一批 `WBCRewardsCfg`/`WBCCurriculumCfg`）实测覆盖。
* **待办**：换到能跑生成地形的机器（autodl 容器）后，先补 `Rough-Slopes-*` 的 2-iter 冒烟，
  再跑一次短训练，最后把数字回填到本节。

### DEF-030 `2026-09-29` 高层缺"带 history 低层的遥操"任务：补 `Isaac-M20-Piper-Teleop-History-v0`

| 项 | 内容 |
|---|---|
| 类型 | 特性（需求 5） |
| 状态 | 已实现并实测（2 iter EXIT=0，replay 打印 `history 窗口: 10 × 70 = 700`） |
| 关联 | `codex/ll-train-detail-fix`；`highlevel/config/high_level/hl_flat_pick_env_cfg.py:TeleopHistoryActionsCfg/TeleopHistoryEnvCfg`、`.../high_level/__init__.py` |

**1. 现象 / 需求**：高层已有 `Isaac-M20-Piper-Teleop-v0`，但它的低层 checkpoint 默认指向
**不带 history encoder** 的旧 WBC 策略（`deeprobotics_m20_wbc_flat/2026-09-18_01-31-58`），
没有一个"默认就走历史自适应低层"的遥操任务。

**2. 根因 / 现状核对**：`TeleopLLAction` **本体早就支持** history 回放
（`build_history_window` 读 `policy_layout.json`，`kind=="history"` 时建 10×70 环形窗口、
`run_low_level_policy` 走双输入 `forward(policy_obs, history_flat)`）。
缺的只是**注册一个默认指向 history checkpoint 的任务**，避免每次都要靠环境变量手动指。

**3. 修正**：新增 `TeleopHistoryActionsCfg`（`policy_path` 默认
`logs/rsl_rl/history_adaptation/2026-09-20_00-50-31/exported_deploy/policy.pt`，
可用 `RL_TRAINING_LOW_LEVEL_POLICY_TELEOP_HISTORY` 覆盖；观测模板沿用
`WBCObservationsCfg().policy` = 83 维含 `ee_goal`）+ `TeleopHistoryEnvCfg`（仅 actions 不同，
`decimation=4` 等全部继承），并注册 `Isaac-M20-Piper-Teleop-History-v0`。

**4. 结果**：2 iter 冒烟 EXIT=0，日志三行关键自检：
`低层 policy action_dim=16, 低层 obs 维度=83 (checkpoint 期望 83)`、
`该 checkpoint 是 history(ROA) 策略：actor 输入宽度 115 = policy_obs 83 + latent 32`、
`history 窗口: 10 × 70 = 700`。原有 `Isaac-M20-Piper-Teleop-v0` 同时回归 EXIT=0（不受影响）。

### DEF-029 `2026-09-29` 多地形任务：随机粗糙（噪声 0.01~0.05）+ 正/反斜坡 + 平地

| 项 | 内容 |
|---|---|
| 类型 | 特性（需求 3） |
| 状态 | 已实现（cfg 级验证通过）；**端到端冒烟/训练待换机器**（见 DEF-031） |
| 关联 | `codex/ll-train-detail-fix`；`velocity/mdp/terrains.py:ROUGH_SLOPES_FLAT_TERRAINS_CFG`、`.../flat_env_wbc_cfg.py:RoughSlopesEnvWBCConfig(_PLAY)`、`.../deeprobotics_m20/__init__.py` |

**1. 需求**：只要随机粗糙 + 正反斜坡 + 平地三种地形，且随机粗糙噪声从 0.02~0.10 降到 0.01~0.05；
直接对标 `Rough-History-Adaptation-Deeprobotics-M20-v0`（官方 `ROUGH_TERRAINS_CFG`，含楼梯/boxes/rails/pit）
与 `Rough-WO-Stairs-History-Adaptation-Deeprobotics-M20-v0`（`NONE_STAIRS_TERRAINS_CFG`）。

**2. 修正**：

* `mdp/terrains.py::ROUGH_SLOPES_FLAT_TERRAINS_CFG`：`random_rough` 比例 0.40、
  `noise_range=(0.01, 0.05)`、`noise_step=0.01`；`hf_pyramid_slope` 0.25、`hf_pyramid_slope_inv` 0.25、
  `flat` 0.10。其余公共参数（`size/border_width/num_rows/num_cols/vertical_scale/slope_threshold`）
  与 `_COMMON_KW` 一致，便于横向对比。
* `flat_env_wbc_cfg.py::RoughSlopesEnvWBCConfig`：继承 `RoughEnvWBCConfig`（因此自动带上
  history 观测、WBC 命令、本轮的两处奖励修复），只换地形 + 显式打开地形课程
  （`terrain_generator.curriculum=True`、`max_init_terrain_level=5`）+ 按 `RoughWOStairs` 的
  步骤重开 v_x 课程（75k/100k/125k/150k 步 → ±2/±3/±4/±5）。
* 注册 `Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0` 与 `-play-v0`；
  `TERRAIN_CFGS["rough_slopes_flat"]` 便于测试脚本按名索引。

**3. 结果**：`train.py` 能解析到 `RoughSlopesEnvWBCConfig` 并完成 `__post_init__`
（之后卡在平台问题 DEF-031）；平地任务（共享同一批奖励/课程类）2 iter EXIT=0。
**未覆盖**：地形本身的可视化与训练效果。

### DEF-028 `2026-09-29` 扰动加强（push 间隔 5~10 s、x±2 / y±1、yaw±0.52）+ 课程拉长

| 项 | 内容 |
|---|---|
| 类型 | 特性（需求 4） |
| 状态 | 已实现（平地 history 任务 2 iter EXIT=0；训练结果见 §4） |
| 关联 | `codex/ll-train-detail-fix`；`velocity_env_cfg.py:EventCfg.randomize_push_robot`、`flat_env_wbc_cfg.py:WBCCurriculumCfg.disturbance_ramp` |

**1. 需求**：`randomize_push_robot` 间隔缩到 (5,10) s，速度扰动改成 vx(-2,2)、vy(-1,1)、yaw(-0.52,0.52)；
并"加一定的课程"（加强扰动必然更难训）。

**2. 修正**：

* `EventCfg.randomize_push_robot`：`interval_range_s=(5.0, 10.0)`（原 10~15），
  `velocity_range={"x": (-2.0, 2.0), "y": (-1.0, 1.0), "yaw": (-0.52, 0.52)}`（原 ±0.5/±0.5，无 yaw）。
  已核对 IsaacLab `push_by_setting_velocity` 的 key 集是 `x/y/z/roll/pitch/yaw` ⇒ `yaw` 是支持的
  （它是**叠加**到当前 root 速度上的角速度冲击）。
* `WBCCurriculumCfg.disturbance_ramp`：`base` 改成新终态幅度（课程按绝对值缩放，必须逐位一致），
  `start_scale` 0.3 → **0.2**、`num_steps` 25k → **50k**（≈2083 iter 才到全量）。
  理由：扰动幅度变大后早期更容易"一推就趴窝"，而趴窝会被 `root_height_below_minimum` 记账。

**3. 兼容性**：`EventCfg` 是所有 M20 任务共享的 ⇒ 这条改动对 flat/rough/WBC/Arm 全体生效；
`disturbance_ramp` 只挂在 WBC/History 类任务上（`_PLAY` 里被置 None，eval/play 直接用全量扰动）。

**4. 结果**：待回填（见 DONE_zh.md / TODO_zh.md P1-3）。

### DEF-027 `2026-09-29` `joint_mirror` 镜像符号错误：对角对要求 θ_fl = θ_hr，与真实镜像关系（−θ）相反 ⇒ "右后腿往右前方撇"

| 项 | 内容 |
|---|---|
| 类型 | 缺陷（奖励项符号约定） |
| 状态 | 已修（平地 history 任务 2 iter EXIT=0；训练结果见 §4） |
| 关联 | `codex/ll-train-detail-fix`；`velocity/mdp/rewards.py:joint_mirror_signed`、`.../deeprobotics_m20/rough_env_cfg.py`；原条目 `TODO_zh.md` P1-3 的 known_issues ① |

**1. 现象**：训出的模型步态不对称 —— 用户描述"**右后腿往右前方撇**"。

**2. 根因（三重独立证据，不是猜）**

原 `joint_mirror` 算 `(θ_a − θ_b)²`，隐含假设"镜像姿态里两侧关节角**相等**"。本机型不成立：

* **MJCF 关节轴**（`deep_robotics_model/M20_Piper_own/mjcf/M20_Piper_own.xml`）：
  四条腿的 hipx 轴都是 `(-1,0,0)`、hipy/knee 轴都是 `(0,-1,0)` —— 左右腿是**镜像副本**，
  不是"旋转副本"（若是旋转副本，hind 的轴应为 `Rz(π)·a = (1,0,0)`）。
* **关节限位**：`fl_hipx ∈ (-0.436, 0.611)` vs `hr_hipx ∈ (-0.611, 0.436)`（恰好取负）；
  `fl_hipy ∈ (-2.583, 2.286)` vs `hl_hipy ∈ (-2.286, 2.583)`（取负）；
  只有"绕 z 轴 180° 对称"能同时解释这两条。
* **默认姿态**（`assets/deeprobotics.py::DEEPROBOTICS_M20_PIPER_CFG.init_state.joint_pos`）：
  `f[l,r]_hipy = -0.6` vs `h[l,r]_hipy = +0.6`、`f[l,r]_knee = +1.0` vs `h[l,r]_knee = -1.0`
  —— 默认站姿本身就是"对角取负"的对称姿态。

由此得到本机型的三条镜像关系（数值可由"绕 z 转 180° 的共轭变换"直接推出）：

| 镜像对 | hipx | hipy | knee |
|---|---|---|---|
| 左/右（fl↔fr、hl↔hr） | −1 | +1 | +1 |
| 前/后（fl↔hl、fr↔hr） | +1 | −1 | −1 |
| 对角（fl↔hr、fr↔hl） | −1 | −1 | −1 |

而 cfg 里 `mirror_joints` 用的**正是对角对** ⇒ 原实现在奖励里要求 `θ_fl = θ_hr`，
把正确的镜像姿态当误差、把"两条腿往同侧掰"当最优：左前腿外展多少，右后腿就被推着往**同一侧**
（即外侧）撇多少，hipy 同理往前 → 与用户描述的"右后腿往右前方撇"完全吻合。
反查基线日志也一致：`Episode_Reward/joint_mirror` 在 s3 是 **−0.164/s**（权重只有 −0.03），
说明"镜像误差"均值确实很大（≈ 对角对的默认姿态就被判成误差）。

**3. 修正**：新增 `rewards.py::joint_mirror_signed`（`(θ_a − s·θ_b)²`，
`mirror_signs` 按**关节名后缀**逐个给符号，不依赖 `find_joints` 的返回顺序），
并把 `deeprobotics_m20` 的 `joint_mirror` 换成它：mirror_joints 扩到 **4 对**
（对角 2 对 + **左右 2 对**，后者才是压住"单侧后腿外撇"的那一对），
`mirror_signs = [{-1,-1,-1}, {-1,-1,-1}, {-1,+1,+1}, {-1,+1,+1}]`，
权重 −0.03 → **−0.06**（4 对 ⇒ 每对等效 −0.015，与原来 2 对 × −0.03 同量级）。
保留原 `joint_mirror`（未删）以免影响仓库外引用。

**4. 结果（"符号反了"的定量对账）**

把默认姿态（= 天然的对称站姿）代进两条公式，**不用跑仿真**就能对账：

| 关节（fl 对 hr） | fl | hr | 旧公式 (θ_fl−θ_hr)² | 新公式 (θ_fl−(−1)·θ_hr)² |
|---|---|---|---|---|
| hipx | 0.0 | 0.0 | 0 | 0 |
| hipy | −0.6 | +0.6 | **1.44** | 0 |
| knee | +1.0 | −1.0 | **4.00** | 0 |
| 每对合计 | | | **5.44** | **0** |

旧公式 × 权重 0.03 ⇒ **每条 env 每步恒定 −0.1632**（还有重力门控 ≈ ×1）。
实测基线 run 的 `Episode_Reward/joint_mirror`（= 每秒速率）在 s3 是 **−0.1639**、
末 1000 是 **−0.1605** —— 与"整项都在惩罚默认站姿"的预测**吻合到 0.4%**。
也就是说策略只能靠**把腿掰成反对称**（一侧外撇）来减掉这笔税，这正是用户看到的撇腿。

新公式在默认姿态上恒等于 0；实测新 run 的 `Episode_Reward/joint_mirror` 前 15 iter 是
−0.0017 ~ −0.0020（旧实现同迭代在 −0.03 量级），量级降了一个多数量级。

**训练侧验收（用 `probe_gait_symmetry.py`，同代 1000 iter / 命令 (1.0,0,0) / 256 envs / 400 步）**：

| 指标 | 旧代码 | 第一版惩罚(重) | **软化版** |
|---|---|---|---|
| 后腿左右不对称 `y_hl+y_hr` (m) | −0.0702 | +0.0287 | **−0.0099** |
| 前/后轮距 (m) | 0.503 / 0.462 | 0.415 / 0.476 | **0.459 / 0.458** |
| `hl~hr` 镜像 RMS（hipx/hipy/knee） | 0.252/0.154/0.522 | 0.316/0.207/0.487 | **0.172/0.115/0.325** |
| `fr~hl` 镜像 RMS（hipx/hipy/knee） | 0.172/0.104/0.464 | 0.149/0.224/0.365 | **0.108/0.105/0.220** |

⇒ **后腿左右不对称 7.0 cm → 1.0 cm、前后轮距差 4.1 cm → 0.1 cm、镜像 RMS 全线下滑**，
用户的"右后腿往右前方撇"在符号修正后基本消失（完整表见 `DONE_zh.md` 第七节 §4）。

### DEF-026 `2026-09-29` 静止伫立时底盘仍以 ~0.15 m/s 前向漂移：零速命令下**没有任何**速度惩罚 + 站姿占比只有 2%

| 项 | 内容 |
|---|---|
| 类型 | 缺陷（奖励/课程缺项，需求 1） |
| 状态 | 已修 + 已实测（训练结果见 §4） |
| 关联 | `codex/ll-train-detail-fix`；`velocity/mdp/rewards.py:stand_still_vel_l2`/`stand_still_wheel_vel_l2`、`velocity/mdp/curriculums.py:ramp_command_param`、`velocity_env_cfg.py:RewardsCfg`、`flat_env_wbc_cfg.py:WBCRewardsCfg/WBCCurriculumCfg`；工具 `scripts/reinforcement_learning/rsl_rl/eval_fixed_command.py` |

**1. 现象（先量化，再改）**

用现成的固定命令探针（`eval_fixed_command.py`，命令钉死在 (0,0,0)、(0.5,0,0)、(1.0,0,0)，
每个 checkpoint 一个进程）跑部署基线 `history_adaptation/2026-09-20_00-50-31/model_19999.pt`：

| 命令 (vx,vy,wz) | ep_len | `err_vel_xy` | 摔倒率 |
|---|---|---|---|
| **(0,0,0)** | 894.5 | **0.1478 m/s** | 0.1332 |
| (0.5,0,0) | 929.0 | 0.1583 | 0.0901 |
| (1.0,0,0) | 912.5 | 0.1920 | 0.1055 |

即：**命令为零时底盘仍以 ~0.148 m/s 前向漂移**，而且"站着"的摔倒率比"跑着"还高（13.3%）。
复现：`python scripts/reinforcement_learning/rsl_rl/eval_fixed_command.py --headless --num_envs 512
--steps 1100 --commands "0,0,0;0.5,0,0;1.0,0,0" --checkpoint <run>/model_19999.pt`

**2. 现状盘点（"站立"相关的奖励项与课程项）**

| 项 | 位置 | 基线里的值 | 作用 |
|---|---|---|---|
| `stand_still`（`stand_still_joint_deviation_l1`） | `rough_env_cfg.py` | 被 `FlatEnvWBCConfig`/`RoughEnvWBCConfig` 置 **0** ⇒ 关 | 零命令时惩罚关节偏离默认姿态 |
| `stand_still_without_cmd` | `velocity_env_cfg.py` | **0**（从未启用） | 同上（另一实现） |
| `wheel_vel_penalty` | `rough_env_cfg.py` | **0** ⇒ 关 | 零命令时惩罚轮速（唯一能直接压"轮子空转"的项） |
| `joint_pos_penalty`（`stand_still_scale=5`） | `velocity_env_cfg.py` | **0** ⇒ 关 | 零命令时 5× 关节偏离 |
| `hipx/hipy/knee_joint_pos_penalty` | `flat_env_wbc_cfg.py` | −0.4 / −0.1 / −0.1 | 用 `joint_pos_penalty_wbc`，但 `is_truly_still` 要求 `body_vel < 0.5` ⇒ **一漂起来这项自己就关了**（鸡生蛋） |
| `lin_vel_xy_l2_with_ang_z_command` | `velocity_env_cfg.py` | 未启用 | 只在"纯 yaw 命令"时惩罚线速度（语义不对，也不覆盖零命令） |
| `feet_contact_without_cmd` | `rough_env_cfg.py` | **+0.1** | 零命令时奖励四足触地（只奖励接触，不惩罚速度） |
| `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` | `rough_env_cfg.py` | 2.0 / 1.0 | 唯一的间接约束；但双高斯核在 \|v\|≈0.15 处已饱和，把 0.15 压到 0 只多 ~0.01/s |
| `commands.base_velocity.rel_standing_envs` | `rough_env_cfg.py` | **0.02** | 只有 **2%** 的 env 会拿到零命令 ⇒ 策略几乎没见过"站着不动" |
| 课程 | `WBCCurriculumCfg` | 无 | **没有任何课程**碰站姿占比或站立惩罚 |

**3. 修正**（两项奖励 + 两条课程，全部只按**命令**门控，不看实测速度）

* `rewards.py::stand_still_vel_l2`：`|v_xy|²·1[‖cmd_xy‖<0.1] + ω_z²·1[|cmd_z|<0.1]`；
* `rewards.py::stand_still_wheel_vel_l2`：`Σ_j ω_wheel,j²·1[‖cmd_xy‖<0.1]`（"轮子空转"那一半）；
* `curriculums.py::ramp_command_param`：新增通用"把某个 **command term** 的标量参数线性爬升"课程
  （直接改 `term.cfg.<param>`，与 `apply_range_stages` 改 `ranges` 同理、幂等）；
* `WBCCurriculumCfg` 三条课程：`rel_standing_envs` 0.02 → **0.15**（25k 步）、
  `stand_still_vel` 权重 −0.2 → **−2.0**、`stand_still_wheel_vel` −5e-05 → **−5e-04**（各 25k 步）。

**权重怎么定（两步：先估、再用实测打脸修正）**

* 【估计】`RewardManager` 返回 `Σ term·weight·dt`，`Episode_Reward/*` 记的是**每秒速率**；
  基线在命令 (0,0,0) 时"回报/秒 = 1.75"。按 \|v\|≈0.15 ⇒ \|v\|²≈0.0225 估：
  `weight=-8.0` 约 −0.18/s（~10%，看起来正合适）；轮速项按 ω≈2.5 rad/s/轮 ⇒ Σω²≈25，
  `-0.01` 约 −0.25/s（~14%）。
* 【实测修正】第一版（−8.0 / −0.01）跑 1500 iter 后，`Episode_Reward` 实测反解出
  真实量级比估计大一个数量级：静止 env 的 \|v\|≈**0.47 m/s**、ω≈**6.1 rad/s/轮**
  ⇒ 两项合计 **−0.483/s** vs 总回报 **+0.396/s**（**122%**）。后果见 §4。
* 【最终】按"合计 ≈ 10~15% 总回报"定成 `-2.0` / `-5e-04`（预估 −0.076/s ≈ 19%）。
  **底盘速度项当主力**（二次型：\|v\|=0.5 时 −0.5/s 会主动刹车、\|v\|=0.05 时 −0.005/s 不干扰微调），
  轮速项只留很小的"别空转"信号 —— 轮式倒立摆要靠轮子**微动**平衡，
  惩罚瞬时 ω² 会直接砍掉平衡作动（"噪声不是越小越好"的同源教训）。

**4. 结果**

* 改动后 `History-Adaptation-Deeprobotics-M20-v0` 2 iter 冒烟 EXIT=0，启动打印
  `RewardManager contains 23 active terms`（原 21），新增
  `stand_still_vel −0.8`、`stand_still_wheel_vel −0.001`；`CurriculumManager contains 15 terms`（原 12），
  新增 `standing_env_ratio_ramp` / `stand_still_vel_ramp` / `stand_still_wheel_vel_ramp`，
  且日志里能看到爬升确实在走（`rel_standing_envs → 0.0200 → …`、`stand_still_vel_ramp −0.80 → −0.81`）。
* **训练 A/B（第一版权重 −8.0 / −0.01，1500 iter，与同代旧代码 `model_1500` 对比）**：
  * 目标指标修好了：命令 (0,0,0) 的 `err_vel_xy` 在**两档难度**都降 22~23%
    （s0 难度 0.1153 → **0.0886**；-play- 0.1475 → **0.1157**）。
  * **但代价太大**：同代摔倒率 s0 难度 0.178 → **0.708**，终止几乎全是 `bad_orientation_2`
    （翻倒 473 次 vs 76 次）⇒ 策略学会"冻住轮子"，而轮式倒立摆靠轮子微动平衡。
  * 用日志反解出量级：iter 1234 两项惩罚合计 **−0.483/s**，而同一步 `Σ Episode_Reward`
    只有 **+0.396/s** —— 惩罚 = 总回报的 **122%**（我第一版的估算严重偏低，见下）。
* **修正（第二版）**：按"两项合计 ≈ 总回报 10~15%"重新定标 ⇒
  `stand_still_vel` −0.2 → **−2.0**、`stand_still_wheel_vel` −5e-05 → **−5e-04**；
  **底盘速度项当主力**（它就是用户看到的现象），轮速项只留很小的"别空转"信号。
* **第二版实测（同代 1000 iter 三方对比，训练任务 / 512 envs / seed 42 / 1100 步）**：

  | 命令 (0,0,0) | 旧代码 | 第一版 −8/−0.01 | **软化版 −2/−5e-4** |
  |---|---|---|---|
  | `err_vel_xy` (m/s) | 0.1030 | 0.0860 | **0.0857** |
  | 摔倒率 | 0.2481 | 0.3401 | **0.1065** |
  | 回合长度 | 890.0 | 845.6 | **920.1** |
  | 每回合回报 | 40.83 | 38.44 | **46.28** |

  ⇒ **软化版三项全赢**（漂移 −17%、摔倒 −57%、回报 +13%）；第一版是"漂移降了但摔得更狠"，
  而且到 1500 iter 时第一版继续恶化（摔倒 0.7080）⇒ 它是"越训越会冻轮子"。
  代价：0.5/1.0 m/s 两档的跟踪误差略差（需全长 run 判断能否训回来，见 TODO P1-1'）。
* 两版的完整数字、复现命令与读法见 **`DONE_zh.md` 第七节**（含 `eval_fixed_command.py` 的三方对比表）。
* **方法学教训**：只看 `Episode_Termination/bad_orientation_2` 这种**聚合**指标会得出
  "权重改了没区别"的错误结论（软化前后该曲线几乎重合，因为要求静止的 env 只占 15%、
  聚合被其它命令稀释）；**必须用固定命令把 100% 的 env 钉在同一任务上测**（`eval_fixed_command.py`）。
* **教训**（值得记住）：`Episode_Reward/*` 是**每秒速率**而不是每步量，
  单看"1.75/s 的站立回报"去估权重会低估一两个数量级 —— 定权重前一定要用
  `Episode_Reward` 的**实测值反解**（`rate/weight = 该门控子集上的物理量均值`），
  并且把新增惩罚的合计与同一步的 `Σ Episode_Reward` 比一比。

### DEF-025 `2026-09-20` 桌面版自动化唤醒必然 422：投递条目缺 `call_id`（线程被永久污染）

### DEF-025 `2026-09-20` 桌面版自动化唤醒必然 422：投递条目缺 `call_id`（线程被永久污染）

| 项 | 内容 |
|---|---|
| 类型 | 工具链缺陷（Codex 桌面版 `26.901.51231` × deepseek provider `wire_api=responses`） |
| 状态 | **未修（外部 bug，仓库侧修不了）**；两条 automation 已置 `PAUSED`，P1-1 监控改回手动 |
| 关联 | automation `p1-1-entropy-coef-a-b`（heartbeat，20 min）、`p1-1-cron`（cron，5 min 试跑）；线程 `01a0be53`（被污染）、`01a0bea6`（cron 试跑）；`TODO_zh.md` P1-1 |

**1. 现象**

* 19:33:59（heartbeat）与 19:49:29（cron 首次试跑）触发的自动化轮次**都**失败：
  `unexpected status 422 Unprocessable Entity: Failed to deserialize the JSON body into the target type:
  input: missing field \`call_id\` at line 1 column N, url: https://api.deepseek.com/responses`。
* 更严重的是**连带污染**：heartbeat 那次之后，用户在**同一线程**里的普通提问（19:36:15 那条
  "现在变体1是在运行训练吗…"）也一起 422，线程状态变成 `systemError` —— 该线程从此每一轮都发不出去。

**2. 根因（实测定位，不是猜）**

* 唤醒载荷是以 `functionCallOutput` 注入线程的，条目只有 `id` / `name` / `namespace` / `output`，
  **没有 `call_id`**：

  ```json
  {"type":"functionCallOutput","id":"fco_01a0be98-2e76-71b3-a29a-636c0929cb4f",
   "name":"automation_update","namespace":"codex_app","output":"<heartbeat>…"}
  ```

  （在 `%CODEX_HOME%/thread_history_1.sqlite` → `thread_items.item_json` 里可直接看到。）
* deepseek 的 `/responses` 严格要求 function-call-output 项带 `call_id` ⇒ **只要线程历史里存在这一条，
  之后每一轮请求都 422**（用户消息也救不回来）。
* **反证**（排除"用了 automation 工具就会坏"）：同一线程更早的 3 条 `mcpToolCall`
  （ordinal 968 / 997，创建与查看 automation）之后线程仍然正常，19:12 那轮是**带着它们**成功的；
  真正让线程报废的是 19:33 注入的那条 `functionCallOutput`（ordinal 1075）。
* **heartbeat 与 cron 同病**：cron 换了全新线程（`01a0bea6`，历史里只有注入条目）照样第一次就 422
  ⇒ 不是"某个线程被写坏"，而是**投递格式对这套 provider 必然失败**。

**3. 处置（本次做了什么）**

* 两条 automation 都置 `PAUSED`：`p1-1-entropy-coef-a-b`（heartbeat）、`p1-1-cron`（cron 试跑），
  避免每 20 min / 5 min 继续刷失败任务、继续把线程写坏。
* **训练本身不受影响**：变体 1（`2026-09-20_18-54-34_cap_noise_std`）照常在 GPU 上跑；
  受影响的只是"跑完自动通知 + 自动回填"。
* P1-1 收尾改**手动**（命令见 `TODO_zh.md` P1-1）；被污染的线程 `01a0be53` 弃用（内容仍可读），
  后续在**新任务**里继续。

**4. 复发条件 / 待办**

* 换 Codex 桌面版新版本、或换 provider 后再试：先建一条 5 min 的 cron 试跑，**看首轮是否 200**
  （成功标志：`%CODEX_HOME%/sqlite/codex-dev.db` 的 `automation_runs.status` 不再是失败、
  且新线程能正常回话），确认后才把间隔改长。
* 本环境若还想要"跑完提醒"，只能走**仓库外**手段（本地看门狗脚本 / Windows 计划任务），
  不要再依赖 automation —— 建 automation 反而会把目标线程写死。

### DEF-024 `2026-09-20` 探索噪声上界可配置（`max_noise_std`）+ P1-1 的 A/B 设计

| 项 | 内容 |
|---|---|
| 类型 | 特性（训练配置开关）+ 实验设计 |
| 状态 | **代码已完成并冒烟验收；A/B 训练进行中**（结果见本节"4. 结果"补记 / `TODO_zh.md` P1-1） |
| 关联 | `rsl_rl/rsl_rl/modules/actor_critic_history.py`（`max_noise_std` / `clamp_noise_std_`）、`rsl_rl/rsl_rl/algorithms/ppo_roa.py`（step 后投影）、`.../deeprobotics_m20/agents/rsl_rl_ppo_cfg.py:RslRlPpoActorCriticHistoryCfg`；上游归因 `DEF-023` |

**1. 需求（为什么做）**

`DEF-023` 把 P1-1 归因成"`log_std` 无上界 + `entropy_coef` 的熵奖励 ⇒ `Policy/mean_noise_std`
顶到 ~1.5 的平台，同时 adaptive 调度把学习率压到 1e-5"。要验证这个归因、并给"精度上界被压住"
一个可选的解，需要一个**能开能关**的探索噪声上界，且必须能在不改代码的情况下做 A/B。

**2. 实现**

* `ActorCriticHistory(..., max_noise_std=0.0)`：`0`/`None` = 不限制（**旧行为，默认**）；
  正数 = 上界。`log` 型噪声在上界处转成 `log(max_noise_std)`。
* `clamp_noise_std_()`：把噪声参数**投影**回 `[0, max_noise_std]`。调用点两处：
  ① `__init__` 末尾（`init_noise_std > 上界` 时第 0 迭代就生效）；
  ② `PPORoA.update` 里每次 `optimizer.step()` 之后（投影梯度）。
  另外 `_update_distribution` 里采样用的 std 也 clamp 一次 —— **只在采样处 clamp 不够**：
  参数本身会沿熵奖励一路爬到无界，`Policy/mean_noise_std` 就还是"一直在涨"，看不出真实行为。
* cfg 字段 `RslRlPpoActorCriticHistoryCfg.max_noise_std: float = 0.0`，用 hydra 直接覆盖：
  `python scripts/reinforcement_learning/rsl_rl/train.py --task <task> --headless agent.policy.max_noise_std=1.2`。
* **踩坑（写下来免得再踩）**：这里**不能**声明成 `float | None = None` —— IsaacLab 的
  `update_class_from_dict` 是按**当前值的类型**校验覆盖值的
  （`value is None or isinstance(value, type(obj_mem))`，见 `isaaclab/utils/dict.py`），
  默认 `None` 时 `agent.policy.max_noise_std=1.2` 会直接报
  `[Config]: Incorrect type under namespace: /policy/max_noise_std. Expected: <class 'NoneType'>`（实测撞上）。

**3. 冒烟验收（改完先验证"封顶链路真的会封顶"）**

| 命令（64 envs × 2 iter） | 观测 | 结论 |
|---|---|---|
| `agent.policy.max_noise_std=0.05` | `Policy/mean_noise_std` = **0.05**（iter 0）/ 0.04999 | 上界**低于** `init_noise_std=1.0` 时被投影下来 ⇒ `__init__` + 采样 + step 后投影三段都生效 |
| `agent.policy.max_noise_std=1.2` | `Policy/mean_noise_std` = 1.0 / 0.9995 | 未越界时不干预（无副作用） |
| 默认（`0.0`）跑两个低层任务 | EXIT=0 | 旧行为不变 |

run 目录：`logs/rsl_rl/history_adaptation/2026-09-20_18-52-54`（cap=0.05）、`2026-09-20_18-53-32`（cap=1.2）。

**4. 结果（A/B 实测）——2026-09-22 收尾**

设计（同一 seed=42、4096 envs、同任务，只改一个变量，与现有基线逐迭代对比）：

| 组 | run 目录 | 改了什么 | 迭代数 |
|---|---|---|---|
| 基线 | `2026-09-20_00-50-31` | ——（`entropy_coef=0.01`，无上界） | 20000 |
| **A（cap）** | `2026-09-20_18-54-34_cap_noise_std` | `agent.policy.max_noise_std=1.2` | 4000 |
| **B（ent）** | `2026-09-20_22-13-37_ent_coef_low` | `agent.algorithm.entropy_coef=0.002` | 4000 |
| **B′（ent=0，意外点）** | `2026-09-20_19-30-43_ent_coef_low` | `agent.algorithm.entropy_coef=0.0`（本意 0.002，命令行被设成了 0） | 4000 |

判定口径（写死，避免事后挑指标）：`Policy/mean_noise_std` 平台 ≤1.2（cap）或显著低于基线（ent）
**且**同迭代点的 `Train/mean_reward`、`Train/mean_episode_length` 不劣于基线 **且**
s3 段（iter ≥3125）的"合计摔倒"（`bad_orientation_2 + root_height_below_minimum`）不高于基线。

**运行经验（同一台 A4000，16 GB）**：两个 Isaac 训练**同时**跑会把每个的 collection time
从 ~1.9 s 抬到 ~7.6 s/iter（互相拖累，总吞吐也不划算）⇒ **串行跑**；
机器有其他负载时单跑也可能只有 ~5.5 s/iter（实测 19:00 前后）。

**口径修正（重要，先说）**：基线是 **20000 iter** 的 run，它的"s3 阶段均值"覆盖
iter 3125–19999，而三个 4000-iter 变体只覆盖 3125–3999 ⇒ **阶段均值不可比**
（基线后期还在继续变好，用全长均值会**低估**基线在 s3 起点附近的摔倒率）。
本节所有判定数字都改在**统一迭代窗口**上取：s3 = iter **3125–3999**、末段 = iter **3000–3999**，
直接对 `.summary_cache.npz` 按窗口重算（命令见本节末尾）。

**结果（统一窗口，s3 = iter 3125–3999 的阶段均值）**

| 组 | `Policy/mean_noise_std` | `Train/mean_reward` | `Train/mean_episode_length` | `Loss/learning_rate` | 合计摔倒（s3） |
|---|---|---|---|---|---|
| 基线（ent 0.01，无上界） | 1.405 | 23.93 | 858.2 | 4.37e-04 | 0.1938 |
| **A cap=1.2（ent 0.01）** | **1.052** | **38.52** | **905.2** | 3.00e-05 | **0.1315** |
| B ent=0.002（无上界） | 0.4209 | 36.92 | 878.4 | 2.03e-04 | 0.1827 |
| B′ ent=0.0（无上界，意外点） | 0.1256 | 34.36 | 856.8 | 1.02e-04 | 0.2223 |

**末段复核（iter 3000–3999，结论一致）**

| 组 | `mean_reward` | `mean_episode_length` | 合计摔倒 |
|---|---|---|---|
| 基线 | 27.19 | 872.0 | 0.1751 |
| **A cap=1.2** | **40.58** | **914.7** | **0.1188** |
| B ent=0.002 | 38.83 | 890.9 | 0.1635 |
| B′ ent=0.0 | 36.68 | 870.2 | 0.2005 |

**结论（按写死的口径逐条判）**

1. **A（`max_noise_std=1.2`）通过，建议作为默认**：噪声被压在 **1.052** ≤1.2 ✓；
   `mean_reward` **+61%**（38.5 vs 23.9）、`mean_episode_length` **+5.5%**（905 vs 858）都不劣 ✓；
   s3 合计摔倒 **0.132 vs 0.194（−32%）** ✓ —— 三条全过，且在四组里 reward / ep_len / 摔倒
   **同时最好**。机制：上界把 `log_std` 投影在 ~1.05，而基线同期一路爬到 **1.44**；
   s3 的摔倒尖峰（0.045→0.19）正来自这一段噪声抬升（DEF-023 的归因被 A/B 证实）。
2. **B（`entropy_coef=0.002`）也通过**：0.42 / +54% / +2.4% / 摔倒 −5.7%，
   但三项都略逊于 A（摔倒 0.183 vs 0.132）⇒ 同为有效修法，优先级排在 A 之后。
3. **B′（`entropy_coef=0`，意外点）只挂摔倒**：噪声 0.126、reward +44%、ep_len −0.2% 都满足，
   但 s3 合计摔倒 **0.222 vs 0.194（+15%）** ⇒ **噪声压到 ~0.13 会反而更容易摔**。
   这条是本次最有信息量的**负面**证据：**"奖励更高"≠"更稳"、"噪声越小越好"不成立**，
   存在中间最优区（本组数据里 ~1.0 的 A 最好，0.42 的 B 次之，0.13 的 B′ 最差）。
4. **遗留（未解）**：三组的 `Loss/learning_rate` 在 s3 仍然偏低，A 甚至低到 **3e-5**
   （基线 4.4e-4）⇒ "adaptive 调度把 LR 压到地板"这条机制**没有被 cap 解决**；
   A 的收益（reward / `error_vel_xy` 0.477 vs 0.891）并非来自 LR。LR 这条要单独立项
   （候选：非 adaptive 调度、调 `desired_kl`）。
5. **部署前建议**：A 只在 4000 iter 上验过，默认值落地前跑一次 **20k 全长**（同 seed=42 / 4096 envs /
   `agent.policy.max_noise_std=1.2`）＋固定命令 eval，再改 cfg 默认。

**复现命令**（三条 run 的对比表，缓存命中后 <1 s；把 `--run` 换成上表任一变体）：

```
python scripts/reinforcement_learning/rsl_rl/summarize_run.py \
  --run logs/rsl_rl/history_adaptation/2026-09-20_18-54-34_cap_noise_std \
  --baseline logs/rsl_rl/history_adaptation/2026-09-20_00-50-31 \
  --derive "合计摔倒=Episode_Termination/bad_orientation_2+Episode_Termination/root_height_below_minimum" \
  --tags mean_noise_std --tags error_vel_xy --tags 合计摔倒 \
  --tags Train/mean_reward --tags Train/mean_episode_length --tags "Loss/learning_rate" \
  --grid 1000,2000,2500,3000,3500,4000
```

统一窗口重算（本节 §4 两张表的数字就是它出的；`logs/...` 里换成四个 run 目录）：

```python
import numpy as np, os
runs = ["2026-09-20_00-50-31", "2026-09-20_18-54-34_cap_noise_std",
        "2026-09-20_22-13-37_ent_coef_low", "2026-09-20_19-30-43_ent_coef_low"]
tags = ["Policy/mean_noise_std", "Train/mean_reward", "Train/mean_episode_length",
        "Loss/learning_rate", "Episode_Termination/bad_orientation_2",
        "Episode_Termination/root_height_below_minimum"]
for lo, hi in [(3125, 3999), (3000, 3999)]:            # s3 窗口 / 末段窗口
    print("window", lo, hi)
    for r in runs:
        d = dict(np.load(os.path.join("logs/rsl_rl/history_adaptation", r, ".summary_cache.npz"),
                         allow_pickle=True))
        vals = []
        for t in tags[:4]:
            x = d[t]; m = (x[:, 0] >= lo) & (x[:, 0] <= hi); vals.append(x[m, 1].mean())
        a = d[tags[4]]; b = d[tags[5]]; m = (a[:, 0] >= lo) & (a[:, 0] <= hi)
        vals.append((a[m, 1] + b[m, 1]).mean())
        print(r, ["%.4g" % v for v in vals])
```

**成本**：A 本机 A4000 / 4096 envs / 4000 iter ≈ **6.0 h**（5.5 s/iter）；B、B′ 在另一台机器 ≈ **2.3 h**
（2.06 s/iter）。除 cfg 覆盖外无其它改动，`--num_envs` / `--max_iterations` / `--seed` 全部一致。

### DEF-023 `2026-09-20` P1-1「训练退化」归因：`noise_std` 是**饱和平台**、`error_vel_xy` 是**命令课程漂移**

| 项 | 内容 |
|---|---|
| 类型 | 诊断（训练质量 / 指标口径） |
| 状态 | 部分已修（归因已定，A/B 修法未跑 —— 见 `TODO_zh.md` P1-1） |
| 关联 | `scripts/reinforcement_learning/rsl_rl/summarize_run.py`（本条目新增）；run `2026-09-20_00-50-31`（A）vs `2026-09-19_09-02-50`（B）；`rsl_rl/rsl_rl/algorithms/ppo_roa.py`、`.../deeprobotics_m20/agents/rsl_rl_ppo_cfg.py:HistoryAdaptationPPORunnerCfg` |

**1. 现象**

TODO P1-1 原来写的是"`mean_noise_std` 1.0→~1.49、`error_vel_xy` 0.38→~0.89 长期退化"。
本次用 `summarize_run.py` 把 20k 迭代按**课程阶段**（s0<1042、s1<2083、s2<3125、s3 之后）
和采样网格聚合后，发现这两条曲线**形状完全不同**，不能并称"退化"：

| 指标 | run A s0 / s1 / s2 / s3 | A 末 1000 | run B s0 / s1 / s2 / s3 | B 末 1000 |
|---|---|---|---|---|
| `Policy/mean_noise_std` | 0.973 / 1.009 / 1.132 / **1.473** | 1.477 | 1.236 / 1.332 / 1.404 / **1.505** | 1.507 |
| `Metrics/base_velocity/error_vel_xy` | 0.525 / 0.385 / 0.457 / **0.851** | 0.891 | 0.154 / 0.274 / 0.456 / **0.768** | 0.769 |
| `Train/mean_reward` | 24.8 / 34.6 / 43.8 / **22.9** | 23.6 | 0.98 / 4.52 / 13.96 / **17.26** | 17.83 |
| `Train/mean_episode_length` | 921 / 960 / 968 / **910** | 917 | 216 / 455 / 710 / **777** | 768 |
| `Episode_Termination/root_height_below_minimum` | 0.0258 / 0.0208 / 0.0275 / **0.1181** | 0.1151 | — | — |
| `Metrics/ee_pose/orientation_error` | 0.341 / 0.325 / 0.317 / **0.852** | 0.864 | 0.844 / 0.991 / 0.957 / **0.955** | 0.964 |

复现命令（A/B 一次出表，缓存命中后 <1 s）：

```powershell
python scripts/reinforcement_learning/rsl_rl/summarize_run.py `
  --run logs/rsl_rl/history_adaptation/2026-09-20_00-50-31 `
  --baseline logs/rsl_rl/history_adaptation/2026-09-19_09-02-50 `
  --derive "合计摔倒=Episode_Termination/bad_orientation_2+Episode_Termination/root_height_below_minimum" `
  --tags mean_noise_std --tags error_vel_xy --tags 合计摔倒 `
  --tags Train/mean_reward --tags Train/mean_episode_length --tags orientation_error
```

**2. 根因**

① `noise_std` 不是发散而是**有界平台**：A 在 iter≈5000 就进入 1.43~1.49 的抖动带
（逐点 1.432/1.486/1.481/1.493/1.479/1.483/1.487/1.477），B 更早（iter=1000 已 1.278）
并停在同一高度 1.51 —— **换掉课程/EE 改动都一样**，说明它是 optimizer 的平衡点，不是任务变难。
机制（读码确认，不是猜）：`actor_critic_history.py:log_std` 是**无上界**自由参数，
而 `ppo_roa.py:238` 的 `loss = surrogate + value_loss_coef*value_loss - entropy_coef*entropy`
持续给熵**正奖励**（`entropy_coef=0.01`），`Loss/entropy` 实测 21.5→28.1 一路涨到平台；
同时 `schedule="adaptive"`/`desired_kl=0.01` 因 KL 超标把 `Loss/learning_rate` 一路压到地板
（s3 均值 2.3e-4、最低触到 **1e-5**）。⇒ **高熵 + 低学习率**：策略分布被撑宽、精度上界被压住，
但训练本身没有崩（reward/ep_len 都不降）。

② `error_vel_xy` 上升**与课程阶段同形**：B（当时**没有** EE 课程/root_height 改动）在同一批
阶段上从 0.15 → 0.77 单调上升，形状与 A 一致 ⇒ 主导因素是 `base_velocity` 命令范围随课程
放宽（终值 vx ±5 m/s），**绝对**速度误差天然变大；A 的 s3 反而比"本该更容易"的直觉更好：
末 1000 reward **23.6 vs 17.8**、ep_len **917 vs 768**、合计摔倒 **0.122 vs 0.331**。
⇒ 用"绝对 `error_vel_xy`"跨阶段/跨 run 判优劣是**错的口径**（这条就是 P1-1 原来的误判来源）。

③ P1-2 的剩余摔倒**与臂相关**（新证据）：A 的 `root_height_below_minimum` 只在 s3 抬头
（0.021~0.027 → 0.118），**同期** `ee_pose/orientation_error` 从 0.32 抬到 0.85
（= s3 才放开臂的大范围摆动）；而高度跟踪的稳态偏差仍只有 1.3~1.8 cm
（`height_error_bias_steady` 末 0.013）⇒ 是**臂摆动时倾覆**，不是高度控制失效。

**3. 修正**

本条目只改**口径 + 归因**（工具 `summarize_run.py`），不改训练配置；配置侧的候选修法已经
收敛成两条，写进 `TODO_zh.md` P1-1（要 A/B）：
① 给 `log_std` 加上界（`max_noise_std` 之类，目标平台 ≤1.2）；
② 调低 `entropy_coef` 0.01 → 0.005/0.002（直接削弱把 std 撑大的那一项）。
原来的"调 `body_*_rew_s3` 的 num_steps / 查 `track_lin_vel_xy_exp` 权重"两条**降级**：
实测 `Curriculum/body_pitch_rew_s3|body_roll_rew_s3` 在 iter≈3125 就到达终值 0.8、
`body_height_rew_s2` 在 s1 就到 0.8，与 `noise_std` 平台、`error_vel_xy` 抬升**不同期**，
不构成解释。

**4. 结果（验收）**

* 工具：`summarize_run.py` 新增 run 后首次解析 70 MB 事件文件 ≈ 30~50 s，之后走
  `<run>/.summary_cache.npz`（按事件文件 size+mtime 失效）<1 s；输出阶段均值表 +
  采样网格表 + 两 run 对比（步进取值）。
* 归因结论：`noise_std` = 有界平台（A/B 同形，Δ末 = −0.035）；`error_vel_xy` = 任务变难的
  口径产物（B 同形）；P1-2 的剩余摔倒与 EE 姿态误差同期、与高度稳态误差无关。
* 未覆盖 / 风险：`Metrics/*` 是"复位那一刻"的均值，抖动大 —— 本条目全部结论都建立在
  **阶段均值**上，单点读数不作证据；**没有**跑新的 A/B 训练（要 GPU 时间，见 TODO P1-1）。

### DEF-022 `2026-09-20` 部署基线没固化：交接 prompt 的 main 哈希滞后 + 产物无指纹

| 项 | 内容 |
|---|---|
| 类型 | 文档 / 交付物（部署可回溯性） |
| 状态 | 已修 |
| 关联 | `main @ 2d49f47`；`docs/review/DONE_zh.md` 第六节；`docs/review/NEXT_SESSION_PROMPT.md`；run `logs/rsl_rl/history_adaptation/2026-09-20_00-50-31` |

**1. 现象**

① `NEXT_SESSION_PROMPT.md` 的【当前状态】写的是 `main = e78d479`，但实际 main 已经走到
`2d49f47`（`093be1a`/`79b1626`/`75bcd63`/`7458672`/`2d49f47` 五个提交之后）—— 下一个
session 按 prompt 里的哈希去 checkout 会拿到**旧代码**（不含 ONNX 导出、部署文档、探针）。
② "当前拿去部署的代码 + 对应哪个 run 的策略"这件事只散落在 DONE 第五节的两行里，
没有**单一入口**，也没有产物指纹：`exported_deploy/policy.pt|onnx|policy_layout.json`
改了/重导了没法判断。
③ 训练代码与部署代码是否同一份，**没有核对过**：run 里的
`git/loco-manip-unified-rl-agent.diff`（rsl_rl 启动时 dump）显示该 run 是在分支
`codex/ll-height-stability @ 96e1b66` 上训的，工作区还有一处未提交改动。

**2. 根因**

prompt 里的哈希是**人肉回填**的（`093be1a` 那条提交就叫"回填 NEXT_SESSION_PROMPT 的 main
哈希"），main 一动就滞后；"部署基线"从来没有被定义成一条**不可漂移**的记录，
所以没人知道该拿哪个 commit 去部署。

**3. 修正**

① `DONE_zh.md` 新增**第六节「部署基线」**：基线 commit、对应 run、checkpoint、
四个产物的 sha256、接口契约，并写清"基线一旦记录就不再漂"；
② `NEXT_SESSION_PROMPT.md` 的【当前状态】改成"部署基线 = `2d49f47`（见 DONE 第六节）"，
并加一条踩坑说明（prompt 里的哈希会滞后 ⇒ 以 DONE 第六节为准）；
③ 核对训练代码 vs `main`：run 的未提交改动是**注释掉死代码** `FKReachableEECommand`
（无 task/配置引用），`git diff 96e1b66 main -- <低层路径>` 只剩占位符清理
（`body_names=""`→`None`）、`stance_width=float`→数值（weight=0）、新增启动自检
`check_policy_layout`、注释/文档 —— **无动力学与观测布局变化**。

**4. 结果（验收）**

* `git rev-parse HEAD` = `2d49f47c46c4f136b47ab3983a4a690761464741`；
  `git rev-list --left-right --count origin/main...main` = `0 0`（基线已推送）。
* `Get-FileHash -Algorithm SHA256`：`policy.pt 43C63D19…F6510`、
  `policy.onnx 77757542…B8D1`、`policy_layout.json 7E3B11EB…A7947`、
  `model_19999.pt 592A50D6…AC14`（完整值见 DONE 第六节）。
* `policy_layout.json` 自述 `source_run/source_checkpoint/source_iteration = 19999`、
  `policy_obs 83 / history 10×70 / latent 32 / action 16 / opset 17 / 相对误差 1.87e-07`
  ⇒ 与上面 run 一一对应，可回溯。
* 基线用 **annotated tag `deploy-baseline-2026-09-20` → `2d49f47`** 固定（已 push 到 origin），
  部署机可以直接 `git checkout deploy-baseline-2026-09-20`，不必记哈希。

### DEF-021 `2026-09-20` 部署交接：sim2sim/sim2real 参考文档 + 部署规格探针

| 项 | 内容 |
|---|---|
| 类型 | 交付物（部署文档 + 工具） |
| 状态 | 已完成（已并入 main） |
| 关联 | `7458672`；`docs/deploy_sim2sim_sim2real_zh.md`、`scripts/reinforcement_learning/rsl_rl/probe_deploy_layout.py` |

**需求**：策略要在**另一台电脑**上先做 sim2sim（MuJoCo）、再转 sim2real。需要一份"改部署脚本时的
对照手册"，把"必须照抄的接口"和"必须自己实现的部分"写死，避免靠猜。

**关键事实（本轮实测，写进文档当权威）**：

1. **三种关节顺序互不相同**：动作序（12 腿 fl,fr,hl,hr + 4 轮）、
   **articulation 原生序**（观测 `joint_pos/joint_vel` 用的 24 维：0-3 是四个 hipx、
   4 是 arm1、5-8 hipy、9 是 arm2、10-13 knee、14 是 arm3、15-18 四个轮、19-21 arm4-6、
   22-23 夹爪）、MuJoCo MJCF 序（每腿 hipx/hipy/knee/wheel 连续 + 臂 + 夹爪）。
   历史上 `known_issues #1`/DEF-016 就是这里出的错。
2. **动作增益不能用内部 `_scale` 张量推断**：探针改成"把动作置 1.0 看关节目标"才测准 ——
   结论是 hipx **0.125**、其余腿关节 **0.25**、轮子速度 **5.0**（默认角偏移另计）。
3. **history 与 policy_obs 里同名项不一样**：history 是**原始值**（`base_ang_vel` 不乘 0.25、
   `joint_vel` 不乘 0.05），且 `joint_pos` 24 维**含轮子不置零**；policy_obs 侧则乘了 scale、
   且 `joint_pos` 的轮子列被置零。
4. **机械臂不由策略动作驱动**：`ee_ik` term 的 `action_dim = 0`，臂由 50 Hz 的
   DLS IK（λ=0.01、绝对位姿、`arm_joint1..6` → `gripper_base`）从 `ee_pose` 命令驱动 ⇒
   部署侧必须自己实现 IK，否则臂不动且观测语义崩。
5. **MuJoCo 侧已有可用模型**：`deep_robotics_model/M20_Piper_own/mjcf/M20_Piper_own.xml`，
   关节轴与 URDF 一致（无符号翻转）；实测默认姿态下 `gripper_base` 相对 `base_link`
   位置 `(0.3492, 0, 0.4326)` vs Isaac `(0.3492, 0, 0.4327)` —— 差 1e-4，运动学对齐。
   但 MJCF 自带 `timestep=0.002`（Isaac 是 0.005），需要显式处理控制周期。
6. **命令终值**（课程跑满后）：`base_velocity` vx (-5,5)/vy (-1,1)/wz (-1,1)、
   `body_pose` height (0.33,0.55)/pitch ±0.35/roll ±0.25、`ee_pose` 球坐标 l (0.30,0.52) 等；
   且 `body_pose.height` 的度量是 `root_z − mean(四轮 z) + 0.09`（不是 root 绝对 z）。

**产出**：① 探针 `probe_deploy_layout.py`（在部署机上跑一次即打印关节序/默认角/限位/
动作增益/观测逐项 scale·clip·noise/关键 body 下标/默认姿态几何）；
② 文档 `docs/deploy_sim2sim_sim2real_zh.md`（接口契约 83/700/16、动作→关节映射、
命令来源与坐标系、IK 复刻要点、MuJoCo 建模参数、sim2sim 七步上线顺序、
sim2real 差异清单、失败模式对照表、权威文件清单）。

### DEF-020 `2026-09-20` 部署态导出增加 ONNX（+ 自检阈值必须按输出幅值归一）

| 项 | 内容 |
|---|---|
| 类型 | 特性（部署工具链） |
| 状态 | 已完成（已并入 main） |
| 关联 | `708ca53`；`scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py`、`docs/train_history_flat_zh.md` |

**现象 / 需求**：`export_deploy_policy.py` 原来只出 TorchScript（`policy.pt`）；
sim2sim 与真机侧要的是 **ONNX**。另外 `2026-09-20_00-50-31` 这个 run 实际训到
**iter=19999**（不是先前以为的 15000），导出应该取最新 checkpoint。

**根因 / 要点**：① ONNX 导出必须与 `policy.pt` **同接口同数值**，否则"换格式"等于换策略；
② 校验判据不能用固定绝对阈值 —— 低层策略输出**没有归一化**（实测 `|a|max = 183.6`），
纯 fp32 舍入就已有 `3.4e-05`，按 `1e-5` 的绝对阈值会把**正确的导出判成失败**（第一次跑就撞上了）；
③ `history_flat` 是 700 维展平向量，"最旧→最新"的顺序如果拼反，ONNX 不会报错、只会静默算错。

**修正**：① 新增 `--onnx/--no-onnx`（默认开）与 `--opset`（默认 17）；
② 自检三步：`onnx.checker.check_model` → onnxruntime 与 TorchScript 在**同一组**随机输入上的
**相对**误差（`max|Δ| / max(1, |ref|max) < 1e-5`）→ batch=1/5 的动态维验证；
③ `policy_layout.json` 增补 `onnx` 段（文件名/opset/输入输出名与形状/相对误差/tolerance/runtime）
与 `history_order`、`history_note`（每步 70 维的构成 + reset 后整窗填满同一帧 + 谁维护缓冲）。
json 改成**最后写**，保证里面记录的结论都是"已经验过"的。

**结果**：run `2026-09-20_00-50-31` 用 `model_19999.pt` 重新导出
`exported_deploy/{policy.pt, policy.onnx(981 KB), policy_layout.json}`：
scripted↔eager **0.000e+00**、与 `ActorCriticHistory.act_inference` **0.000e+00**、
ONNX↔TorchScript 相对误差 **1.87e-07**（绝对 3.43e-05 / 输出幅值 183.6）；
独立复核（另取随机输入、直接 `onnxruntime` + `torch.jit.load` 对比）B=1/3/8 相对误差
**1.5e-07 ~ 2.7e-07**；ONNX 图：opset 17、`policy_obs['batch',83]` + `history_flat['batch',700]`
→ `action['batch',16]`，batch 维动态。

### DEF-019 `2026-09-20` 合并 ⑦⑧ 与 R1：同一批 `__init__` 的两侧改写（冲突解法）

| 项 | 内容 |
|---|---|
| 类型 | 缺陷（合并冲突 / 重构耦合） |
| 状态 | 已修（已并入 main） |
| 关联 | `e9edc34`（⑦⑧）、`2c5a85a`（R1）、合并提交 `af4602d`/`07601e9`；`mdp/low_level_policy_action.py`、`mdp/low_level_replay.py`、`mdp/pre_trained_nav_action.py` |

**现象**：把 `codex/hl-ckpt-params`（⑦ checkpoint 路径参数化 + ⑧ 懒加载低层 cfg）与
`codex/hl-replay-base-class`（⑤ 的 R1：抽 `LowLevelPolicyActionBase`、迁移 nav）先后合进 main 时冲突：
① `low_level_replay.py` 两侧都在文件同一处**追加了一段**（⑦ 的 `resolve_policy_path`/
`load_low_level_policy` vs history 回放的 `history_single_step_ll`/`run_low_level_policy`/
`build_history_window`），git 把它们当成同一 hunk；
② `pre_trained_nav_action.py` 的 `__init__`：⑦ 侧是"老式类 + 内联加载 + 就地改观测 cfg"的完整函数体，
R1 侧把这整段删掉换成 `_build_low_level_obs_cfg()` 钩子；
③ 更隐蔽的一条：R1 新加的基类文件里**内联**复刻了一份 `check_file_path + torch.jit.load`，
而 ⑦ 把它抽成了 `load_low_level_policy` —— 这条不会报冲突（文件在 ⑦ 那侧根本不存在），
只能靠代码里的注释（"合并后可直接替换"）发现。

**根因**：两条分支都从 `codex/hl-fix-ee-command` 派生的同一段代码上做"同位置结构改写"：
⑦ 是**横切**所有 action term 的载入段，R1 是**纵切**这批 `__init__` 的骨架；
两个方向都改同一批行 ⇒ 文本合并必然失败，且失败原因与"哪一侧更正确"无关。

**修正**（按 TODO P0-1 的既定顺序，先 ⑦⑧ 再 R1）：
① `low_level_replay.py` 取**并集**：import 行合并（`base_ang_vel/joint_pos_rel/joint_vel_rel/
projected_gravity` + `check_file_path/read_file`），文件尾部保留两段（history 回放段 + ⑦ 段），
并补回被 marker 吃掉的 `# ---` 分隔行；
② `pre_trained_nav_action.py` 取 R1 侧（基类写法）——R1 已经把 nav 的加载/布局/观测/tick
全部上移进基类，老式函数体是**重复**实现；
③ `low_level_policy_action.py`（基类）：删掉内联加载与"兼容用的 `load_low_level_policy_inline`"，
统一调用 `load_low_level_policy(cfg.policy_path, env, tag=...)`；顺手把 `verify_low_level_layout`
的 `policy_layout_json=self._policy_layout_json` 补上（R1 注释里预留的那一步，可多打印
"actor 输入 = policy_obs + latent"）。

**结果**：合并后 4 个高层任务 2 iter 全 EXIT=0、启动打印的 obs 维度与 checkpoint 一致
（Pick-Flat 83、Pick-WBC/Teleop 76、Nav 69）、reward 与分支上一致（1.11/1.28/0.15/10.25）：
即"⑦ 的统一报错"与"R1 的单一骨架"两件事同时生效，没有一边被另一个 merge 吃掉。

### DEF-018 `2026-09-20` 合并高层分支时的 Windows 大小写路径冲突（`next_session_prompt.md`）

| 项 | 内容 |
|---|---|
| 类型 | 缺陷（工具链 / 跨平台路径） |
| 状态 | 已修（已并入 main） |
| 关联 | 合并提交 `129848e`；修复 `30d5411`；`docs/review/next_session_prompt.md`、`docs/review/NEXT_SESSION_PROMPT.md` |

**现象**：合并高层链后 `git status` 报 `docs/review/NEXT_SESSION_PROMPT.md` 被修改（内容变成分支上那份
旧 prompt），而磁盘上只剩 `next_session_prompt.md`；随后 `git commit -- <该路径>` 又**把内容写进了错误的那个
路径**（提交里 `next_session_prompt.md` 被"改写"成 main 的版本，而不是被删除）。

**根因**：Windows 文件系统大小写不敏感 —— `NEXT_SESSION_PROMPT.md`（main 侧新增）与
`next_session_prompt.md`（分支侧新增）是**同一个物理文件**，但 git 索引是大小写敏感的，
于是同一条路径在索引里出现两份；`git checkout` 按分支那侧写盘后，索引里 main 的那份就被判成"被修改"。
而 `git commit -- <pathspec>` 是按**路径**去工作区取内容，在大小写不敏感的文件系统上解析到了另一个文件，
所以它提交的是"内容替换"而不是"删除"。

**修正**：① 把分支那份从索引里摘掉（`git rm --cached docs/review/next_session_prompt.md`，
不动工作区文件），再 `git checkout HEAD -- docs/review/NEXT_SESSION_PROMPT.md` 恢复 main 的内容；
② 路径解析搞错的那次提交用 `git reset --soft <上一个 merge commit>` 回退后重提（只动本地提交，
工作区不变）；③ 后续 5 份旧文档改用不带 pathspec 的普通提交删除。

**结果**：合并后 `docs/review/` 只剩 TODO/DONE/DEFECT_LOG/NEXT_SESSION_PROMPT（`30d5411`、`0772757`），
`git status` 干净、`git ls-tree HEAD docs/review` 只有正确的大小写。教训：在 Windows 上合并
"两边各自新增、只差大小写"的文件时，**不要用 `git commit -- <path>`**，先清索引再 `git add -A`。

### DEF-017 `2026-09-18` 三条独立缺陷：手臂奖励坐标系 / privileged 缓存 / legacy 权重

| 项 | 内容 |
|---|---|
| 类型 | 缺陷 |
| 状态 | 已修 |
| 关联 | `b3496a5`、`905c2df`；`velocity/mdp/arm_rewards.py`、`velocity/mdp/observations.py`、`velocity_env_cfg.py` |

**现象**：① 手臂奖励用 `pose_command_b`（root 系）与世界系 EE 位姿比较，目标非零 yaw 时误差完全错误
（旧算法把奖励算成 0.0000，正确值 0.0010）；② `privileged_*` 观测"第 2 次调用后永久缓存"，
而 `randomize_actuator_gains` 是 reset 模式 → 第 2 个 episode 起 `gain_scale` 过期；
③ 手臂 env 零动作稳态实测 `Στ²≈2.4e4`、`Σq̈²≈8.7e6`，
legacy 注释里的 `arm_joint_acc=-1e-5` 每步贡献 -87 → 总奖励 -86.2，训练信号被压死。

**根因**：① 坐标系不统一；② 缓存没区分 startup/reset 随机化；
③ 权重是按"数值量级"拍的，没标定；且所有臂奖励乘 `arm_weight`，而 `ArmWeightCommand`
`init_max_weight=0.0` 且课程被注释掉 → 臂奖励长期是死的。

**修正**：① 统一在 root 系比较（`_ee_pose_root_frame`）+ EE body 索引缓存，并删掉
`ee_position_tracking` 里 `return` 之后的死代码；② 抽出 `_PrivilegedCachedTerm`，
按 `update_on_reset` 只在对应随机化是 reset 模式时刷新（并为被 reset 的行刷新）；③ 权重重标定为
`torque=-1e-5 / vel=-1e-3 / acc=-1e-8`。

**结果**：① 奖励数值正确；② @4096 envs 每次 reset 净增 1.92 ms（摊销 0.0019 ms/step），
每步缓存路径 0.165 ms 不变；③ 总奖励回到 +0.53。
顺带发现的坑：`ObsTerm.params` 会被 manager 原样透传给 `__call__`，自定义开关必须 `pop` 掉。

### DEF-016 `2026-09-18` `joint_pos_rel_without_wheel` 的索引空间混用（清错关节）

| 项 | 内容 |
|---|---|
| 类型 | 缺陷 |
| 状态 | 已修（+ 后续两次加固） |
| 关联 | `b3496a5`（低层）、`dc45d0e`（replay 侧）、`6d22006`/`fef34a7`（断言） |

**现象**：`joint_pos` 观测"把轮关节置零"作用到了错误的关节上：实测被清的是
`['hr_wheel_joint','arm_joint1','arm_joint2','arm_joint3']`，放行的是 `fl/fr/hl_wheel`。

**根因**：`asset_cfg.joint_ids` 是"列 → articulation 原生 id"的映射，而
`wheel_asset_cfg.joint_ids` 是**原生 id**；用后者直接索引前者，在 `preserve_order=True`
的重排列下必然错位。低层（原生序 24 维）恰好"看起来对"，replay 侧列序是 leg→wheel→arm
（22 维）就暴露了 —— replay 实测列下标应是 `[12,13,14,15]`，而它用了原生 id `[15,16,17,18]`。

**修正**：① 低层把 policy 的 `joint_pos/joint_vel` 回到 `[".*"]`（原生序 24 维，掩码自然正确，
代价：观测 22→24 维，旧 checkpoint 作废）；② replay 侧改用按**列**置零的
`joint_pos_rel_without_wheel_columns()`，并在 `verify_wheel_columns()` 里做启动期断言；
③ `joint_pos_rel_without_wheel` 里补"列序 == 原生序"的精确断言
（对每个要清零的原生 id `c`，要求列映射 `asset_ids[c] == c`，否则报错并指向列版本）。

**结果**：被清关节纠正为 `fl/fr/hl/hr_wheel`；三个高层任务 + 4 个低层任务回归 EXIT=0。

### DEF-015 `2026-09-19` checkpoint 路径硬编码 + 模块级低层 cfg 实例化

| 项 | 内容 |
|---|---|
| 类型 | 缺陷（可维护性/可移植性） |
| 状态 | 已修（在分支 `codex/hl-ckpt-params`，**未合并 main**） |
| 关联 | `1b6d5c8`；`low_level_replay.py`、`high_level_env_cfg.py`、4 个高层 cfg |

**现象**：高层 cfg 里 4 处写死带时间戳的 `logs/.../policy.pt`（`logs/` 被 .gitignore），
换机器或清一次 logs → 所有高层任务起不来；报错只有一句 `Policy file ... does not exist.`。
另外 `high_level_env_cfg.py:30` 在 import 期就 `DeeproboticsM20RoughEnvCfg()`（内部 deepcopy 全部嵌套 cfg），
且 `render_interval = 低层 decimation(4) < 高层 decimation(40)` → 每个 env step 渲染 10 次（IsaacLab 给 WARNING）。

**根因**：路径与低层参数没有参数化通道；"只为拿 dt/decimation 就建一份低层 cfg"。

**修正**：`resolve_policy_path()`（环境变量 `RL_TRAINING_LOW_LEVEL_POLICY_<KEY>` 优先，
命中打印提示；命令行仍可 hydra 覆盖）+ `load_low_level_policy()` 统一报错
（给出环境变量/hydra/重新导出三种修法）；`LOW_LEVEL_ENV_CFG` 改懒加载单例；
`render_interval` 改成等于高层 `decimation`。

**结果**：`Rendering step-size` 0.02→0.2、警告消失、reward 不回归（0.88→1.28 与改前一致）；
无效路径给出三种修法；有效覆盖（指到另一个 83 维 checkpoint）83=83 通过。

### DEF-014 `2026-09-19` 高层 replay 不支持带 history encoder 的低层策略

| 项 | 内容 |
|---|---|
| 类型 | 特性（缺失能力） |
| 状态 | 已修（分支 `codex/hl-replay-history`，**未合并 main**） |
| 关联 | `0d37c99`；`highlevel/mdp/low_level_replay.py`、3 个 action term |

**现象**：`play.py` 导出的 `policy.pt` 只有 actor，输入 108/115 维（含 32 维 latent），
latent 由 `history_encoder` 从 10 步历史算出 → 数值能跑但不是训练出来的策略；
replay 侧遇到 `policy_layout.json` 的 `kind=history` 直接 `NotImplementedError`。

**根因**：回放侧缺"10 步历史窗口"的维护；且 `last_action` 若用 `env.action_manager.action`
会拿到**高层**动作（11/12 维）而不是低层 16 维动作，语义错位。

**修正**：新增 `history_single_step_ll()`（与训练函数逐项一致，只差 last_action 来源）、
`LowLevelReplayState`（复位检测 + 低层动作缓存清零 + 复用 IsaacLab `CircularBuffer` 的 10 步窗口）、
`run_low_level_policy()`（按 layout 单/双输入分支）；三个 action term 改为
`on_tick() → compute_group → run_low_level_policy`。顺带修掉"用
`episode_length_buf == 0` 判复位"的旧写法（该条件在复位后的整步内都为真 → 整步每 tick 都清零）。

**结果**：窗口最后一帧 vs 用**训练函数**独立复算 `0.000e+00`；整窗顺序（含复位填充）
40 次 tick `0.000e+00`；teleop / Pick-WBC（用 history checkpoint）2 iter EXIT=0；
两个旧 checkpoint 回归无变化。

### DEF-013 `2026-09-19` 观测/动作布局变化会静默让旧 checkpoint 失效（⑯）

| 项 | 内容 |
|---|---|
| 类型 | 缺陷（静默失效） |
| 状态 | 已修（`dc45d0e` replay 侧 + `fef34a7` 低层侧） |
| 关联 | `dc45d0e`、`fef34a7`；`low_level_replay.py`、`velocity/mdp/observations.py`、`velocity_env_cfg.py` |

**现象**：`mdp.last_action` 观测宽度 = 动作总维度 ⇒ 任何 action term 维度变化都会改变 policy
观测布局。实测：IK 从普通 action term（7 维）改成 `CommandDrivenIKAction`（0 维）后，
高层 replay 第一次 `env.step()` 抛 `mat1 and mat2 shapes cannot be multiplied (Nx76 and 83x512)`。

**根因**：没有任何启动期校验；错误要等到 matmul 才暴露，且报错信息与"布局"无关。

**修正**：replay 侧新增"布局单一来源 + 启动期打印 + 与 checkpoint 严格比对"
（`low_level_replay.verify_low_level_layout`）；低层侧新增
`mdp.check_policy_layout`（挂在 `EventCfg` 的 `mode="startup"`）打印每个观测组的逐项维度、
每个动作项维度、动作总维度，并断言 `actions` 槽位宽度 == `action_manager.total_action_dim`。

**结果**：flat 83=83、WBC/teleop 86=86；低层启动打印 `policy 86 = … + actions22`、
`history 700`；不一致时直接 RuntimeError 而不是 matmul。

### DEF-012 `2026-09-19` 6 份重复的 replay 实现 + nav 奖励项读不存在的 `ll_command`

| 项 | 内容 |
|---|---|
| 类型 | 重构 + 缺陷 |
| 状态 | R1 已完成（分支 `codex/hl-replay-base-class`，**未合并 main**）；policy/openvla 迁移见 TODO P2 |
| 关联 | `2c5a85a`；`highlevel/mdp/low_level_policy_action.py`、`pre_trained_nav_action.py` |

**现象**：6 个 action term 各抄一份"关节名单 / last_action 闭包 / 低层 obs 就地覆写 / tick 循环"；
`PreTrainedNavAction` 没有 `ll_command` 属性，而 nav 的 `lateral_velocity_penalty`(-0.5) /
`angular_velocity_penalty`(-0.2) 会读它 → 一读就 `AttributeError`。

**根因**：缺少公共基类；`ll_command` 只在 3 个 term 上实现。

**修正**：新增 `LowLevelPolicyActionBase`（载入策略 / 布局与观测 / 低层 tick 状态（含 history）/
调策略 / 路由低层 action term / `ll_command`+`ll_command_w`），迁移 nav（并把 nav 的
22 关节低层观测模板改成在 **cfg 层**显式声明，消掉"就地改 cfg"）；
`pre_trained_policy_action` / `openvla_pick_action` 先补 `ll_command` 接口（完整迁移待做）。

**结果**：nav 2 iter EXIT=0、`低层 obs 69 = checkpoint 期望 69`、动作分块 12/4/0、
轮关节列下标 `[12,13,14,15]`；`lateral_velocity_penalty = -0.0750`、
`angular_velocity_penalty = -0.0275`（以前直接崩）。

### DEF-011 `2026-09-19` 高层三条 P0：`ll_command` 缺失 / IK 目标写错字段 / `ee_goal` 世界系

| 项 | 内容 |
|---|---|
| 类型 | 缺陷 |
| 状态 | 已修（分支 `codex/hl-fix-ll-command`、`codex/hl-fix-ee-command`，**未合并 main**） |
| 关联 | `e064bc6`、`7a22759`；`pre_trained_pick_action.py`、`low_level_replay.py`、`highlevel/mdp/rewards.py` |

**现象**：① 普通 pick teacher 第一次 `env.step()` 抛
`AttributeError: 'PreTrainedPickAction' object has no attribute 'll_command'`；
② 高层给机械臂的目标写进了 `pose_command_w`，而 IK 读 `pose_command_b`
（父类 `_update_metrics` 也读 w，让人误以为目标生效）→ 机械臂一直在跟 `ee_pose` 自己采样的随机目标；
③ replay 喂给低层的 `ee_goal` 是**世界系**（含机器人世界坐标），训练时是 root 系。

**根因**：三处接口/坐标系不统一；`ll_command` 只在部分 term 上实现。

**修正**：① 补 `ll_command`/`ll_command_w`，并把 8 处"拿命令位置与物体世界坐标比较"的奖励项
统一改用 `ll_command_world()`（对 WBC/teleop 取值完全一致）；② 统一写 `pose_command_b`，
并同步 `pose_start_b/pose_end_b`（否则 `_update_command` 会用 start/end 插值覆盖，指标与 marker 又回到自采样值）；
③ 统一用 root 系的 `ll_command[:, 3:10]`。

**结果**：① 三任务 reward 与改前逐位一致；② `|pose_command_b − 高层root目标| = 0.000e+00`；
③ 低层 obs 的 `ee_goal` 槽位 vs root 系命令 `0.000e+00`（对照世界系 8.9093）。

### DEF-010 `2026-09-19` 低层动作 scale 在 replay 里硬编码（实测是死代码）

| 项 | 内容 |
|---|---|
| 类型 | 缺陷 + 实测纠偏 |
| 状态 | 已修（分支 `codex/hl-replay-layout`，**未合并 main**） |
| 关联 | `dc45d0e`；`pre_trained_*_action.py`、`low_level_replay.py` |

**现象**：6 处硬编码 wheel 速度 scale（低层训练 5.0，replay 20.0，看起来不一致）。

**根因/实测结论**：这些事后赋值**本来就是死代码** —— `JointAction.__init__` 会把
`cfg.scale/clip/joint_names` 编译成内部张量，之后 `term.scale = 20.0` 只是新增一个没人读的实例属性。
实测：`_wheel_vel_action_term._scale = 5.0`（来自 cfg），而 `__dict__["scale"] = 20.0`。

**修正**：删掉全部事后赋值，scale/clip/joint_names 一律从传入的低层 action cfg 读取，
并在 `check_low_level_action_cfgs()` 里校验与布局一致。

**结果**：replay 与训练的 scale 来源统一（5.0）；不再有"看着不一致其实是死代码"的误导。

### DEF-009 `2026-09-19` `__init__` 里就地改传入的 low-level obs cfg

| 项 | 内容 |
|---|---|
| 类型 | 缺陷（共享状态被改写） |
| 状态 | 已修（`dc45d0e`） |
| 关联 | `dc45d0e`；`pre_trained_pick_action.py`、`pre_trained_pick_wbc_action.py` |

**现象/根因**：action term 在 `__init__` 里直接改 `cfg.low_level_observations.actions.func/params`，
甚至 `cfg.low_level_observations = WBCObservationsCfg().policy`。cfg 在同一 env 内多个 term
间共享时互相覆盖，且"回放观测"与"训练观测"的差异被藏在这些赋值里。

**修正**：`build_low_level_observation_group()` 对模板 `deepcopy` 后再覆写；
WBC 观测模板由 `HLFlatPickWBCActionsCfg` / `TeleopActionsCfg` 在 cfg 层显式提供。

**结果**：三个高层任务 2 iter EXIT=0；模板复用不再互相污染。

### DEF-008 `2026-09-19` 清单外 A：高层喂给低层 policy 的 `actions` 观测少 7 维

| 项 | 内容 |
|---|---|
| 类型 | 缺陷 |
| 状态 | 已修（`dc45d0e`） |
| 关联 | `dc45d0e`；`low_level_replay.LowLevelActionLayout` |

**现象**：`Isaac-Deeprobotics-High-Level-Pick-Flat-Teacher-v0` 第一次 `env.step()` 抛
`RuntimeError: mat1 and mat2 shapes cannot be multiplied (Nx76 and 83x512)`
（比 DEF-011 的 ① 更早触发）。

**根因**：replay 用 `_ee_ik_action_term.action_dim` 决定 `actions` 观测宽度，
IK 改成 `CommandDrivenIKAction` 后该值为 0 → `actions` 只有 16 维，而旧 checkpoint 训练时是 23 维。

**修正**：把"产生 checkpoint 那次训练"的布局显式写进 `LowLevelActionLayout`
（`ee_action_dim`，可由 cfg 覆盖），`actions` 观测按 `[leg | wheel | ee_ik]` 原样拼接，
`ee_ik` 槽位取低层 policy 上一帧输出（IK 仍由 CommandManager 驱动）。

**结果**：flat 76→83、WBC/teleop 79→86，与 checkpoint 期望一致。

### DEF-007 `2026-09-19` 高层清单 ⑥（就地改 cfg）与 ⑨（引用不存在的观测项）

| 项 | 内容 |
|---|---|
| 类型 | 缺陷 |
| 状态 | ⑥ 已修 `dc45d0e`；⑨ 已修 `dc45d0e`（该类仍未启用，完整迁移见 TODO P2） |
| 关联 | `dc45d0e`；`pre_trained_policy_action.py`、`low_level_replay.py` |

见 DEF-009（⑥）与 DEF-014 的"引用不存在的观测项"部分（⑨：`ee_pose_commands` → `ee_goal`，
切片同步改 `[:, 3:10]`）。

### DEF-006 `2026-09-20` `root_height_below_minimum` 的"记账陷阱" + 稳态高度误差被塌陷污染

| 项 | 内容 |
|---|---|
| 类型 | 缺陷（指标误读 + 任务设计） |
| 状态 | 已修（`7ff5b86`，已并入 main） |
| 关联 | `7ff5b86`（原 `96e1b66`）；`flat_env_wbc_cfg.py`、`curriculums.py`、`commands.py`、`probe_root_height_termination.py` |

**现象**：用户 20k run 里 `bad_orientation_2` 只有 0.7%，但 `root_height_below_minimum` **0.349**；
同一日志 `Metrics/body_pose/height_error_bias` 长期 +0.10~0.21 m，看起来像"机器人系统性蹲低 10~20 cm"。

**根因（实测）**：① 两项是**记账迁移**（旧 run 30° 阈值 `0.624+0.015=0.639`；新 run 45.8°
`0.007+0.349=0.356` ⇒ 总摔倒率其实降 44%，趴窝改由高度项记账）；
② 高度项抓的是**真摔**（触发瞬间 `root_z` 均值 0.188、最小 0.125，实际高度比命令低 0.345 m，
倾角只有 6.1% 超 45.8°），不是"蹲得低"；③ `height_error_bias` 的均值被塌陷瞬间（±0.35 m）拉高，
稳态其实只有 +0.02~0.03 m。

**修正**：① 阈值 0.30 **不动**（反事实：0.26 只把 20s 触发率 25.8%→24.4%）；
② EE 课程 s0 锚点从"默认（举起）位姿"改成**低位锚点**（实测锁低位 1.0% vs 锁默认 55.5% vs 无课程 25.8%）；
③ `body_pose.height_range` 上界 0.60→0.55（0.60 够不到且是摔倒率最高的桶）；
④ push/外力扰动从 30%→100% 课程（25k 步）；⑤ 新增 `height_error_bias_steady`（裁剪 ±0.15 m）。

**结果**：新 run `2026-09-20_00-50-31`（15k iter）`root_height` 0.361→**0.122**、
合计摔倒 0.371→**0.132**、ep_len 805→883、reward 15.4→22.7（与旧 run 同迭代对比）；
`height_error_bias_steady` = 1.1 cm。

### DEF-005 `2026-09-19` EE 姿态命令在重采样瞬间跳变

| 项 | 内容 |
|---|---|
| 类型 | 缺陷 |
| 状态 | 已修（`469fbd4`，已并入 main） |
| 关联 | `469fbd4`（原 `9ccb8ec`）；`velocity/mdp/commands.py` |

**现象**：`HeightInvariantEECommand._update_command` 位置按 `T_traj` 插值，但**姿态直接取终点四元数**
→ 每 5 s 重采样时机械臂姿态参考瞬时跳变（`o_yaw` 范围 ±π，跳变可近 180°），关节速度/力矩尖峰打到底盘。

**根因**：姿态没做插值。

**修正**：新增批量版 `quat_slerp_batch`（最短路径 + 无副作用 + 近平行退化走 lerp），
`_update_command` 姿态改 slerp。**不能直接用** `isaaclab.utils.math.quat_slerp`：
它用 `torch.dot` + `if tau == 0.0` 判断，只支持单个四元数，而且会**就地修改输入**
（探针实测确认 q2 被翻转）。

**结果**：与官方单样本实现最大分量偏差 1.19e-07；受控测试（T_traj=1s、dt=0.02s、夹角 155.7°）
单步姿态跳变 **155.7° → 3.12°**。

### DEF-004 `2026-09-19` `bad_orientation_2` 阈值过紧 + 早期"一动臂就终止"

| 项 | 内容 |
|---|---|
| 类型 | 缺陷（任务设计/终止阈值） |
| 状态 | 已修（`45f9e74` + `469fbd4`，已并入 main） |
| 关联 | `45f9e74`、`469fbd4`；`velocity/mdp/events.py`、`velocity_env_cfg.py`、`flat_env_wbc_cfg.py` |

**现象**：6~7 月的 run 该终止只有 0.6~3.3%，9 月起 62~78%；早期（500 iter）高达 96.6%，
episode 平均只有 184 步。

**根因**：① 旧实现 `(g_z>0) | (\|g_xy\|>0.5).any(-1)` 等价于"绕单轴倾斜约 30°"，且边界是方形
（沿 x/y 30°、沿对角 45°）；② 命令空间 `body_pose` pitch ±20°/roll ±14° 叠加已 24.5°，
而实测跟踪误差本身有 7.5~21.6° ⇒ **正常跟踪误差会被判成摔倒**；
③ 机械臂"扰动底盘的能力"变了（资产/执行器 300/20/EE 参考系），早期一直处在"一动臂就终止"的区间。

**修正**：① `bad_orientation_2` 改成旋转不变的总倾角 `acos(-g_z) > limit_angle`，
默认 0.8 rad(45.8°)，并在 `DoneTerm` 里显式给参数；② 加 EE 目标课程（s0 锚点 + s1/s2/s3 区间阶梯）。

**结果**：该终止从 0.62 → **0.007~0.011**；机制对照见 DEF-006。

### DEF-003 `2026-09-19` `ee_goal` 观测的坐标系（root vs world）

见 DEF-011 的 ③（高层 replay 侧统一到 root 系）。

### DEF-002 `2026-09-19` IK 目标写进死字段（`pose_command_w`）

见 DEF-011 的 ②。

### DEF-001 `2026-09-19` `PreTrainedPickAction` 缺 `ll_command`

见 DEF-011 的 ①。
