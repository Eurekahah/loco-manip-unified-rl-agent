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

---

# 记录（新→旧）

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
