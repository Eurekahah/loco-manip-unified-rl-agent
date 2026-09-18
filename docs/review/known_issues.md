# 低层与工程性隐患清单（记录用，不含修改）

基线：`main @ 905c2df`。行号会漂移，请以符号名为准。
high_level 侧的问题单独放在 `high_level_todo.md`。

图例：`[已修]` = 已在 main 或分支上修复；`[待办]` = 尚未处理；`[确认]` = 需要进一步定位。

---

## 一、正确性

### 1. `[已修]` `joint_pos_rel_without_wheel` 的索引空间混用

- 原问题：`asset_cfg.joint_ids` 是"列 → articulation 原生 id"的映射，而 `wheel_asset_cfg.joint_ids`
  是原生 id；用后者直接索引前者会在 `preserve_order=True` 的重排下清错关节
  （实测会清掉 `hr_wheel + arm_joint1/2/3`，放行 `fl/fr/hl_wheel`）。
- 现修法：`rough_env_cfg.py:352-353` 那两行被注释掉，policy 的 `joint_pos/joint_vel` 回到
  `[".*"]`（原生序、24 维），掩码自然正确。
- 仍在的隐患（建议后续加一行注释或断言）：函数本身仍**依赖调用方保证"列序 == 原生序"**。
  可加：
  ```python
  if len(asset_cfg.joint_names) == len(asset_cfg.joint_ids):
      masked = [asset_cfg.joint_names[c] for c in wheel_asset_cfg.joint_ids]
      assert masked == list(wheel_asset_cfg.joint_names), masked
  ```
- 注意：该修复把 policy 观测从 22 维变成 24 维（多了 2 个 gripper 关节），
  与"修改观测和动作维度，先前的checkpoint不能使用"那次提交一致 —— 旧 checkpoint 需要重训。

> 后续（`codex/hl-replay-layout`）：高层 replay 侧已经**实际踩到同一个坑并被修掉** ——
> replay 的关节列序是 leg → wheel → arm（非原生序），却用
> `wheel_asset_cfg.joint_ids`（原生 id `[15,16,17,18]`）去索引重排后的列，
> 实测清掉的是 `['hr_wheel_joint','arm_joint1','arm_joint2','arm_joint3']`。
> 现在 replay 侧改用按**列**置零的 `joint_pos_rel_without_wheel_columns()`，
> 并在 `verify_wheel_columns()` 里做启动期断言。
> 本函数自身那条断言仍待办（见 #16 末尾），因为它依赖调用方保证"列序 == 原生序"。

### 2. `[已修]` privileged 观测缓存对 reset 模式随机化不敏感

- 原问题：`privileged_*` 用"第 2 次调用后永久缓存"，但 `randomize_actuator_gains`
  是 `mode="reset"` → 第 2 个 episode 起 `gain_scale` 就是过期值。
- 现修法（`velocity/mdp/observations.py`）：抽出 `_PrivilegedCachedTerm`，
  只有"对应 reset 模式随机化"的项才刷新（当前仅 `privileged_joint_gain_scale`），
  刷新只更新被 reset 的行；可用 ObsTerm 参数 `update_on_reset` 覆盖。
- 实测 @4096 envs：每次 reset 净增 **1.92 ms**（摊销 0.0019 ms/step），
  每步缓存路径 0.165 ms 不变。
- 顺带发现的坑（已写进代码注释）：`ObsTerm.params` 会被 manager 原样透传给 `__call__`
  （`isaaclab/managers/observation_manager.py:549`），任何自定义开关都必须 `pop` 掉，
  否则 `TypeError: __call__() got an unexpected keyword argument`。

### 3. `[已修]` 手臂奖励的坐标系不一致（root 系 vs 世界系）

- 原问题：`arm_rewards.py` 拿 `pose_command_b`（root 系）去和世界系 EE 位姿比较；
  yaw/位置非 0 时误差完全错误（实测旧的错误算法直接把奖励算成 0.0000，正确值 0.0010）。
- 现修法：统一在 root 系比较（`_ee_pose_root_frame`），并给 EE body 索引加了缓存。
- 顺带修掉 `ee_position_tracking` 里 `return` 之后的死代码。
- 注：姿态误差本身与参考系无关，改坐标系只影响位置项。

### 4. `[已修]` legacy 手臂奖励权重不可用（实测标定）

- 零动作稳态实测：`Στ²≈2.4e4`、`Σq̈²≈8.7e6` → legacy 注释里的 `arm_joint_acc=-1e-5`
  每步贡献 **-87**，总奖励 -86.2，训练信号被完全压死。
- 现修法（手臂 env）：`torque=-1e-5`、`vel=-1e-3`、`acc=-1e-8` → 总奖励回到 +0.53。
- 背景：原实现里所有臂奖励都乘 `arm_weight`，而 `ArmWeightCommand` 的 `init_max_weight=0.0`
  且 `advance_arm_weight` 课程被注释掉 → 臂奖励长期是死的（`disable_zero_weight_rewards`
  只看 weight，看不出这种"有权重但恒 0"的项）。

### 5. `[待办]` `body_names=""` / `joint_names=[""]` 占位符会直接抛异常

- 例如 `velocity_env_cfg.py:636/655/683/690/712/722/731/740/757/768/780/826`
  里的 `SceneEntityCfg("robot", body_names="")`、`joint_names=[""]`
- 实测：`resolve_matching_names([""], ...)` 抛
  `Not all regular expressions are matched!`；本次做手臂 env 时因为子类没触发
  `disable_zero_weight_rewards()` 而真的炸了一次（`feet_air_time_variance`）。
- 建议：占位符改成 `None`；或让"空串=全选"成为显式约定。

### 6. `[待办]` `disable_zero_weight_rewards()` 的两处脆弱

- 位置：`velocity_env_cfg.py:918-924`
- `reward_attr.weight` 直接取属性：若某项已被置 `None` → `AttributeError: 'NoneType'`。
- 调用被 `if self.__class__.__name__ == "XXX":` 守卫（`rough_env_cfg.py:493`、
  `flat_env_cfg.py:31` 等）→ 新加子类会静默跳过清理，行为随继承深度变化。
- 建议：改成显式白名单/黑名单 + 对 `None` 容错。

### 7. `[待办]` `feet_distance_y_exp` 的 `"stance_width": float`

- 位置：`velocity_env_cfg.py:781`（`feet_distance_xy_exp` 的注释里同样）
- 传的是**类型对象**而不是数值；一旦权重非 0 打开该项就会炸。

### 8. `[待办]` `action_mirror` / `action_sync` 的索引空间与关节名

- 位置：`velocity/mdp/rewards.py:373-429`
- 用 articulation 关节 id 去索引 `env.action_manager.action`（按 term 拼接的向量），
  仅在两者顺序一致时成立；
- `action_sync` 的关节名（`FR_hip_joint` 等）在本机器人上不存在；
- 当前两者 weight=0（会被 `disable_zero_weight_rewards` 置 None），属于"埋着的雷"。

### 9. `[待办]` `joint_mirror` 用"平方差"做镜像惩罚

- 位置：`velocity/mdp/rewards.py:351-371`，配置见 `rough_env_cfg.py:443-447`
- 左右对称关节（如 hipx）符号约定通常相反，用差值惩罚可能鼓励了非镜像姿态。
  建议按具体关节符号约定确认（hipx 用和、hipy/knee 用差）。

### 10. `[待办]` `arm_rewards.py` 里的 `grasp_success` / `ee_approach_object`

- 依赖不存在的 `object` 实体（低层场景没有物体），属于 dead code；
  建议删除或标注为"仅在带物体的场景使用"。

### 11. `[待办]` `HeightInvariantEECommand` 覆盖 `_update_command` 时没调 `super()`

- 位置：`velocity/mdp/commands.py:191-208`；后果：`pose_command_w` 永远不更新，
  父类的 goal-pose 可视化一直画在原点/单位姿态（`commands.py:429-434`），
  也是 high_level replay 写 `pose_command_w` 无效的原因之一（见 `high_level_todo.md` #2）。

### 12. `[待办]` reset 后第一帧 `ee_pose` 命令是全 0

- `_resample_command` 只写 `pose_end_b`，`pose_command_b` 要到第一次
  `command_manager.compute()`（即第一次 `env.step()`）才被插值更新。
- 后果：reset 后的第一帧观测/奖励看到一个"零位姿"目标（IK 版本同样如此）。
- 建议：在 `reset()` 里同步一次 `_update_command()`，或让 `command` 返回 `pose_end_b` 的初值。

### 13. `[待办]` `UniformThresholdVelocityCommand._resample_command` 的阈值是空操作

- 位置：`velocity/mdp/commands.py:35-40`，`* (norm > 0.0)` 恒真（官方实现是 `> 0.2`）。

### 14. `[待办]` EE 目标碰撞检查会静默降级 + 硬编码包围盒

- 位置：`velocity/mdp/commands.py:394-404`（重采样 `max_resample_attempts` 次后接受碰撞目标）、
  `commands.py:470-474`（AABB 范围针对特定安装位置写死）。

### 15. `[待办]` 硬编码常数散落

- 站立高度 `0.513`：`rough_env_cfg.py:408`、`flat_env_wbc_cfg.py:332/374`、
  `pre_trained_pick_wbc_action.py:175/230`、`BodyPoseCommandCfg` 默认值
- 轮半径 `0.09`：`velocity/mdp/utils.py:19`
- `ee_offset_z=0.135`：`hl_flat_pick_env_cfg.py:120/143`
- 动作 scale `0.125/0.25/5.0/20.0`：见 `high_level_todo.md` #4

### 16. `[已修]` 观测/动作布局变化会静默让旧 checkpoint 失效

- `mdp.last_action` 观测 = 动作总维度，所以任何 action term 维度变化都会改变 policy 观测布局
  （`CommandDrivenIKAction.action_dim` 从 7 改成 0 就让 flat/WBC checkpoint 全废）。
- 建议：训练启动时打印 `action_manager.total_action_dim` + 各 obs group 形状，
  或写入 config 版本号；加载 checkpoint 时校验 `actor.0.weight.shape[1]`。

> 修复于 `codex/hl-replay-layout`：新增 `highlevel/mdp/low_level_replay.py`，
> 高层 replay 启动时打印低层 policy 的 `action_dim` / 各 obs group 形状 / 布局分块，
> 并与 checkpoint 的 `actor.0.weight.shape` 严格比对，不一致直接 `RuntimeError`
> （而不是等 matmul 报错）。
>
> **实测确认这条隐患已经不是隐患，而是现实故障**：修复前
> `Isaac-Deeprobotics-High-Level-Pick-Flat-Teacher-v0` 第一次 `env.step()` 就抛
> `mat1 and mat2 shapes cannot be multiplied (Nx76 and 83x512)`
> （flat 低层 checkpoint 期望 83，replay 只给了 76），
> WBC / teleop 是 79 vs 86。修复后三者分别为 83/86/86，维度全部对齐。
> 详见 `high_level_todo.md` 的"清单外新增 A / B"。
>
> 仍待办（另开分支）：
> - 低层 env 自身还缺同样的启动期打印/断言（现在只覆盖了高层 replay 侧）；
> - `velocity/mdp/observations.py::joint_pos_rel_without_wheel` 里补文档建议的那条断言
>   （依赖"列序 == 原生序"，低层训练侧目前成立，但没有任何保护）。

### 17. `[已确认：不是 bug]` "机械臂/夹爪持续接触 90 N" 是归因错误 —— 实际只有轮子接触地面

**结论：不存在"默认位姿下臂/夹爪自碰撞"的问题**，无需为此改碰撞体/位姿。
上一轮的数字来自一个归因 bug：接触传感器的 `sensor.body_names` 顺序与
`robot.body_names`（PhysX articulation 顺序）**不同**，用后者去索引
`sensor.data.net_forces_w`，就会把四个轮子的地面支撑力错映射到 `arm_link5` /
`gripper_link2` / `hl_knee` 这些索引位置上。

按 `sensor.body_names` 正确归因后的实测（`Flat-Deeprobotics-M20-Piper-v0`，64 envs，step 20–80）：

| body | mean \|F\| | max \|F\| | 触发率 |
|---|---|---|---|
| fl/fr/hl/hr_wheel | 85–115 N | 208–249 N | ~99.7%（正常地面支撑 ≈ 体重/4） |
| gripper_link1 | 1.6 N | 178 N | 18.0% |
| gripper_link2 | 1.3 N | 130 N | 17.4% |
| gripper_base | 36 N | 2676 N | 4.4% |
| base_link | 29 N | 2676 N | 4.0% |
| fl_hipx / fr_hipx | 13.6 N | 1289 N | 4.7% / 3.1% |
| arm_link5 | 4.1 N | 598 N | 1.5% |

- 因此 `undesired_contacts`（weight −1.0，实测稳态 0.69）捕捉到的是**间歇性真实接触**，
  不是恒定惩罚：主要来自 ① 复位姿态随机化（roll/pitch ±0.3）下底盘/腿磕地；
  ② IK 驱动的机械臂摆动时偶尔碰到自身或地面。属于"惩罚项在正常工作"。
- 同时排除的其它猜测（都有实验）：执行器延迟/阻尼（`max_delay=5` → Στ² 2.49e4，
  `max_delay=0` → 2.83e4，`+damping=40` → 4.03e4，反而更差）；
  "零动作下臂被外力顶开"（`arm_joint2` 0.5 → 1.68 rad）——那是 IK 动作项在跟踪
  `ee_pose` 课程目标，不是碰撞推的。
- 修正后的推论：#4 的权重标定结论仍然成立（臂 τ² 在**无碰撞**情况下也在 2.4e4 量级，
  来自 IK/PD 跟踪与复位瞬态），但"臂奖励偏大是因为碰撞"这个说法作废。

### 18. `[待办]` 接触传感器与 articulation 的 body 顺序不同（归因陷阱）

- `sensor.body_names`（USD 遍历顺序：`base_link → arm_* → gripper_* → 腿/轮`）与
  `robot.body_names`（PhysX 顺序：hipx 组 → `arm_joint1` → hipy 组 → …）**顺序不同**。
- 仓库自身的奖励是安全的：`SceneEntityCfg` 针对 `contact_forces` 这个 sensor 解析
  `body_ids`（`undesired_contacts` / `contact_forces` / `feet_*` 都走 sensor），顺序自洽。
- 但人工调试脚本若写成 `sensor.data.net_forces_w[:, robot.body_names.index(name)]`
  会得到完全错误的结论（本次即踩此坑）。
- 建议：调试统一用 `sensor.body_names` 索引；或加断言提示二者不同。

---

## 二、工程性

1. `[待办]` `mdp/__init__.py` 的星号导入造成同名遮蔽：`velocity/mdp/terrains.py:285` 的
   `ROUGH_TERRAINS_CFG` 与 IsaacLab 官方同名对象不同；`randomize_rigid_body_inertia`、
   `randomize_com_positions` 也覆盖了官方实现；`import *` 还把 `terrain_gen`、
   `TerrainGeneratorCfg` 泄漏进 namespace。
2. `[待办]` 死代码/占位：`highlevel/mdp/encoder.py:69-71`（`my_project.models`、
   `/path/to/checkpoint.pth`）、`:57-60`（`torch.hub.load` 需联网）、
   `velocity_env_cfg.py:927-971`（未使用的 `create_obsgroup_class`）、
   `scripts/reinforcement_learning/rl_utils.py:35-43`（未使用的 import/空函数）。
3. `[待办]` `setup.py` 的 `packages=["rl_training"]` 只列了顶层包（非 editable 安装会缺子包）；
   `install_requires` 里拉了 `cusrl[all]`，但注册的 `cusrl_cfg_entry_point` 指向
   不存在的 `agents.cusrl_ppo_cfg`（`deeprobotics_m20/__init__.py` 里 6 处）。
4. `[待办]` 依赖被本地魔改：`D:\nvidia-isaac-sim\IsaacLab-5.1.0\...\task_space_actions.py`
   里有 `[IK DEBUG]` 调试打印和本地逻辑改动；`CommandDrivenIKAction` 覆盖
   `_compute_frame_jacobian` 并依赖 `_offset_pos/_body_idx/_ik_controller` 等私有成员，
   升级 IsaacLab 会静默改变行为。
5. `[待办]` VR/遥操作：`devices/vr_extented.py:59-63` 模块级 print、构造函数里直接起
   HTTPS/WebSocket 线程且无超时；`xtrainer_utils/XLeVR/xlevr/utils.py` 用 subprocess 调外部命令。
6. `[待办]` `scripts/utils/mp4-png-composition.py` 有 4 处裸 `except:`（170/189/241/249）。
7. `[待办]` 中文/英文注释混排、大段注释掉的代码
   （`velocity_env_cfg.py:213-225/285-309/667-675/785-805/853-866`、
   `commands.py:41-98`、`rewards.py:430-482/720-752`、`flat_env_wbc_cfg.py:477-485`）。
8. `[待办]` `logs/` 下每个 run 有 50 个 `model_*.pt`（~5 MB/个）；源码里引用带时间戳的
   路径（见 `high_level_todo.md` #7）→ 建议统一走命令行参数。

---

## 三、已在本次工作中修复（供追溯）

| 项 | 位置 | 提交 |
|---|---|---|
| 新增"策略直接控制机械臂(关节空间)"的隔离 env | `velocity/config/.../flat_env_arm_cfg.py` | `b3496a5` |
| 手臂奖励坐标系修复 + EE body 索引缓存 | `velocity/mdp/arm_rewards.py` | `b3496a5` |
| 新增线性课程 `ramp_reward_weight` | `velocity/mdp/curriculums.py` | `b3496a5` |
| 手臂 env 注册与 runner cfg | `deeprobotics_m20/__init__.py`、`agents/rsl_rl_ppo_cfg.py` | `b3496a5` |
| privileged 观测 reset 感知（含 `ObsTerm.params` 透传修正） | `velocity/mdp/observations.py` | `905c2df` |
