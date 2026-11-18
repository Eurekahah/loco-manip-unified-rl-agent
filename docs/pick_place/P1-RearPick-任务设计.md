# P1 RearPick 任务设计与实施规格

版本：v0.1  
日期：2026-11-18  
状态：Teacher / Teleop / Collect 已实现；视觉学生与正式评测未开始

关联文档：

- [WBC-OmniBench 任务设计](../WBC-OmniBench-任务设计.md)
- [Isaac 任务实施交接单](../Isaac任务实施交接单.md)
- [low-shelf-pick-place 制作记录](low_shelf_place_zh.md)
- [two-cylinder-pick-place 制作记录](two_cylinder_place_zh.md)

---

## 0. 当前实现状态

| 能力 | 状态 | 说明 |
|---|---|---|
| 后侧工作桌、糖盒、托盘 | 已完成 | 糖盒和托盘按左右侧随机布置 |
| `RearPick-Teacher-v0` | 已完成 | 状态教师、奖励、终止、1 环境训练 smoke 通过 |
| `RearPick-Teleop-v0` | 已完成 | 键盘/VR 高层动作入口可用 |
| `RearPick-Collect-v0` | 已完成 | top/workcell/front/wrist + four fisheye HDF5 采集链路通过 |
| 初始可见性截图 | 已完成 | 初始前视/腕部看不到后方目标，全景可见 |
| 重捕获截图 | 已完成 | 机器人面向后方桌面后，腕部相机可近距离看到目标 |
| 视觉学生 | 未开始 | 尚未接入全景/前视/腕部图像策略 |
| 统一 evaluator / 消融 | 未开始 | 当前仍是任务级 termination |

实现目录：

```text
source/rl_training/rl_training/tasks/manager_based/locomotion/highlevel/config/rear_pick/
scripts/data/capture_rear_pick_views.py
```

---

## 1. 任务定位

P1 RearPick 是 WBC-OmniBench 的第一个核心任务，用于验证：

1. 目标初始只出现在尾部高位全景相机中；
2. 机器人能够利用全景信息选择转向和接近方向；
3. 目标进入前视或腕部视场后，机器人能够完成局部重捕获；
4. 机器人能够完成抓取、搬运和托盘放置；
5. 去掉全景后，任务不能依赖固定初始视角或预置目标位置完成。

P1 不检验：

- 门、抽屉等关节物体操作；
- 动态目标拦截；
- 长距离跨房间导航；
- 单腿独立操作；
- 语言或属性目标选择。

这些能力分别留给 P3、W1、P2 和 L1。

---

## 2. 设计原则

### 2.1 先做局部后侧任务

初版不把目标放在远处房间，而放在机器人后方约 `1.2–1.8 m` 的工作桌周围。这样可以把“全景发现”和“跨视角重捕获”从长距离导航中分离出来。

### 2.2 状态教师和视觉学生分离

实现顺序固定为：

```text
状态教师
  -> 场景、奖励、成功判定和动作链路验证
  -> 视觉学生
  -> 全景输入和跨视角重捕获
  -> 遥操采集与消融
```

视觉学生不得读取物体或托盘的仿真真值位姿。真值只允许用于：

- 状态教师训练；
- evaluator；
- 可见性审计；
- 课程和失败分析。

### 2.3 先复用 13 维 WBC 动作

初版沿用现有高层动作：

```text
[vx, vy, wz,
 delta_ee_x, delta_ee_y, delta_ee_z,
 delta_ee_roll, delta_ee_pitch, delta_ee_yaw,
 delta_body_height, delta_body_pitch, delta_body_roll,
 gripper]
```

底盘速度用于转向和接近，EE 增量用于抓取和放置。只有在 P1 的局部导航被证明超出 WBC 能力时，才增加 `PreTrainedNavAction` 与 `PreTrainedPickWBCAction` 的分阶段切换。

---

## 3. 场景规格

### 3.1 世界坐标和初始位姿

初版使用固定世界坐标，机器人初始朝向 `+X`：

```text
robot_root: (0.0, 0.0, 0.55)
robot_yaw:  0 rad
```

目标工作桌位于机器人后方：

```text
table_center: (-1.35, 0.0, 0.525)
table_size:   (0.85, 1.00, 0.05)
table_top_z:  0.55
```

目标物体放在桌面左右两侧，托盘放在另一侧：

```text
target_side = random_choice(-1, +1)
target_xy = (-1.35, target_side * 0.24)
tray_xy   = (-1.35, -target_side * 0.22)
```

初始目标相对机器人前向的方位角限制在：

```text
azimuth ∈ [120°, 240°]
```

目标到机器人 root 的水平距离限制在：

```text
distance_xy ∈ [1.2 m, 1.8 m]
```

### 3.2 物体

初版只使用：

```text
{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd
```

选择糖盒的原因：

- 已在两个基础任务中验证静置稳定性；
- 尺寸和夹爪开口已知；
- 可以先隔离搜索和重捕获问题，不把成功率绑定到新的抓取几何。

后续物体扩展顺序：

1. `006_mustard_bottle.usd`；
2. `025_mug.usd`；
3. `005_tomato_soup_can.usd`；
4. `003_cracker_box.usd`。

### 3.3 托盘

托盘使用一个可碰撞的矩形开口区域，初版可简化为：

```text
tray_outer_size: (0.32, 0.24, 0.04)
tray_floor_z:    0.57
success_xy_radius: 0.08
```

为了降低建模复杂度，初版可以用 kinematic 托盘底板加四个低矮边框。成功区域以托盘中心为圆心，在 evaluator 中单独定义。

### 3.4 可选遮挡

初版不放强制隔断，因为目标位于机器人后方，前视和腕部自然看不到目标。

在 OOD 场景中可增加：

- 左侧或右侧矮隔断；
- 桌后挡板；
- 两个相似错误目标；
- 可移动障碍物。

这些内容不进入 P1 v0 的最小成功条件。

### 3.5 相机

初版视觉配置：

| 相机 | 配置 | 用途 |
|---|---|---|
| 全景 | four fisheye，4 × 320×200，约 10 Hz | 初始发现和全局关系 |
| 前视 | RGB，224×224，约 10 Hz | 转向后的局部重捕获 |
| 腕部 | RGB，224×224，约 10 Hz | 抓取阶段精细对准 |
| 顶部调试相机 | 可选，只用于日志 | 不进入策略 |

全景优先使用四路原始鱼眼，不默认对每个并行环境输出高分辨率等距柱面图。

可选的 `pano_equirect` 初版使用 `640×320`，只在以下情况启用：

- 评测可视化；
- VLA 或 2D backbone 实验；
- 与真实拼接相机接口对齐。

---

## 4. 初始可见性门槛

每个 episode reset 后必须执行一次可见性审计。

### 4.1 必须满足

- 目标中心或至少一个有效投影点在全景视场内；
- 目标中心不在前视相机视场内，或被前视可见物体完全遮挡；
- 目标中心不在腕部相机视场内，或被遮挡；
- 前视和腕部中目标可见像素比例低于阈值。

### 4.2 建议阈值

```text
pano_center_visible = true
front_center_visible = false
wrist_center_visible = false
front_visible_ratio < 0.01
wrist_visible_ratio < 0.01
```

### 4.3 实现方式

评测侧可以使用：

1. 目标中心投影到相机图像；
2. 从相机到目标中心发射射线；
3. 检查射线是否被其他碰撞体阻挡；
4. 使用 RGB 目标分割或深度一致性作为辅助验证。

可见性计算只用于 evaluator、课程和日志，不能直接输入视觉学生。

### 4.4 reset 失败处理

如果 reset 后不满足可见性门槛：

- 重新采样目标位置，最多尝试 20 次；
- 仍失败则该 episode 标记为 `invalid_reset`；
- 训练时可跳过该环境；
- 评测时必须计入有效 episode 比例，不能静默丢弃。

---

## 5. 观测设计

### 5.1 状态教师

状态教师可以使用特权信息：

```text
target_pose_in_robot:  7
tray_pose_in_robot:    7
target_pose_in_ee:     7
body_pose_state:       3   # height, pitch, roll
gripper_joint_pos:     2
base_lin_vel:          3
base_ang_vel:          3
previous_action:      13
```

复用现有：

- `object_pose_in_robot_root_frame`
- `object_pose_in_ee_frame`
- `body_pose_state`
- `gripper_joint_state`

### 5.2 视觉学生

视觉学生输入建议：

```text
pano_fisheye:  [4, H, W, 3]
front_rgb:     [H, W, 3]
wrist_rgb:     [H, W, 3]
body_pose:     [3]
gripper_pos:   [2]
base_twist:    [6]
last_action:   [13]
camera_times:  [N_camera]
```

视觉学生不得输入：

- `target_pose_in_robot`；
- `tray_pose_in_robot`；
- 目标或托盘的仿真 root state；
- 接触力真值；
- 目标可见性布尔值；
- 任何由 evaluator 计算的真值标签。

### 5.3 视觉编码

初版建议：

- 四路鱼眼共享一个轻量 CNN 编码器；
- 前视和腕部分别使用轻量 CNN；
- 保留相机身份 embedding；
- 融合后接历史帧或 GRU；
- 训练早期允许使用状态教师蒸馏。

不建议第一版直接把四路鱼眼、前视、腕部和全部本体状态拼成一个超大 MLP 输入。

---

## 6. 动作与阶段

### 6.1 动作空间

初版使用 13 维 WBC 动作，见第 2.3 节。

### 6.2 内部任务阶段

阶段只用于奖励、指标和调试，不作为视觉学生必须输出的离散标签。

| 阶段 | 进入条件 | 主要动作 |
|---|---|---|
| SEARCH | episode 开始 | 全景确认目标、转向 |
| APPROACH | 选定目标方向 | 底盘转向和接近 |
| REACQUIRE | 目标进入前视/腕部 | 停止大范围搜索，局部对准 |
| GRASP | 目标进入腕部抓取区 | EE 对准、闭爪 |
| TRANSPORT | 物体抬升 | 收回机械臂、移动到托盘 |
| PLACE | 物体到达托盘上方 | 下降、开爪、后退 |
| SETTLE | 已释放物体 | 保持稳定并等待成功判定 |

### 6.3 阶段回退

允许以下回退：

- REACQUIRE 阶段目标再次丢失，回到 APPROACH；
- GRASP 阶段物体未被夹起，回到 REACQUIRE；
- TRANSPORT 阶段物体掉落，进入失败或允许一次重抓取；
- PLACE 阶段物体未进入托盘，允许一次重新放置。

初版只允许一次重抓取，防止长时程恢复掩盖基础技能问题。

---

## 7. 奖励设计

奖励全部使用势能或进度塑形，避免固定位置刷奖励。

### 7.1 建议奖励项

| 奖励 | 作用 | 建议权重 |
|---|---|---:|
| `pano_search_progress` | 初始朝向目标方向 | 0.3 |
| `approach_progress` | 机器人接近目标工作区 | 0.8 |
| `heading_progress` | 车头对准目标方向 | 0.3 |
| `reacquire_bonus` | 目标进入前视/腕部并稳定 | 1.0 |
| `ee_alignment_progress` | 两指对称接近目标 | 1.0 |
| `lift_progress` | 物体抬升进度 | 1.5 |
| `object_to_tray_progress` | 物体接近托盘 | 1.5 |
| `place_success` | 最终成功 | 4.0 |
| `action_rate` | 平滑动作 | -0.001 |
| `undesired_contacts` | 非预期碰撞 | -0.05 |
| `workspace_violation` | 越界或远离工作区 | -0.5 |

### 7.2 奖励门控

- `approach_progress` 只在物体尚未抬起时生效；
- `ee_alignment_progress` 只在目标进入前视或腕部后生效；
- `object_to_tray_progress` 只在物体曾抬升超过 `0.04 m` 后生效；
- `place_success` 只在稳定放置条件满足后触发；
- 不允许在目标未接近时持续获得 EE 对准奖励。

### 7.3 奖励禁止项

- 不按固定目标坐标给奖励；
- 不按初始方位角直接给成功奖励；
- 不按“全景看到目标”本身持续给大奖励；
- 不在物体未抬起时给托盘放置奖励；
- 不把物体被传送带、外力或脚本直接送入托盘计为成功。

---

## 8. 成功与失败判定

### 8.1 成功条件

以下条件同时满足并持续 `1.0 s`：

- 物体曾被抬升至少 `0.04 m`；
- 物体中心在托盘成功区域 XY 半径内；
- 物体高度与托盘底面匹配；
- 物体线速度低于 `0.05 m/s`；
- 物体角速度低于 `0.2 rad/s`；
- 物体姿态未翻转；
- 夹爪已张开；
- 机器人未摔倒；
- 无严重碰撞或越界；
- episode 时间小于 `90 s`。

### 8.2 失败条件

- 物体高度低于 `0.16 m`，视为掉落；
- 物体离开工作区边界；
- 托盘翻倒；
- 机器人 root 高度低于 `0.20 m`；
- 机器人 roll/pitch 超过安全阈值；
- 严重碰撞超过阈值；
- 超出 `90 s`；
- 发生非物理资产破坏。

### 8.3 失败类型

```text
invalid_reset
target_not_found
wrong_turn
navigation_failed
reacquire_failed
grasp_failed
object_dropped
transport_failed
place_failed
tray_tipped
robot_fallen
collision_limit
out_of_bounds
timeout
asset_broken
```

---

## 9. 指标

### 9.1 任务指标

- `success`；
- `progress`；
- `completion_time`；
- `path_length`；
- `reacquire_time`；
- `regrasp_count`；
- `failure_type`。

### 9.2 感知指标

- `pano_first_detect_time`；
- `pano_visible_ratio`；
- `front_visible_ratio`；
- `wrist_visible_ratio`；
- `pano_to_front_reacquire_time`；
- `front_to_wrist_reacquire_time`；
- `cross_view_association_error`；
- `camera_switch_count`。

### 9.3 机体与接触指标

- `base_roll_max`；
- `base_pitch_max`；
- `wheel_slip_distance`；
- `contact_force_peak`；
- `collision_count`；
- `max_joint_torque_ratio`；
- `energy`。

### 9.4 可见性计算约束

`*_visible_ratio` 由 evaluator 计算，只写入 episode metrics。视觉策略不能读取该字段。

---

## 10. 消融

### 10.1 全景因果性

- 无全景，只保留前视 + 腕部；
- 全景分辨率减半；
- 全景延迟 200 ms；
- 全景延迟 500 ms；
- 全景随机遮挡 20%；
- 全景随机遮挡 50%；
- 全景外参扰动 ±2° / ±2 cm；
- 目标改为初始前视可见。

### 10.2 任务因果性

- 固定目标侧别；
- 随机左右侧别；
- 固定场景布局；
- 随机场景布局；
- 固定物体；
- 随机物体实例。

### 10.3 机体消融

P1 不是机体协作主任务，但仍应记录：

- 固定机身高度；
- 固定 pitch/roll；
- 禁用轮速差；
- 禁用轮制动。

---

## 11. 课程

### Stage 0：固定后侧

- 固定目标在后方左侧；
- 固定物体；
- 固定相机；
- 状态教师；
- 目标始终满足初始可见性。

### Stage 1：随机左右

- 目标在左右两侧随机；
- 托盘在对侧随机；
- 机器人初始 yaw 小范围随机。

### Stage 2：随机物体

- 加入糖盒、芥末瓶、马克杯和罐头；
- 重新标定抓取高度和夹爪宽度；
- 失败样本按物体类型统计。

### Stage 3：传感器扰动

- 全景遮挡、降采样和延迟；
- 前视/腕部时间偏移；
- 相机外参扰动。

### Stage 4：场景泛化

- 不同桌子尺寸；
- 增加隔断和错误目标；
- 改变托盘位置；
- OOD 场景。

---

## 12. 任务实现映射

建议新增目录：

```text
source/rl_training/rl_training/tasks/manager_based/locomotion/highlevel/config/rear_pick/
  __init__.py
  scene_cfg.py
  mdp.py
  env_cfg.py
  agents.py
```

### 12.1 `scene_cfg.py`

负责：

- 机器人后方工作桌；
- 目标物体；
- 托盘；
- 可选隔断和错误目标；
- four fisheye、前视、腕部和调试顶部相机工厂；
- 场景边界。

### 12.2 `mdp.py`

负责：

- 目标初始可见性审计；
- 目标重捕获检测；
- 物体抬升和托盘稳定判定；
- 阶段进度；
- 失败类型；
- 任务指标缓存。

### 12.3 `env_cfg.py`

建议配置：

```text
RearPickTeacherEnvCfg
RearPickStudentEnvCfg
RearPickTeleopEnvCfg
RearPickCollectEnvCfg
```

### 12.4 `__init__.py`

建议任务 ID：

```text
Isaac-M20-Piper-RearPick-Teacher-v0
Isaac-M20-Piper-RearPick-Student-v0
Isaac-M20-Piper-RearPick-Teleop-v0
Isaac-M20-Piper-RearPick-Collect-v0
```

### 12.5 `agents.py`

初版复用现有 high-level PPO runner：

- teacher 使用状态输入；
- student 使用视觉输入；
- 可选 teacher-to-student distillation；
- 训练名固定为 `rear_pick`。

---

## 13. 实施顺序

### 步骤 1：场景和可见性探针

- 建立后方工作桌、糖盒和托盘；
- 记录机器人、目标、托盘和相机的初始位姿；
- 实现 reset 后可见性审计；
- 输出全景、前视和腕部截图；
- 验证初始帧前视/腕部看不到目标。

### 步骤 2：状态教师

- 注册 `RearPick-Teacher-v0`；
- 复用 13 维 WBC 动作；
- 实现搜索、接近、重捕获、抓取、放置奖励；
- 完成 1/4 环境无头 smoke；
- 记录成功、失败和阶段进度。

### 步骤 3：遥操和采集

- 注册 `RearPick-Teleop-v0`；
- 注册 `RearPick-Collect-v0`；
- 配置 four fisheye、前视和腕部相机；
- 记录 HDF5；
- 采集至少一条脚本化或人工成功 episode。

### 步骤 4：视觉学生

- 接入全景、前视和腕部图像编码；
- 实现 teacher-to-student distillation；
- 禁止学生读取物体/托盘真值；
- 记录跨视角重捕获时间。

### 步骤 5：评测和消融

- 实现统一 evaluator；
- 完成无全景、降采样、延迟和遮挡消融；
- 完成固定初始前视可见的对照实验；
- 统计 100 个 ID episodes 和 OOD episodes。

---

## 14. 验收门槛

P1 进入“可训练”状态的最低门槛：

- 场景可稳定创建；
- 1/4 环境无头 smoke 通过；
- reset 可见性审计通过率不低于 99%；
- 状态教师能完成动作链路和终止判定；
- HDF5 能记录目标、托盘、机器人、动作、三路相机和终止项。

P1 进入“可发表/可对比”状态的最低门槛：

- 视觉学生不读取任何物体或托盘真值；
- 无全景基线的成功率显著低于有全景基线；
- 目标重捕获时间可统计；
- 至少完成一次全景延迟和遮挡消融；
- 成功和失败类型可复现；
- 初始前视可见的对照实验能显著缩小方法间差距。

建议的初始因果性判据：

```text
有全景成功率 - 无全景成功率 >= 30 个百分点
无全景的 target_not_found / reacquire_failed 比例显著上升
```

如果无全景基线没有明显下降，应优先检查：

- 目标是否实际处于前视/腕部初始可见位置；
- 是否存在固定路线或固定目标位置作弊；
- 是否把全景真值或物体真值泄漏给学生；
- 场景是否允许机器人仅靠固定初始转向完成。

---

## 15. 主要风险

### 15.1 WBC 局部导航能力

如果 13 维 WBC 无法稳定完成后侧 `1.2–1.8 m` 的转向和接近，应增加导航阶段，而不是放宽成功判定。

### 15.2 鱼眼渲染成本

四路鱼眼会增加 render product 和内存开销。训练默认使用低分辨率原始鱼眼，高分辨率等距柱面只用于评测和可视化。

### 15.3 目标检测误差

如果视觉检测不稳定，应先把目标检测拆成独立感知模块，再训练端到端策略。不能用仿真真值掩盖感知失败。

### 15.4 抓取成功率

糖盒抓取仍然可能受夹爪接触和姿态影响。P1 初版只使用已验证的糖盒，不把透明、薄壁或极小物体放入首轮验收。

### 15.5 阶段作弊

禁止：

- reset 时直接把目标转到前视可见位置；
- 用物体真值控制视觉学生；
- 用固定脚本路径代替策略；
- 用外力把物体送入托盘；
- 把目标直接设置为成功状态；
- 无全景基线和有全景基线使用不同的初始目标分布。

---

## 16. 下一步

完成本设计后，下一步直接实现：

1. `rear_pick/scene_cfg.py`；
2. 可见性探针和截图脚本；
3. `rear_pick/mdp.py`；
4. `rear_pick/env_cfg.py`；
5. `rear_pick/__init__.py` 和 `agents.py`；
6. `RearPick-Teacher-v0` 的 1/4 环境 smoke。

在状态教师和可见性门槛通过前，不开始视觉学生训练。

---

## 17. 已实现内容与复现

### 17.1 任务 ID

```text
Isaac-M20-Piper-RearPick-Teacher-v0
Isaac-M20-Piper-RearPick-Teleop-v0
Isaac-M20-Piper-RearPick-Collect-v0
```

### 17.2 配置解析

```powershell
conda run --no-capture-output -n env_isaac_lab python -u `
  scripts/reinforcement_learning/rsl_rl/probe_task_cfg.py `
  --task Isaac-M20-Piper-RearPick-Teacher-v0 --headless
```

### 17.3 状态教师 smoke

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run --no-capture-output -n env_isaac_lab python -u `
  scripts/reinforcement_learning/rsl_rl/train.py `
  --task Isaac-M20-Piper-RearPick-Teacher-v0 `
  --headless --num_envs 1 --max_iterations 1
```

已通过：24 env steps，日志和 `model_0.pt` 正常写出。

### 17.4 截图

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run --no-capture-output -n env_isaac_lab python -u `
  scripts/data/capture_rear_pick_views.py `
  --headless `
  --output_dir recordings/scene_views/rear_pick
```

输出：

```text
recordings/rear_pick_views/rear_pick_initial_top_camera.png
recordings/rear_pick_views/rear_pick_initial_workcell_camera.png
recordings/rear_pick_views/rear_pick_initial_front_camera.png
recordings/rear_pick_views/rear_pick_initial_wrist_camera.png
recordings/rear_pick_views/rear_pick_initial_panorama.png
recordings/rear_pick_views/rear_pick_initial_fisheye_montage.png
recordings/rear_pick_views/rear_pick_initial_montage.png
recordings/rear_pick_views/rear_pick_reacquired_montage.png
recordings/rear_pick_views/rear_pick_reacquired_wrist_camera.png
```

最终 `quality` 渲染图片位于 `recordings/scene_views/rear_pick/`：

![P1 RearPick 俯视](../../recordings/scene_views/rear_pick/rear_pick_initial_top_camera.png)

![P1 RearPick 四视角](../../recordings/scene_views/rear_pick/rear_pick_initial_montage.png)

![P1 RearPick 全景](../../recordings/scene_views/rear_pick/rear_pick_initial_panorama.png)

![P1 RearPick 重捕获](../../recordings/scene_views/rear_pick/rear_pick_reacquired_montage.png)

### 17.5 遥操采集 smoke

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run --no-capture-output -n env_isaac_lab python -u `
  scripts/data/collect_demonstrations.py `
  --task Isaac-M20-Piper-RearPick-Collect-v0 `
  --profile rear_pick `
  --input keyboard --headless --num_envs 1 --max_steps 8 `
  --front_camera `
  --fisheye-rig four --fisheye-width 320 --fisheye-height 200 `
  --record-hz 10 `
  --output recordings/rear_pick_collect_smoke/rear_pick_keyboard_smoke.hdf5 `
  --overwrite
```

已通过。HDF5 包含：

```text
top_camera
workcell_camera
front_camera
wrist_camera
fisheye_cam0..3
```

检查命令：

```powershell
conda run --no-capture-output -n env_isaac_lab python -u `
  scripts/data/replay_hdf5.py `
  --input recordings/rear_pick_collect_smoke/rear_pick_keyboard_smoke.hdf5 `
  --check-only
```

### 17.6 已知限制

- 视觉学生和 teacher-to-student distillation 尚未接入。
- 初始可见性目前通过截图和场景几何验证，尚未形成自动 reset 重采样。
- P1 尚未完成真实抓取、放置和成功率统计。
- `front_camera` 默认关闭；采集时需要显式传 `--front_camera`。
- `RearPick-Collect-v0` 使用 `fast_robot_only_write=True`，避免向 kinematic 托盘写入速度。
