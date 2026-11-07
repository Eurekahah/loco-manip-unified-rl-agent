# M20 + Piper 低层托板取放任务制作流程

**任务分支**：`codex/low-shelf-pick-place`

**目标**：制作可执行的 low-shelf pick-and-place 场景，并让 M20 的机身高度与 pitch 成为完成任务的必要动作。

**当前状态（2026-11-05）**：第一版场景、状态教师任务、相机化数据采集任务、键盘/VR 输入链路、HDF5 记录和最小回放/检查工具已实现。训练任务 1/4 环境无头 smoke、两路相机 headless smoke、键盘 headless/GUI smoke、VR 初始化 smoke 均通过。本阶段只完成第一部分的数据采集闭环；完整抓取/放置成功率仍待后续验证。

第二部分的双圆柱台任务已单独记录在 [two_cylinder_place_zh.md](two_cylinder_place_zh.md)，训练、遥操和采集配置与本任务保持独立。

## 1. 任务与配置分离

| 用途 | Gym task | 控制接口 | 相机 | HDF5 recorder |
|---|---|---|---|---|
| RL 训练/状态教师 | `Isaac-M20-Piper-LowShelf-Place-v0` | `PreTrainedPickWBCAction` | 默认关闭 | 默认关闭 |
| 纯遥操调试 | `Isaac-M20-Piper-LowShelf-Teleop-v0` | `TeleopLLAction` | 默认关闭 | 默认关闭 |
| 相机化数据采集 | `Isaac-M20-Piper-LowShelf-Collect-v0` | `TeleopLLAction` | 默认场景相机 + 腕部相机 | 默认启用 |

训练配置 `LowShelfPlaceEnvCfg` 明确把 `workcell_camera`、`wrist_camera`、`arm_camera`、`nav_camera` 置空，`RecorderManager` 保持 0 个 term。采集配置 `LowShelfPlaceCollectEnvCfg` 才显式创建两路 RGB-D 相机和 HDF5 recorder。这样不会把相机更新或数据落盘带回无头训练。

任务 ID：

```text
Isaac-M20-Piper-LowShelf-Place-v0
Isaac-M20-Piper-LowShelf-Teleop-v0
Isaac-M20-Piper-LowShelf-Collect-v0
```

语言指令：

```text
把下层托盘里的糖盒放到上层桌面。
```

任务流程：

```text
对位
-> 降低机身
-> 向前俯仰
-> 机械臂伸入下层
-> 抓取糖盒
-> 抬升并收回机械臂
-> 移动到上层目标区
-> 放置并松开夹爪
-> 后退保持稳定
```

第一版只做单物体、单目标区、固定物体初始朝向的小范围随机化，不引入多物体、货架内部深处抓取或复杂厨房资产。

## 2. 场景设计

场景使用简洁的双层实验架：

- 下层取料板高度约 `0.30 m`。
- 上层放置板高度约 `0.55 m`。
- 上层板在下层取料区上方形成悬挑，使机器人需要降低机身并向前俯仰才能从前方伸入。
- 目标区使用独立、可观测的蓝色放置垫。
- 机器人初始位置位于取料架前方，不使用 DexCube。

默认物体：

```text
{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd
```

如果资产审计发现糖盒碰撞体或抓取稳定性不足，再切换到 `006_mustard_bottle.usd`。切换前必须重新确认夹爪宽度、质量和初始高度。

## 3. 控制接口

高层动作复用现有 WBC 版本：

```text
[vx, vy, wz,
 ΔEE_x, ΔEE_y, ΔEE_z,
 ΔEE_roll, ΔEE_pitch, ΔEE_yaw,
 Δbody_height, Δbody_pitch, Δbody_roll,
 gripper]
```

前 12 维由 `TeleopLLAction` 转成低层 WBC 命令，夹爪继续使用独立的二值开合动作。键盘使用增量语义，VR 使用“标定基准 + 手柄偏移”的绝对命令语义。

低层策略复用现有 `PreTrainedPickWBCAction` / `TeleopLLAction` 和 WBC checkpoint。新任务只定义场景、物体、目标、观测、奖励、成功条件和采集接口。

非 headless 的键盘/VR 遥操会在 Isaac Sim viewport 中只显示红色 EE 目标位姿箭头。底盘目标速度箭头、当前速度箭头、低层 IK debug marker 和命令 debug marker 均不启用。

## 4. 相机化数据采集

### 4.1 相机配置

两路相机都在 `scene_cfg.py` 中通过工厂函数配置：

- `make_workcell_camera(...)`：固定场景相机，默认位于工作单元前左侧，使用世界系 look-at 朝向工作台。
- `make_wrist_camera(...)`：挂在 `{ENV_REGEX_NS}/Robot/gripper_base/WristCamera`，沿 Piper 工具轴向前下视。
- 两路都输出 `rgb` 和 `depth`，默认 `224 x 224`、`10 Hz`。
- 相机外参通过 `TiledCamera.data` 读取。腕部相机设置 `update_latest_camera_pose=True`，随机械臂更新。
- 采集脚本自动设置 `--enable_cameras`，并允许用 `--scene_camera/--no-scene_camera`、`--wrist_camera/--no-wrist_camera` 选择相机。
- 分辨率、更新周期可配置：`--camera_width`、`--camera_height`、`--camera_update_period`。

实拍确认：

- 场景相机同时看到下层糖盒、上层板和蓝色目标垫。
- 腕部相机在初始高位能看到上层目标区，机械臂进入下层时覆盖夹爪前方作业区。
- 场景相机 `+X` 光轴指向 look-at 目标，`+Z` 图像上方接近世界 `+Z`。
- 腕部相机 `+X` 光轴沿 Piper 工具轴向前并向下约 `30°`。

### 4.2 记录生命周期

使用 Isaac Lab 原生 `RecorderManager` + 自定义 `LowShelfHDF5Recorder`，不重写环境架构：

- `record_pre_step`：在物理推进前记录当前状态、相机 RGB-D、内外参、时间戳和本步高层动作。
- `record_post_step`：在环境自动 reset 前写入各终止项布尔值、`terminated`、`truncated` 和 `place_success`。
- `record_pre_reset`：在 Isaac Lab 清空 terminated env 前写出完整 episode，避免状态被 reset 覆盖。
- `close`：用户直接退出或到达 `--max_steps` 时保存当前未完成 episode，标签为 `shutdown_manual_stop`。

自动成功由终止项 `place_success` 触发。人工标签：

- 键盘：`K` 成功，`L` 失败。
- VR：左手 `Y` 成功，左手 `X` 失败。

人工标签会在下一次 `env.reset()` 时写入 HDF5。

### 4.3 HDF5 schema

根组：

```text
/episodes/episode_000000
```

文件属性包含：

```text
schema = low_shelf_pick_place/v1
task, instruction, source, action_layout
camera_names, quaternion_convention
camera_convention, depth_convention, intrinsics_convention
sim_dt, control_dt, decimation, render_interval, num_envs
```

Episode 数据：

```text
timestamp [T]              wall-clock Unix time
sim_time [T]               simulation time
step_index [T]
language_instruction       scalar UTF-8
action [T, 13]             raw high-level action
processed_action [T, D]    concatenated processed action terms
ll_command [T, 13]         canonical low-level command

robot/root_pos_w [T, 3]
robot/root_quat_w [T, 4]   wxyz
robot/root_lin_vel_w [T, 3]
robot/root_ang_vel_w [T, 3]
robot/joint_pos [T, J]
robot/joint_vel [T, J]
robot/ee_pos_w [T, 3]
robot/ee_quat_w [T, 4]
robot/gripper_pos [T, 2]
robot/roll, pitch, yaw [T]

object/root_pos_w, root_quat_w, root_lin_vel_w
target_pad/root_pos_w, root_quat_w

termination/flags [T, N_terms]
termination/terminated [T]
termination/truncated [T]
termination/success [T]
termination attrs: names, label

camera/<name>/rgb [T, H, W, 3]                  uint8
camera/<name>/depth [T, H, W]                   metres
camera/<name>/intrinsics [T, 3, 3]              row-major pinhole K
camera/<name>/pos_w [T, 3]
camera/<name>/quat_w_world [T, 4]
camera/<name>/quat_w_ros [T, 4]
camera/<name>/T_world_camera_world [T, 4, 4]
camera/<name>/T_world_camera_ros [T, 4, 4]
camera/<name>/T_camera_world_world [T, 4, 4]
camera/<name>/T_camera_ros_world [T, 4, 4]
camera/<name>/sensor_frame [T]
```

深度使用 `distance_to_image_plane`，单位为米。当前 `depth_clipping_behavior="zero"`，超出裁剪范围或无效像素为 `0`，下游应把 `0` 视为无效深度。

## 5. 实施步骤

- [x] 建立任务分支和流程文档。
- [x] 新增双层架、YCB 糖盒、目标垫和固定相机场景。
- [x] 新增目标垫位姿观测、机身姿态观测和夹爪状态观测。
- [x] 新增抓取、搬运、放置奖励及组合成功判定。
- [x] 注册 RL teacher 任务、纯遥操任务和相机采集任务。
- [x] 将场景相机和腕部相机改为采集任务显式启用，训练/纯遥操默认关闭。
- [x] 新增 RGB-D、内外参、机器人状态、高层动作、语言、时间戳和终止标签的 HDF5 记录。
- [x] 新增键盘测试、VR 正式采集脚本和人工成功/失败标签。
- [x] 新增 HDF5 检查与 RGB-D 回放脚本。
- [x] 做资产静置冒烟测试；抓取/放置成功率测试待做。
- [x] 做 1/4 环境无头训练 smoke test。
- [x] 做启用相机的 headless smoke 和短 GUI/键盘 smoke。
- [x] 更新本文档的实测结果和遗留问题。

## 6. 验收记录

### 6.1 2026-11-04 第一版场景

- 场景创建时间：约 `1.2 s`（1/8 env，Windows + A4000）。
- 物体静置：稳定。糖盒中心最终高度约 `0.321 m`，下层架顶面 `0.300 m`。
- 物体初始位置：默认 `(0.68, -0.15, 0.34)`，随机偏移限制为 x `[-0.04, 0.02]`、y `[-0.03, 0.03]`、z `[0.0, 0.01]`。
- 目标垫：中心 `(0.78, 0.20, 0.555)`。
- 夹爪是否能稳定夹持：尚未做成功率测试。
- 2 iteration 训练：`1 env` 和 `8 env` 均 `EXIT=0`。
- 终止统计：8 env 冒烟中 `object_dropped=0`；随机策略下仍有 `bad_orientation`，属于未训练策略的正常现象。

### 6.2 2026-11-05 相机与 HDF5

环境：Windows 10 Pro、NVIDIA RTX A4000、驱动 `580.88`、Isaac Sim `5.1.0`、Isaac Lab `5.1.0`。

配置解析：

- 训练任务：`LowShelfPlaceEnvCfg`，`cameras=[]`，`recorders=[]`。
- 采集任务：`LowShelfPlaceCollectEnvCfg`，`cameras=['workcell_camera', 'wrist_camera']`，`recorders=['record_hdf5']`。

无头训练：

- 1 env、1 iteration：`28.46 s`，`24` env steps。
- 4 env、1 iteration：`31.85 s`，`96` env steps，约 `3 env step/s`。
- 训练仍为 `PreTrainedPickWBCAction`，无相机、无 recorder，未改变训练语义。

相机 smoke：

- `--headless --enable_cameras` 创建两路 `224 x 224` RGB-D，成功输出 PNG 和 HDF5。
- 场景相机深度中位数约 `1.04 m`，符合前侧到工作台的距离。
- 腕部相机外参随 `gripper_base` 更新；当前高位主要看到上层目标区，进入下层后覆盖夹爪前方。
- Vulkan 初始化正常。之前记录的“启用相机触发 Vulkan/相机崩溃”不是驱动根因；实际是两个 Python 错误：
  - `RecorderManager` 初始化早于 `TerminationManager`，记录器不能在该阶段读取终止项。
  - `TiledCamera.frame` 是 CUDA tensor，不能直接 `np.int64(...)`。
- 两个错误已修复。异常发生后 Omniverse 在退出阶段产生的 access violation 是二次崩溃，不是相机渲染失败。

键盘采集 smoke：

- headless 5 step：`recordings/collect_smoke/keyboard.hdf5`，约 `848 KB`。
- `source=keyboard`，语言指令正确，`action=(5, 13)`，两路相机均 `rgb=(5,224,224,3)`、`depth=(5,224,224)`、`K=(5,3,3)`、外参 `(5,4,4)`。
- GUI 3 step：`recordings/collect_smoke/keyboard_gui.hdf5`，键盘设备识别为 `Isaac Sim 5.1.0`。
- GUI 首次启动本次耗时约 `145 s`，主要花在 RTX 窗口/着色器初始化；这是启动开销，不是逐步控制卡死。
- 人工成功路径：`manual_success` 已实测写入，episode 属性 `success=True`。

VR 初始化 smoke：

- `--input vr --headless --max_steps 1` 成功初始化 XLeVR HTTPS `8443` 和 WebSocket `8442`。
- 生成 `source=vr` 的 1 step HDF5。
- 未连接真实头显，因此只验证了服务、动作接口和记录链路，不把无头初始化当成完整 VR 采集验收。

回放与检查：

- `replay_low_shelf_hdf5.py --check-only` 能检查 schema、动作、机器人状态、终止项和相机数据。
- RGB + 深度伪彩色 MP4 回放已从真实采集 HDF5 生成。

### 6.3 GUI 帧率诊断

- 高层训练环境每个 step 推进 `0.2 s` 仿真时间（`sim.dt=0.005`，`decimation=40`），并执行 10 次低层 WBC 策略。
- 本任务把 viewer 刷新周期设为低层控制步长 `4`，GUI 刷新率约为环境 step 率的 10 倍；物理语义、控制频率和策略行为不变。
- 采集/纯遥操任务使用 `decimation=4`，控制步长约 `0.02 s`，即约 `50 Hz`；相机默认 `10 Hz`，因此 HDF5 中相邻帧可能重复同一张相机图。
- 正式训练仍建议 `--headless`；GUI 建议 `--num_envs 1`。

## 7. 已知问题

- 尚未用真实 VR 头显完成一次完整 pick-and-place episode，也没有统计抓取/放置成功率。
- GUI smoke 没有自动注入键盘按键，只验证了设备初始化、主循环和 HDF5 落盘；实际键位移动仍需人工短测。
- 采集控制频率约 `50 Hz`，相机约 `10 Hz`。若后续 ACT/VLA 训练需要固定 10/20 Hz，建议在离线转换阶段按 `sim_time` 下采样，避免改变遥操响应频率。
- 深度无效值为 `0`。腕部相机看向远处时深度中位数可能为 0，属于裁剪设置，不应误判为相机损坏。
- 腕部相机安装位姿依赖当前 Piper `gripper_base` 坐标系。若更换夹爪、相机支架或 USD，需要重新做相机朝向 smoke。
- 当前 HDF5 每个 episode 在 reset/close 时整段写出，适合单环境遥操采集；长时间多环境流式写入尚未实现。
- 任务成功标签已接入，但自动 `place_success` 的完整 episode 仍待真实遥操数据验证。

## 8. 复现命令

先激活 conda 环境并在仓库根目录执行：

```powershell
conda activate env_isaac_lab
```

### 8.1 配置解析

训练任务应显示 `cameras=[]`、`recorders=[]`：

```powershell
python scripts/reinforcement_learning/rsl_rl/probe_task_cfg.py --task Isaac-M20-Piper-LowShelf-Place-v0 --headless
```

采集任务应显示两路相机和 `record_hdf5`：

```powershell
python scripts/reinforcement_learning/rsl_rl/probe_task_cfg.py --task Isaac-M20-Piper-LowShelf-Collect-v0 --headless
```

### 8.2 无头训练 smoke

1 环境：

```powershell
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-M20-Piper-LowShelf-Place-v0 --headless --num_envs 1 --max_iterations 1
```

4 环境：

```powershell
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-M20-Piper-LowShelf-Place-v0 --headless --num_envs 4 --max_iterations 1
```

### 8.3 相机 smoke

采集脚本会自动启用 Isaac Sim 相机：

```powershell
python scripts/data/smoke_low_shelf_cameras.py --task Isaac-M20-Piper-LowShelf-Collect-v0 --headless --num_envs 1 --steps 3 --output_dir recordings/camera_smoke
```

输出：

```text
recordings/camera_smoke/workcell_camera_rgb.png
recordings/camera_smoke/workcell_camera_depth.png
recordings/camera_smoke/wrist_camera_rgb.png
recordings/camera_smoke/wrist_camera_depth.png
recordings/camera_smoke/camera_smoke.hdf5
```

### 8.4 键盘功能测试

短 headless 采集：

```powershell
python scripts/data/collect_low_shelf_hdf5.py --task Isaac-M20-Piper-LowShelf-Collect-v0 --input keyboard --headless --num_envs 1 --max_steps 5 --output recordings/collect_smoke/keyboard.hdf5 --overwrite
```

GUI 键盘遥操：

```powershell
python scripts/data/collect_low_shelf_hdf5.py --task Isaac-M20-Piper-LowShelf-Collect-v0 --input keyboard --num_envs 1 --output recordings/low_shelf_place/keyboard.hdf5
```

键盘标签：

```text
K  保存当前 episode 为成功并 reset
L  保存当前 episode 为失败并 reset
G  切换夹爪开合
```

### 8.5 VR 正式采集

```powershell
python scripts/data/collect_low_shelf_hdf5.py --task Isaac-M20-Piper-LowShelf-Collect-v0 --input vr --num_envs 1 --output recordings/low_shelf_place/vr.hdf5
```

VR 标签：

```text
右 B  开始控制并重新标定
左 Y  保存当前 episode 为成功并 reset
左 X  保存当前 episode 为失败并 reset
```

### 8.6 HDF5 检查与回放

只检查 schema 和数值：

```powershell
python scripts/data/replay_low_shelf_hdf5.py --input recordings/collect_smoke/keyboard.hdf5 --check-only
```

生成场景相机 RGB-D 回放：

```powershell
python scripts/data/replay_low_shelf_hdf5.py --input recordings/collect_smoke/keyboard.hdf5 --camera workcell_camera --output recordings/collect_smoke/keyboard_workcell_replay.mp4
```

生成腕部相机 RGB-D 回放：

```powershell
python scripts/data/replay_low_shelf_hdf5.py --input recordings/collect_smoke/keyboard.hdf5 --camera wrist_camera --output recordings/collect_smoke/keyboard_wrist_replay.mp4
```

## 9. 维护约定

每次完成一个实施步骤，都要同步更新本文件的：

- 当前状态。
- 任务与配置分离表。
- 实施步骤复选框。
- 验收记录。
- 复现命令和已知问题。
