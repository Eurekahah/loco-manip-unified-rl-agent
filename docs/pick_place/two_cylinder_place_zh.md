# M20 + Piper 双圆柱台取放任务制作记录

**分支**：`codex/low-shelf-pick-place`

**基准提交**：`b0b9aaf`

**目标**：在低层货架任务之外，增加一个更简单、可语言描述、可被 M20 + Piper 执行的单物体 pick-and-place 场景。矮圆柱台放 YCB 糖盒，高圆柱台放目标垫；两平台横向错开，搬运阶段需要底盘侧移，取料阶段需要降低机身并轻微前俯，放置阶段需要抬升机身。

**下一步数据采集基准**：以本任务的“矮圆柱台取糖盒 -> 高圆柱台目标垫”为唯一示例任务，先采 VR 成功 episode，再走 LeRobot v3 转换和 SmolVLA 微调。低层货架数据只作为后续扩展，不与本基准混在同一数据集里。

## 1. 任务与配置分离

| 用途 | Gym task | 控制接口 | 相机 | HDF5 |
|---|---|---|---|---|
| RL 训练/状态教师 | `Isaac-M20-Piper-TwoCylinder-Place-v0` | `PreTrainedPickWBCAction` | 默认关闭 | 默认关闭 |
| 纯遥操调试 | `Isaac-M20-Piper-TwoCylinder-Teleop-v0` | `TeleopLLAction` | 默认关闭 | 默认关闭 |
| 相机化数据采集 | `Isaac-M20-Piper-TwoCylinder-Collect-v0` | `TeleopLLAction` | 顶部 + 腕部 + 场景 RGB | 默认启用 |

配置继承关系：

```text
HighLevelFlatEnvCfg
  -> LowShelfPlaceEnvCfg
     -> TwoCylinderPlaceEnvCfg
        -> TwoCylinderPlaceTeleopEnvCfg
           -> TwoCylinderPlaceCollectEnvCfg
```

新任务复用：

- `HighLevelFlatEnvCfg` 的平地训练语义。
- `PreTrainedPickWBCAction` 与现有 WBC checkpoint。
- `TeleopLLAction` 的统一绝对偏移语义：键盘增量由采集脚本累积成与 VR 相同的“标定锚点 + 偏移”，每次 reset 重新标定。
- 低层货架任务的教师观测、目标垫相对位姿、机身姿态和夹爪状态观测。
- 现有 `LowShelfHDF5Recorder`、RGB 相机工厂和 HDF5 检查/回放脚本。

三路相机固定映射到 SmolVLA 的键名，顺序不能按 HDF5 组的字母序推导：

| HDF5 相机组 | LeRobot 键 | 位置 | 作用 |
|---|---|---|---|
| `top_camera` | `observation.images.camera1` | 顶部 `(0.66, 0.08, 1.50)` 向下看 `(0.66, 0.08, 0.34)` | 两个圆柱台、糖盒和目标垫的全局俯视 |
| `wrist_camera` | `observation.images.camera2` | `gripper_base/WristCamera`，向前下约 30° | 抓取阶段的近距离视角 |
| `workcell_camera` | `observation.images.camera3` | 前左 `(0.12, -1.08, 0.80)` 看向 `(0.72, 0.0, 0.42)` | 机器人、两圆柱台和工作区的全局视角 |

官方 SO100 参考数据集只有 `top` + `wrist` 两路（对应 `camera1` + `camera2`），第三路是可选的额外视角，不是 SmolVLA 的硬性要求。当前两圆柱台任务默认采三路，目的是给放置阶段提供顶部全局上下文。

三路默认只采 RGB，不采深度。SmolVLA 的预训练和微调都只接收 RGB，深度不会成为策略输入；`--record_depth` 仅用于几何调试，会增加 GPU readback 和 HDF5 体积。

没有修改低层货架任务本身的训练、遥操或采集语义。

## 2. 语言指令

```text
把糖盒从矮圆柱台放到高圆柱台顶面。
```

## 3. 场景设计

场景由两个静止圆柱台、一个目标垫和一个 YCB 糖盒组成。

| 对象 | 位置/尺寸 | 说明 |
|---|---|---|
| 矮圆柱台 | 中心 `(0.60, -0.22, 0.15)`，半径 `0.20 m`，高 `0.30 m` | 取料台 |
| 高圆柱台 | 中心 `(0.72, 0.38, 0.26)`，半径 `0.20 m`，高 `0.52 m` | 放置台 |
| 目标垫 | `0.13 x 0.13 x 0.008 m`，位于高台顶面 | 成功判定目标 |
| YCB 糖盒 | `004_sugar_box.usd`，绕本地 X 轴 90° | 单物体，可语言描述 |

两圆柱台中心横向错开约 `0.60 m`。机器人从原点附近出发，取料后可到达矮台，但要移动到高台完成放置需要底盘侧移。该距离刻意控制在工作区附近，避免引入长距离导航。

糖盒绕本地 X 轴旋转 90° 后，世界尺寸约为：

```text
0.093 x 0.045 x 0.176 m
```

约 `0.045 m` 的短边朝向夹爪开口，能进入 Piper 实测约 `0.063 m` 的指间空间；底面为长条面，静置时不会像圆柱体一样滚动。

## 4. 奖励与终止

新增任务奖励保留低层货架任务的抓取、抬升、物体到目标距离和成功项，并增加：

- `heading_to_object`：引导底盘朝向取料台。
- `body_pitch_reach`：物体仍在低位时，鼓励轻微前俯。
- `body_height_for_place`：取料阶段目标机身高度约 `0.40 m`，物体抬起后目标高度约 `0.52 m`。

终止项：

```text
time_out
root_height_below_minimum
bad_orientation
object_dropped
place_success
```

放置成功要求：

- 物体曾经被抬升至少 `0.04 m`。
- 物体与目标垫 XY 距离小于 `0.07 m`。
- 物体中心与目标垫高度差小于 `0.035 m`。
- 物体速度足够低。
- 夹爪已张开。
- 条件持续 `0.5 s`。

糖盒旋转后的半高约 `0.088 m`，高台目标垫高度差参数取 `0.088 m`。这是按糖盒落到高台后实际质心高度调过的值，后续若更换糖盒姿态或缩放，必须重新标定。

## 5. 实施步骤

- [x] 检查 `git status`、分支和 `b0b9aaf` 基准提交。
- [x] 阅读低层货架文档、场景、奖励、高层动作、相机、键盘/VR 遥操和 HDF5 实现。
- [x] 新增双圆柱台场景配置。
- [x] 新增 YCB 糖盒、目标垫和相机工厂复用。
- [x] 新增训练、遥操、采集三套任务注册。
- [x] 新增机身高度与 pitch 相关奖励。
- [x] 新增资产静置/高台碰撞探针。
- [x] 新增自动抓取 smoke 探针。
- [x] 完成配置解析。
- [x] 完成 1/4 环境无头 smoke。
- [x] 完成双相机 headless smoke。
- [x] 完成短 GUI/键盘采集 smoke。
- [x] 完成 HDF5 检查。
- [ ] 完成自动稳定抓取抬升，当前仍是已知问题。

## 6. 验收记录

环境：

```text
Windows 10 Pro
NVIDIA RTX A4000
Driver 580.88
Isaac Sim 5.1.0
Isaac Lab 5.1.0
```

### 6.1 配置解析

- 训练任务：`cameras=[]`、`recorders=[]`，动作仍为 `PreTrainedPickWBCAction`。
- 纯遥操任务：`cameras=[]`、`recorders=[]`，动作为 `TeleopLLAction`。
- 采集任务：`cameras=['workcell_camera', 'wrist_camera']`、`recorders=['record_hdf5']`。

### 6.2 资产静置与高台碰撞

探针：`scripts/data/probe_two_cylinder_place.py`

- 糖盒初始中心约 `0.360 m`。
- 30 个 teleop step 后中心约 `0.388 m`。
- 静置线速度约 `0.0025 m/s`，角速度约 `0.025 rad/s`。
- 世界包围盒约 `0.093 x 0.045 x 0.176 m`，与旋转后的糖盒尺寸一致。
- 高台落放后中心约 `0.616 m`，与目标垫高度加半高一致。
- 高台落放后线速度约 `0.001 m/s`，角速度约 `0.016 rad/s`。
- 双圆柱台碰撞正常，没有穿透或掉落。

结论：资产静置稳定，高台碰撞支撑正常。

### 6.3 夹爪适配与抓取

- 实测夹爪指间有效宽度约 `0.063 m`。
- 糖盒短边约 `0.045 m`，几何上可进入指间。
- 自动接近过程中已观察到夹爪与糖盒接触，夹爪关节在闭合时被物体阻挡，说明碰撞链路有效。
- 自动脚本尚未形成稳定夹持和抬升；当前脚本的 IK 终点对齐、姿态保持和推动量仍需调参。

结论：几何适配和碰撞接触已验证，稳定抓取抬升尚未通过。不能把当前状态标记为完整 pick-and-place 成功率通过。

### 6.4 训练 smoke

1 环境和 4 环境、1 iteration 均通过：

- 1 环境：24 env steps，约 `29.3 s`。
- 场景创建约 `1.04 s`。
- 4 环境：96 env steps，约 `33.2 s`。
- 训练仍使用 `PreTrainedPickWBCAction`。
- 无相机、无 recorder。
- 终止项和奖励项均正常初始化。

### 6.5 相机 smoke

命令：

```powershell
python scripts/data/smoke_low_shelf_cameras.py --task Isaac-M20-Piper-TwoCylinder-Collect-v0 --headless --num_envs 1 --steps 3 --output_dir recordings/two_cylinder_camera_smoke_final
```

结果：

- `workcell_camera` 与 `wrist_camera` 均输出 `224 x 224` RGB-D。
- 场景相机同时看到机器人、两个圆柱台、糖盒和目标垫。
- Vulkan 正常。
- 未发现相机或 Vulkan 根因错误。

输出：

```text
recordings/two_cylinder_camera_smoke_final/workcell_camera_rgb.png
recordings/two_cylinder_camera_smoke_final/workcell_camera_depth.png
recordings/two_cylinder_camera_smoke_final/wrist_camera_rgb.png
recordings/two_cylinder_camera_smoke_final/wrist_camera_depth.png
recordings/two_cylinder_camera_smoke_final/camera_smoke.hdf5
```

### 6.6 短 GUI/键盘与 HDF5

命令：

```powershell
python scripts/data/collect_low_shelf_hdf5.py --task Isaac-M20-Piper-TwoCylinder-Collect-v0 --input keyboard --num_envs 1 --max_steps 3 --output recordings/two_cylinder_smoke/keyboard_gui.hdf5 --overwrite
```

结果：

- GUI 正常创建窗口。
- 键盘设备识别为 `Isaac Sim 5.1.0`。
- HDF5 写出 3 step。
- `task=Isaac-M20-Piper-TwoCylinder-Collect-v0`。
- `source=keyboard`。
- 语言指令正确。
- `action=(3, 13)`。
- 两路相机 `rgb=(3,224,224,3)`、`depth=(3,224,224)`、`K=(3,3,3)`、外参 `(3,4,4)`。

## 7. 已知问题

- 自动抓取脚本尚未稳定抬升糖盒，当前只验证了尺寸适配和碰撞接触。后续优先调整末端接近姿态、夹爪闭合时机和底盘/机械臂协同，不应通过放宽成功判定掩盖问题。
- 高台成功高度偏移 `0.088 m` 与当前糖盒旋转姿态绑定。更换物体、缩放或旋转后必须重新测量。
- 取料点已收近到 `x≈0.60 m`，高台仍在 `y≈0.38 m`；底盘侧移价值保留，但需要真实遥操验证最优站位。
- 现有采集脚本文件名仍带 `low_shelf`，任务本身已通过 `--task` 参数复用；后续可统一改成通用名称。
- 尚未完成真实 VR 头显下的完整双圆柱台 episode。

## 8. 复现命令

先激活环境并进入仓库根目录：

```powershell
conda activate env_isaac_lab
```

### 8.1 配置解析

```powershell
python scripts/reinforcement_learning/rsl_rl/probe_task_cfg.py --task Isaac-M20-Piper-TwoCylinder-Place-v0 --headless
python scripts/reinforcement_learning/rsl_rl/probe_task_cfg.py --task Isaac-M20-Piper-TwoCylinder-Teleop-v0 --headless
python scripts/reinforcement_learning/rsl_rl/probe_task_cfg.py --task Isaac-M20-Piper-TwoCylinder-Collect-v0 --headless
```

### 8.2 资产探针

```powershell
python scripts/data/probe_two_cylinder_place.py --headless --settle_steps 30
```

### 8.3 训练 smoke

```powershell
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-M20-Piper-TwoCylinder-Place-v0 --headless --num_envs 1 --max_iterations 1
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-M20-Piper-TwoCylinder-Place-v0 --headless --num_envs 4 --max_iterations 1
```

### 8.4 相机 smoke

```powershell
python scripts/data/smoke_low_shelf_cameras.py --task Isaac-M20-Piper-TwoCylinder-Collect-v0 --headless --num_envs 1 --steps 3 --output_dir recordings/two_cylinder_camera_smoke_final
```

### 8.5 键盘功能测试与采集

键盘与 VR 使用同一动作语义；`action` 中的 `delta_*` 字段表示相对重新标定锚点的偏移，不是逐帧滚动增量。

```powershell
python scripts/data/collect_low_shelf_hdf5.py --task Isaac-M20-Piper-TwoCylinder-Collect-v0 --input keyboard --num_envs 1 --max_steps 3 --output recordings/two_cylinder_smoke/keyboard_gui.hdf5 --overwrite
```

正式键盘采集：

```powershell
python scripts/data/collect_low_shelf_hdf5.py --task Isaac-M20-Piper-TwoCylinder-Collect-v0 --input keyboard --num_envs 1 --output recordings/two_cylinder_place/keyboard.hdf5
```

### 8.6 VR 采集

```powershell
python scripts/data/collect_low_shelf_hdf5.py --task Isaac-M20-Piper-TwoCylinder-Collect-v0 --input vr --num_envs 1 --output recordings/two_cylinder_place/vr.hdf5
```

### 8.7 自动抓取 smoke

该命令用于继续调参，当前预期结果可能为 `FAIL`：

```powershell
python scripts/data/smoke_two_cylinder_grasp.py --headless --max_move_steps 70
```

### 8.8 HDF5 检查

```powershell
python scripts/data/replay_low_shelf_hdf5.py --input recordings/two_cylinder_smoke/keyboard_gui.hdf5 --check-only
```

### 8.9 三路相机目标可见截图

该脚本会把机身降低并移动到矮台预抓取位姿，然后在同一时刻保存三路画面：

```powershell
conda activate env_isaac_lab
python scripts/data/capture_two_cylinder_views.py `
  --headless `
  --num_envs 1 `
  --output_dir recordings/two_cylinder_target_views
```

输出：

```text
recordings/two_cylinder_target_views/camera1_top_rgb.png
recordings/two_cylinder_target_views/camera2_wrist_rgb.png
recordings/two_cylinder_target_views/camera3_workcell_rgb.png
recordings/two_cylinder_target_views/three_view_target_visible.png
```

三联图中三路都能看到糖盒。腕部视角是近距离特写，顶部和前左场景相机负责展示两个圆柱台与目标垫的相对位置。

### 8.10 两圆柱台 VR 采集 -> LeRobot v3

VR 采集（Windows 本机，仓库根目录）：

```powershell
conda activate env_isaac_lab
python scripts/data/collect_low_shelf_hdf5.py `
  --task Isaac-M20-Piper-TwoCylinder-Collect-v0 `
  --input vr `
  --num_envs 1 `
  --output recordings/two_cylinder_place/vr_task.hdf5 `
  --overwrite
```

- 不传 `--output` 时默认写到 `recordings/two_cylinder_place/<YYYYMMDD_HHMMSS>_vr.hdf5`。
- 默认三路 RGB、`224 x 224`、`10 Hz`；控制频率约 `50 Hz`。
- 想更接近 `smolvla_base` 的原始 `256 x 256` 输入，可加 `--camera_width 256 --camera_height 256`；转换和训练命令不用改。`224 x 224` 也已经过完整链路验证。
- 右 B 开始并重新标定；左 Y 保存成功并 reset；左 X 保存失败并 reset；直接退出会把当前 episode 以 `shutdown_manual_stop` 保存。
- 只采两路用于快速验证时可加 `--no-top_camera`；需要深度调试时再加 `--record_depth`。

把 HDF5 拷到远程训练机：

```powershell
scp -P 558 recordings/two_cylinder_place/vr_task.hdf5 `
  root@10.60.144.11:/root/tievnas/hjx/smolvla/datasets/raw/
```

远程转换：

```bash
cd /root/loco-manip-unified-rl-agent
/data/smolvla/venv/bin/python scripts/data/convert_hdf5_to_lerobot.py \
  --input /root/tievnas/hjx/smolvla/datasets/raw/vr_task.hdf5 \
  --output /root/tievnas/hjx/smolvla/datasets/two_cylinder_vr_10hz \
  --repo-id local/two_cylinder_vr_10hz \
  --fps 10 \
  --successful-only \
  --overwrite
```

- `--fps 10` 用 stride=5 取第 `0,5,10,...` 行，去掉 50 Hz 控制循环里重复的相机帧，同时让 action/state/video 对齐。
- `--successful-only` 只保留人工标成成功的 episode；做失败恢复数据时去掉。
- 转换后检查 `meta/info.json` 的特征顺序应为 `camera1, camera2, camera3`，`meta/conversion_info.json` 应记录 `camera_map` 和 `stride=5`。

### 8.11 两圆柱台 SmolVLA 微调

远程新建 `/root/tievnas/hjx/smolvla/train_two_cylinder_vr.yaml`：

```yaml
dataset:
  repo_id: local/two_cylinder_vr_10hz
  root: /root/tievnas/hjx/smolvla/datasets/two_cylinder_vr_10hz
  video_backend: pyav
policy:
  type: smolvla
  pretrained_path: lerobot/smolvla_base
  repo_id: local/smolvla_two_cylinder_vr
  push_to_hub: false
  load_vlm_weights: true
  train_expert_only: true
  train_state_proj: true
  chunk_size: 20
  n_action_steps: 20
output_dir: /root/tievnas/hjx/smolvla/outputs/two_cylinder_vr
job_name: two_cylinder_vr
steps: 20000
batch_size: 1
num_workers: 0
log_freq: 10
eval_steps: 0
save_freq: 1000
wandb:
  enable: false
```

无卡 CPU 先跑 2 步验证数据与训练链路：

```bash
cd /root/tievnas/hjx/smolvla
HF_HOME=/root/tievnas/hjx/smolvla/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/data/smolvla/venv/bin/lerobot-train \
  --config_path=/root/tievnas/hjx/smolvla/train_two_cylinder_vr.yaml \
  --steps=2
```

正式训练时去掉 `--steps=2`，或把 YAML 里的 `steps` 改大。输出在 `output_dir/checkpoints/last/pretrained_model`；中断后可加 `--resume=true`。

注意：

- 不要加 `rename_map`；转换器已经直接输出 `camera1/camera2/camera3`。
- 同一个 LeRobot 数据集必须保持相同的相机集合和键名，不要混用两路旧 HDF5 与三路新 HDF5。
- `push_to_hub: false` 是离线训练必须项。
- 原始 50 Hz HDF5 不要直接按 50 fps 训练；那会让 SmolVLA 看到 5 帧重复画面配不同 action。微调使用 `--fps 10` 去重后的数据即可。

“5 帧重复”的具体含义：控制循环是 `50 Hz`（`control_dt = 0.02 s`），相机 `update_period = 0.1 s`，即相机只有 `10 Hz`。因此原始 HDF5 里每 5 个连续 step 共用同一张 RGB，但对应 5 个不同的 action 和机器人状态。按 50 fps 训练时，模型会看到“画面完全不变、动作却在变”的样本，动作变化率被放大 5 倍，视频时间轴也和真实时间不一致。用 `--fps 10`（stride=5）保留 `0,5,10,...` 帧后，图像、状态、action 和视频时间轴都回到真实的 10 Hz。

### 8.12 特权脚本示例 episode（无需 VR）

真实 VR 数据到位前，可以先用脚本化示例打通“采集 -> LeRobot -> SmolVLA”整条链路。脚本直接读取仿真里的机器人、糖盒和目标垫世界位姿，用关节空间关键帧搜索规划一条“矮台抓取 -> 抬升 -> 底盘侧移 -> 高台放置”的完整路线；13 维动作、三路相机和 HDF5 schema 与 VR/键盘采集完全一致。

```powershell
conda activate env_isaac_lab
python scripts/data/collect_two_cylinder_planned_demo.py `
  --headless --num_envs 1 `
  --base_x 0.30 --base_y -0.16 `
  --output recordings/two_cylinder_place/planned_demo_final.hdf5 `
  --overwrite
```

输出与验证：

```powershell
python scripts/data/replay_low_shelf_hdf5.py `
  --input recordings/two_cylinder_place/planned_demo_final.hdf5 --check-only

python scripts/data/replay_two_cylinder_three_view.py `
  --input recordings/two_cylinder_place/planned_demo_final.hdf5 `
  --episode 0 --stride 5 --fps 10 `
  --output recordings/replay/two_cylinder_planned_three_view.mp4 `
  --frame recordings/replay/two_cylinder_planned_last.png
```

预期：`source=privileged_planner`、`success=True`、`action shape=(N, 13)`、三路 RGB、`depth=not-recorded`。

该示例不是人工遥操：糖盒在夹爪贴近后以运动学方式刚性跟随夹爪（糖盒高而窄，Piper 小开口下的真实摩擦抓取不稳定），底盘 XY 和机身高度也按标定锚点脚本化驱动。它只用于验证数据格式、LeRobot 转换和 SmolVLA 训练流程，不能代替真实抓取成功率统计；正式微调数据仍以 VR 采集为准。

## 9. 后续优先级

1. 用键盘/VR 人工确认矮台可达站位和抓取姿态。
2. 将人工可行的末端位姿与夹爪时机固化到自动抓取 smoke，直到物体抬升超过 `0.08 m`。
3. 完成一次矮台到高台的完整 episode，验证 `place_success`。
4. 用 1/4 环境训练确认奖励和终止统计，再进行完整 MLP RL 训练。
5. 用真实 VR 头显采集至少一条成功 episode，检查 HDF5 回放。
