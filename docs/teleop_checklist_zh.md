# M20-Piper 遥操清单（键盘 / VR）

本文面向日常遥操、任务联调和 GUI 冒烟验证。默认命令不再需要重复写分辨率、
`--num_envs`、鱼眼模式、全景后端或预览窗口开关。

## 入口

| 脚本 | 输入 | 用途 |
|---|---|---|
| `scripts/teleoperation/teleop_m20_piper_keyboard.py` | 键盘 | 底盘 + 机身姿态 + 机械臂 + 夹爪的完整 loco-manip 遥操 |
| `scripts/teleoperation/teleop_m20_piper_vr.py` | VR 手柄 | 同一套 13 维高层接口的 VR 遥操 |
| `scripts/teleoperation/teleop_m20_loco_keyboard.py` | 键盘 | 仅底盘/机身姿态，保留旧 locomotion 默认行为 |
| `scripts/data/collect_demonstrations.py` | 键盘 / VR | 带 HDF5 采样的任务数据采集 |

最简单的启动方式：

```powershell
# 键盘：默认 free 主视口 + 四路鱼眼 + 全景/腕部/前视底部窗口
python scripts\teleoperation\teleop_m20_piper_keyboard.py `
  --task Isaac-M20-Piper-Teleop-History-v0

# VR：同样的默认相机和窗口布局
python scripts\teleoperation\teleop_m20_piper_vr.py `
  --task Isaac-M20-Piper-Teleop-History-v0
```

## 默认行为

```text
--num_envs 1
--fisheye-rig four
--fisheye-width 480 --fisheye-height 300
--panorama-width 960 --panorama-height 480
--panorama-backend gpu          # CUDA 不可用时自动回退 CPU
--show-panorama
--show-wrist-camera
--show-front-camera
--camera-view free              # 主视口保留原来的自由相机位置
```

GUI 默认保留主视口的自由视角，并在主视口底部自动排成一行：

```text
[ 全景 560x280 ] [ 腕部 240x240 ] [ 前视 320x240 ]
```

- 默认按 `--width 1280 --height 720` 计算位置，窗口标题和边框也计入尺寸。
- 三个窗口自动居中且互不重叠，不需要手动拖拽。
- 显式传入 `--panorama-window-pos`、`--wrist-window-pos`、`--front-window-pos`
  时才使用指定位置。
- `--camera-view chest/front/both` 会把主视口绑定到前视相机；`wrist` 会绑定到腕部。
  这时不会再额外打开同视角的底部窗口。
- headless 模式会自动关闭全景、腕部和前视预览，不会因为默认开关而报错。

切换鱼眼：

```powershell
# 前后两路，替代默认四路
python scripts\teleoperation\teleop_m20_piper_keyboard.py `
  --task Isaac-M20-Piper-Teleop-History-v0 `
  --fisheye-rig two

# 完全关闭鱼眼和全景
python scripts\teleoperation\teleop_m20_piper_keyboard.py `
  --task Isaac-M20-Piper-Teleop-History-v0 `
  --fisheye-rig none
```

`two` 和 `dual` 等价，都是 `fisheye_front` / `fisheye_back`；`four` 是
`fisheye_cam0..fisheye_cam3` 的水平环视。

## 可打开的选项

### 任务与运行

| 选项 | 默认 | 说明 |
|---|---:|---|
| `--task` | 必填 | Gym 任务 ID，见下方任务清单 |
| `--num_envs` | 1 | 遥操只需要单环境 |
| `--device` | 单环境默认 CPU | 显式 `--device cuda:0` 可强制 GPU 物理 |
| `--headless` | 关闭 | 无 GUI；预览窗口自动关闭 |
| `--width` / `--height` | 1280 / 720 | GUI 尺寸和自动窗口排布基准 |
| `--render-interval` | 4 | 每 4 个物理步渲染一次 |
| `--max-steps` | 0 | 0 表示一直运行；用于 smoke test 时设小值 |

### 主视口

| 选项 | 键盘可选值 | VR 可选值 | 说明 |
|---|---|---|---|
| `--camera-view` | `free` / `chest` / `front` / `wrist` / `both` | `free` / `chest` / `wrist` / `both` | `free` 为默认自由视角；其他值把主视口绑定到机器人相机 |

### 鱼眼与全景

| 选项 | 默认 | 说明 |
|---|---:|---|
| `--fisheye-rig` | `four` | `none` / `two` / `dual` / `four` |
| `--fisheye-width` / `--fisheye-height` | 480 / 300 | 单路鱼眼分辨率 |
| `--fisheye-fov-deg` | 200 | 单路理想鱼眼全 FOV |
| `--fisheye-mount-pos X Y Z` | `-0.44 0.0 0.34` | 相对 `base_link` 的安装点 |
| `--fisheye-lens-offset` | 0.02 | 双路前后镜头半间距（米） |
| `--fisheye-update-period` | 0.0 | 传感器更新周期；0 跟随预览/采样频率 |
| `--show-panorama` / `--no-show-panorama` | 开启 | 是否打开拼接全景窗口 |
| `--panorama-width` / `--panorama-height` | 960 / 480 | 拼接全景内部等距柱面分辨率 |
| `--panorama-backend` | `gpu` | `auto` / `gpu` / `cpu`；GPU 不可用时回退 CPU |
| `--panorama-display-hz` | 12 | 全景 UI 刷新率 |
| `--panorama-window-width` / `--panorama-window-height` | 560 / 280 | 全景窗口显示尺寸 |
| `--panorama-window-pos X Y` | 自动 | 覆盖默认底部位置 |
| `--record-panorama` / `--no-record-panorama` | 关闭 | 采集时额外保存拼接全景 |

### 腕部与前视预览

| 选项 | 默认 | 说明 |
|---|---:|---|
| `--show-wrist-camera` / `--no-show-wrist-camera` | 开启 | 腕部相机窗口 |
| `--wrist-display-hz` | 20 | 腕部 UI 刷新率 |
| `--wrist-window-width` / `--wrist-window-height` | 240 / 240 | 腕部窗口显示尺寸 |
| `--wrist-window-pos X Y` | 自动 | 覆盖默认底部位置 |
| `--show-front-camera` / `--no-show-front-camera` | 开启 | 前视相机窗口 |
| `--front-display-hz` | 20 | 前视 UI 刷新率 |
| `--front-window-width` / `--front-window-height` | 320 / 240 | 前视窗口显示尺寸 |
| `--front-window-pos X Y` | 自动 | 覆盖默认底部位置 |

### Fast path 与 profiling

| 选项 | 默认 | 说明 |
|---|---:|---|
| `--fast-teleop` / `--no-fast-teleop` | 开启 | 使用轻量遥操热路径 |
| `--fast-render-hz` | 15 | 墙钟渲染频率 |
| `--fast-write-each-tick` | 关闭 | 每个 5 ms 物理 tick 都写目标；默认每 20 ms 写一次 |
| `--fast-robot-hot-loop` | 开启 | 机械臂目标按 hot period 更新 |
| `--fast-robot-hot-period` | 0.01 | hot loop 周期（秒） |
| `--fast-robot-only-write` | 关闭 | 只写 robot articulation；仅无其他被控 articulation 时使用 |
| `--ik-update-period` | 0.01 | IK 重算周期（仿真秒） |
| `--profile-detailed` | 关闭 | 拆分 action/policy/IK/scene/sim/render 计时 |
| `--profile-json` | 无 | 输出 profiling JSON |
| `--profile-interval` | 5.0 | JSON 快照间隔（秒） |
| `--ui-snapshot-dir` | 无 | 保存主 viewport、全景/腕部/前视 UI 验证图 |

### VR 专用

| 选项 | 默认 | 说明 |
|---|---:|---|
| `--relative-ik` | 关闭 | 用每步相对位姿 IK；默认绝对目标 IK |
| `--arm-clutch` / `--body-clutch` | 开启 | 兼容旧命令的无操作开关 |
| `--no-stick-shaping` | 关闭 | 兼容旧命令的无操作开关 |

## 键盘控制

### 完整键盘遥操

`teleop_m20_piper_keyboard.py`：

| 功能 | 按键 |
|---|---|
| 前进 / 后退 `vx` | `↑` / `↓` |
| 左移 / 右移 `vy` | `←` / `→` |
| 左转 / 右转 `ωz` | `Z` / `X` |
| 升高 / 降低机身 | `C` / `V` |
| 抬头 / 低头 | `B` / `N` |
| 左倾 / 右倾 | `[` / `]` |
| 末端 `x` 前后 | 小键盘 `8` / `2` |
| 末端 `y` 左右 | 小键盘 `4` / `6` |
| 末端 `z` 上下 | 小键盘 `7` / `9` |
| 末端 roll | 小键盘 `1` / `3` |
| 末端 pitch | 小键盘 `0` / `.` |
| 末端 yaw | 小键盘 `+` / `-` |
| 夹爪开/合 | `G`（切换） |
| 重置环境 | `L` |

### 仅底盘键盘遥操

`teleop_m20_loco_keyboard.py` 使用另一套按键：

| 功能 | 按键 |
|---|---|
| 前进 / 后退 | `W` / `S` |
| 左移 / 右移 | `A` / `D` |
| 左转 / 右转 | `Q` / `E` |
| 升高 / 降低机身 | `R` / `F` |
| 抬头 / 低头 | `T` / `G` |
| 左倾 / 右倾 | `Y` / `H` |
| 重置姿态 | `P` |
| 重置环境 | `L` |

## VR 控制

| 功能 | 操作 |
|---|---|
| 前后 / 侧移 | 左摇杆 Y / X |
| 左转 / 右转 | 右摇杆 X |
| 机身升降 | 左手 Z 移动；按住左扳机可持续升高 |
| 机身 pitch / roll | 左手旋转 |
| 机械臂位置 | 右手位置偏移（按住右 grip 时跟随，松开冻结） |
| 机械臂姿态 | 右手旋转偏移（按住右 grip） |
| 夹爪 | 右扳机：按住开爪，松开合爪 |
| 开始控制 + 重新标定 | 右 `B` |
| 重置环境 + 命令清零 | 左 `X` / 左 `Y` |

默认 `arm_clutch_enabled=True`、`body_clutch_enabled=True`，所以右 grip 控制机械臂离合，
左 grip 控制机身姿态离合；这样推摇杆时不会把腕部抖动串到机身姿态。

## 任务清单

命名约定：

- `*-Teleop-v0`：纯遥操调试，通常不启用 HDF5 recorder。
- `*-Collect-v0`：采集任务，启用相机和 HDF5 recorder。
- 多数任务还有 `*-Teacher-v0`：只提供状态观测，用于教师策略训练。
- `LowShelf` / `TwoCylinder` 的训练变体使用 `*-Place-v0`，不是 `Teacher`。

| 代号 | Teleop task | 场景 / 目标 | Teleop 回合 |
|---|---|---|---:|
| 通用调试 | `Isaac-M20-Piper-Teleop-v0` | 平地高层抓取/操作基础场景；验证底盘、机身、机械臂、夹爪和相机 | 由任务配置决定 |
| 通用 + history | `Isaac-M20-Piper-Teleop-History-v0` | 同上，但低层默认使用 history encoder 部署策略；日常遥操默认推荐 | 由任务配置决定 |
| LowShelf | `Isaac-M20-Piper-LowShelf-Teleop-v0` | 双层架；把下层托盘里的糖盒放到上层桌面，需要降低机身并前俯 | 60 s |
| TwoCylinder | `Isaac-M20-Piper-TwoCylinder-Teleop-v0` | 矮圆柱台取糖盒，放到高圆柱台目标垫；需要底盘侧移和抬升机身 | 60 s |
| P1 RearPick | `Isaac-M20-Piper-RearPick-Teleop-v0` | 目标在机器人后方 120°–240°，初始只被全景看到；全景发现→绕行→前视/腕部重捕获→抓取→放托盘 | 120 s |
| P2 BlindIntercept | `Isaac-M20-Piper-BlindIntercept-Teleop-v0` | 目标沿 U 形/圆弧/传送带运动，经过侧后方 1–3 s 盲区；全景连续跟踪并移动拦截抓取 | 120 s |
| P3 PeekOver | `Isaac-M20-Piper-PeekOver-Teleop-v0` | 0.6–1.0 m 隔断后的目标；利用高位全景越过隔断，选正确绕行侧，抓取指定目标并放置 | 120 s |
| W1 DrawerLadder | `Isaac-M20-Piper-DrawerLadder-Teleop-v0` | 三层抽屉（约 0.2/0.6/1.0 m）；拉开目标抽屉、取物、放托盘、再把抽屉推回 | 150 s |
| W2 DoorBrace | `Isaac-M20-Piper-DoorBrace-Teleop-v0` | 带弹簧/阻尼的门；需要轮制动、轮速差、机身 pitch/roll 或接触支撑，不能只靠机械臂拉门 | 180 s |
| W5 HighLowServe | `Isaac-M20-Piper-HighLowServe-Teleop-v0` | 同一回合先取 0.1–0.3 m 低位目标，再取 1.2–1.5 m 高位目标，均放入同一托盘 | 180 s |
| L1 HomeMission | `Isaac-M20-Piper-HomeMission-Teleop-v0` | 长时程家庭任务，例如“把右侧柜子的蓝色盒子放到门后篮子并关柜”；组合导航、开门/抽屉、抓取和恢复 | 240 s |
| R1 FrontDeskPickup | `Isaac-M20-Piper-FrontDeskPickup-Teleop-v0` | Office 接待厅；接待台上的糖盒在初始前视外，用全景搜索，取物并放到访客侧矮桌托盘 | 120 s |
| H1 WardDelivery | `Isaac-M20-Piper-WardDelivery-Teleop-v0` | Hospital 病房；从约 1.32 m 高药架取药，绕过病床/轮椅，放到床头柜约 0.90 m 的托盘 | 150 s |
| K1 KitchenDrawerServe | `Isaac-M20-Piper-KitchenDrawerServe-Teleop-v0` | KitchenRoom；拉开低位抽屉、低姿取物，受上层柜门空间约束，放到台面托盘 | 150 s |
| H2 SupplyRestock | `Isaac-M20-Piper-SupplyRestock-Teleop-v0` | HospitalSupply；从约 1.40 m 高耗材架取物，绕过推车/病床/轮椅，放到地面低位托盘；允许一次重抓 | 180 s |

### 任务成功条件要点

| 任务 | 关键判据 |
|---|---|
| RearPick | 目标在托盘内稳定约 1 s；90 s 内完成；机器人未摔倒 |
| PeekOver | 抓取指定目标、未先碰倒错误目标、放入目标区；90 s 内完成 |
| BlindIntercept | 抓住运动目标，抬离运动表面至少 10 cm，保持约 1 s |
| DrawerLadder | 目标抽屉位移 > 0.15 m、取出目标、托盘中稳定、抽屉最终关回 5 cm 内 |
| DoorBrace | 门开角 > 75°，机器人整体通过门框，未发生非物理破坏或超限接触力 |
| HighLowServe | 高低两个目标都在托盘中稳定，托盘未翻倒 |
| HomeMission | 必要子目标全部完成，目标进入指定容器，关节物体恢复要求状态 |
| LowShelf / TwoCylinder | 目标进入目标垫/目标区并稳定由 `place_success` 判定 |

R1/H1/K1/H2 目前是“可遥操、可采集、可复现”的场景工程基线；其完整 MDP 成功判定和
长时程子目标仍在逐步补齐，详见
[`WBC-OmniBench-环境主题任务实现记录.md`](WBC-OmniBench-环境主题任务实现记录.md)。

## 推荐命令

### 任务专用场景

```powershell
python scripts\teleoperation\teleop_m20_piper_keyboard.py `
  --task Isaac-M20-Piper-RearPick-Teleop-v0
```

### 关闭全景，只保留四路鱼眼 + 腕部 + 前视

```powershell
python scripts\teleoperation\teleop_m20_piper_vr.py `
  --task Isaac-M20-Piper-Teleop-History-v0 `
  --no-show-panorama
```

### Headless 采集冒烟

```powershell
python scripts\data\collect_demonstrations.py `
  --task Isaac-M20-Piper-LowShelf-Collect-v0 `
  --input keyboard --headless --max_steps 20 `
  --no-scene_camera --no-wrist_camera --no-top_camera `
  --fisheye-rig four `
  --output recordings\smoke\four_fisheye.hdf5 --overwrite
```

### 验证 GUI 窗口

```powershell
python scripts\teleoperation\teleop_m20_piper_keyboard.py `
  --task Isaac-M20-Piper-Teleop-History-v0 `
  --max-steps 20 `
  --ui-snapshot-dir outputs\teleop_ui_smoke
```

退出后会生成 `main_viewport.png`、`panorama_ui.png`、`wrist_ui.png`、`front_ui.png`
和 `ui_status.json`，可用于确认三个预览窗口都可见且持续更新。

## 性能与限制

- 默认四路鱼眼 + 全景 + 腕部 + 前视是最完整但最重的组合；显卡压力大时可先切
  `--fisheye-rig two`，或 `--fisheye-rig none`。
- 全景默认走 GPU；没有 CUDA 时会打印一次回退提示并使用 CPU。
- 自动布局按启动时的 `--width/--height` 计算；如果桌面实际可用区域更小，仍可用
  三个 `--*-window-pos` 手动覆盖。
- 四路鱼眼仍对应四个独立 render product，尚未合并为单个 tiled camera。
