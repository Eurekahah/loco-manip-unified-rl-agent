# WBC-OmniBench 七场景实现记录

版本：v0.4  
日期：2026-11-18  
渲染：Isaac Sim 5.1，`rendering_mode=quality`

最近验证：2026-11-18 20:08–20:36（Asia/Shanghai）。P1、P2、P3、W1、W2、W5、L1
已逐个使用最新场景配置完成 headless quality 渲染，进程退出码均为 0。截图直接覆盖
写入文档引用的 `recordings/scene_views/<task>/` 目录，不是临时目录或仅更新文字。

## 本轮更新摘要

- 七个通用核心任务统一使用 Isaac 官方 `Simple_Room` 作为封闭室内基础场景。
- `Simple_Room` 整体抬升 `0.78 m`，使官方木地板与仿真地面 `z=0` 对齐。
- startup event 隐藏 `Simple_Room` 自带的中央展台，避免与机器人出生点重叠。
- 室内光照改为顶部球灯 + `hospital_room_4k.hdr` Dome IBL。
- 桌面、门板、货架、柜体和金属支架改用 Isaac 自带 MDL 纹理材质。
- 前视相机前移到 `x=0.58 m`，避免 M20 机身和前轮遮挡。
- 前视、腕部、workcell 相机统一改为 `(0.001 m, 1.0e9 m)` 裁剪范围。
- 七个场景的俯视、四视角、全景和鱼眼截图均已重新生成。
- P2 改为 Isaac 官方 `ConveyorBelt_A05.usd`，并用 PhysX
  `PhysxSurfaceVelocityAPI` 驱动带面，不再用 Cuboid 拼装传送带。
- W2 改为 Isaac 官方 `Sektion_Cabinet` articulation，复用已有
  `door_left_joint`/`door_right_joint` 铰链，不再使用静态门板 Cuboid。

> 上述 `Simple_Room` 约定只适用于当前七个通用核心任务。接待厅、医院和厨房主题任务
> 必须分别在 `Office`、`Hospital` 和独立 `KitchenRoom` 场景资产中制作任务处理资产，
> **不得加载 `Simple_Room` 代替主题场景**。

## 通用场景约定

### 基础房间

- 资产：`Environments/Simple_Room/simple_room.usd`
- 地面偏移：`z=0.78 m`
- 房间地板：官方木地板纹理
- 墙面：官方白色墙面和木饰面
- 天花板：官方天花板和灯具造型
- 隐藏 prim：`/World/WBCRoom/table_low_327`

### 光照与渲染

- 顶部球灯：2 盏，`intensity=12000`，半径 `0.28 m`
- Dome IBL：`NVIDIA/Assets/Skies/Indoor/hospital_room_4k.hdr`，`intensity=6000`
- Dome 不参与主射线可见性，避免窗外出现不属于当前房间的 HDRI
- 质量截图启用：
  - direct lighting
  - global illumination
  - reflections
  - shadows
  - DL denoiser
  - DLAA
  - 4 samples per pixel

### 纹理材质

| 用途 | Isaac 材质 |
|---|---|
| 桌面、门板、货架 | `Materials/Base/Wood/Oak.mdl` |
| 柜体、门框、隔断 | `Materials/Base/Miscellaneous/Paint_Matte.mdl` |
| 桌腿、支撑、导轨 | `Materials/Base/Metals/Steel_Stainless.mdl` |
| 传送带 | `Materials/Base/Plastics/Plastic_ABS.mdl` |

地板、墙面、天花板和房间木饰面直接使用 `Simple_Room` 自带纹理。

### 相机

| 相机 | 安装位置 | 裁剪范围 | 说明 |
|---|---|---|---|
| 前视 | `base_link` 前向 `(0.58, 0.0, 0.12)` | `(0.001, 1.0e9)` | 机身最大前向包络约 `0.434 m`，保留约 `0.15 m` 净空 |
| 腕部 | `gripper_base` | `(0.001, 1.0e9)` | 近距离接触物体不会再被裁成背景 |
| workcell | 世界坐标固定安装 | `(0.001, 1.0e9)` | 远裁剪等价无限远 |
| 顶部 | 世界坐标固定安装 | `(0.1, 4.0–6.0)` | 保持原有全局俯视范围 |

### 现成实现复用

本轮先检索并验证已有实现，再决定是否自行建模：

- 官方传送带资产：`Props/Conveyors/ConveyorBelt_A01–A49.usd`。
- 官方 PhysX 表面速度接口：
  `PhysxSchema.PhysxSurfaceVelocityAPI`。
- 官方铰链柜体：`Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd`。
- 开源 Isaac Lab 开门任务：
  [soom1017/isaaclab_door_open](https://github.com/soom1017/isaaclab_door_open)。
- 开源铰链门 benchmark：
  [PatrizioAcquadro/AlexDoor-XAS](https://github.com/PatrizioAcquadro/AlexDoor-XAS)。
- 传送带速度控制参考：
  [Derman0524/isaac-sim-mqtt-conveyor](https://github.com/Derman0524/isaac-sim-mqtt-conveyor)。

本轮没有直接复制上述仓库代码。门使用本地官方 articulation 资产；传送带使用本地官方
USD 和 PhysX 官方 API。

## 验证矩阵

| 任务 | Gym task 前缀 | 最新 headless 渲染 | 截图状态 | 主要未完成项 |
|---|---|---|---|---|
| P1 RearPick | `Isaac-M20-Piper-RearPick` | 通过 | 已更新 | 视觉策略和正式 benchmark 指标 |
| P2 BlindIntercept | `Isaac-M20-Piper-BlindIntercept` | 通过 | 已更新 | 变速/反向/启停命令 |
| P3 PeekOver | `Isaac-M20-Piper-PeekOver` | 通过 | 已更新 | 正式 evaluator 和视觉策略 |
| W1 DrawerLadder | `Isaac-M20-Piper-DrawerLadder` | 通过 | 已更新 | 多抽屉顺序和联合成功判定 |
| W2 DoorBrace | `Isaac-M20-Piper-DoorBrace` | 通过 | 已更新 | 完整房间门资产、门轴阻力和接触指标 |
| W5 HighLowServe | `Isaac-M20-Piper-HighLowServe` | 通过 | 已更新 | 高低双目标联合评分 |
| L1 HomeMission | `Isaac-M20-Piper-HomeMission` | 通过 | 已更新 | 子目标 evaluator、语言目标和恢复逻辑 |

所有任务均提供 `Teacher / Teleop / Collect-v0` 入口；当前记录只验证场景和采集工程链路，
不代表视觉策略或完整任务成功率已经验收。

### 最新截图文件核对

下表是 2026-11-18 20:08–20:36 实际重跑后的四视角拼图文件。哈希为 SHA-256 前 16 位，
用于确认图片内容已重新写入。

| 任务 | 四视角截图 | 修改时间 | 大小 | SHA-256 前缀 |
|---|---|---|---:|---|
| P1 RearPick | `rear_pick_initial_montage.png` | 20:08:33 | 1,258,188 | `9FBED0E35F36ABD3` |
| P2 BlindIntercept | `blind_intercept_montage.png` | 20:35:20 | 1,396,511 | `CC912D4F4485AA20` |
| P3 PeekOver | `peek_over_initial_montage.png` | 20:11:23 | 1,172,322 | `B4645FCEF6D626BC` |
| W1 DrawerLadder | `drawer_ladder_montage.png` | 20:12:33 | 1,058,189 | `32250639A71D09EF` |
| W2 DoorBrace | `door_brace_montage.png` | 20:36:24 | 979,955 | `43F02B1F0515CDFD` |
| W5 HighLowServe | `high_low_serve_montage.png` | 20:14:59 | 1,302,559 | `CC537926B5C3C26D` |
| L1 HomeMission | `home_mission_montage.png` | 20:16:07 | 1,267,255 | `0FF9451F279718E5` |

本轮重跑后，七个文件的尺寸、哈希和修改时间均已变化；文档中的全部截图链接也已逐项
验证存在。

---

## P1 RearPick

当前实现：

- 后方工作桌、四条桌腿、蓝色托盘和 YCB sugar box。
- 机器人初始朝向使目标不进入前视相机，仅由全景相机发现。
- 重捕获脚本会把机器人移动到后方桌前的可达位置，并采集前视和腕部图像。
- 最新腕部截图能在极近距离看到 sugar box，不再退化为背景。

俯视：

![P1 RearPick 俯视](../recordings/scene_views/rear_pick/rear_pick_initial_top_camera.png)

四视角：

![P1 RearPick 四视角](../recordings/scene_views/rear_pick/rear_pick_initial_montage.png)

全景：

![P1 RearPick 全景](../recordings/scene_views/rear_pick/rear_pick_initial_panorama.png)

重捕获：

![P1 RearPick 重捕获](../recordings/scene_views/rear_pick/rear_pick_reacquired_montage.png)

近距离腕部：

![P1 RearPick 近距离腕部](../recordings/scene_views/rear_pick/rear_pick_reacquired_wrist_camera.png)

---

## P2 BlindIntercept

当前实现：

- 使用 Isaac 官方 `ConveyorBelt_A05.usd`，包含完整纹理、滚筒和支撑结构。
- 官方外观网格使用静态碰撞时会与运动学带面互相卡住，因此预启动阶段关闭其碰撞，
  在同一带面高度附加不可见运动学碰撞板。
- `prestartup` event 为运动学带面应用 PhysX `PhysxSurfaceVelocityAPI`，速度
  `0.35 m/s`，并沿资产本地 `+X`。
- 目标 sugar box 从 `x≈1.35` 连续移动到 `x≈2.58`，到末端后自动回到
  `x≈1.12` 继续循环，方便持续遥操和可视化。
- 当前只验证了单向恒速；变速、反向和启停命令仍待接入。
- 最新四视角和全景已包含新光照、材质和相机裁剪。

俯视：

![P2 BlindIntercept 俯视](../recordings/scene_views/blind_intercept/blind_intercept_top_camera.png)

四视角：

![P2 BlindIntercept 四视角](../recordings/scene_views/blind_intercept/blind_intercept_montage.png)

全景：

![P2 BlindIntercept 全景](../recordings/scene_views/blind_intercept/blind_intercept_panorama.png)

---

## P3 PeekOver

当前实现：

- `1.0 m` 高隔断、后方工作桌、目标物体和托盘。
- 初始视角用于验证高位全景相机越过隔断发现目标。
- 侧向绕行截图用于验证绕行方向和局部相机重捕获。
- 最新初始和侧向绕行截图均已重新生成。

俯视：

![P3 PeekOver 俯视](../recordings/scene_views/peek_over/peek_over_initial_top_camera.png)

四视角：

![P3 PeekOver 四视角](../recordings/scene_views/peek_over/peek_over_initial_montage.png)

全景：

![P3 PeekOver 全景](../recordings/scene_views/peek_over/peek_over_initial_panorama.png)

侧向绕行：

![P3 PeekOver 侧向绕行](../recordings/scene_views/peek_over/peek_over_side_approach_montage.png)

---

## W1 DrawerLadder

当前实现：

- Isaac `Sektion_Cabinet` articulation、抽屉/柜门关节、目标物体和托盘。
- 使用统一前视、腕部、workcell 和顶部相机。
- 当前场景已完成基础关节和采集入口，多抽屉顺序和联合成功判定仍待补。

俯视：

![W1 DrawerLadder 俯视](../recordings/scene_views/drawer_ladder/drawer_ladder_top_camera.png)

四视角：

![W1 DrawerLadder 四视角](../recordings/scene_views/drawer_ladder/drawer_ladder_montage.png)

全景：

![W1 DrawerLadder 全景](../recordings/scene_views/drawer_ladder/drawer_ladder_panorama.png)

---

## W2 DoorBrace

当前实现：

- 使用 Isaac 官方 `Sektion_Cabinet/sektion_cabinet_instanceable.usd` 作为带铰链门
  articulation，复用 `door_left_joint` 和 `door_right_joint`。
- 两个门关节为低阻尼被动关节；抽屉关节使用高刚度保持关闭，避免无意图移动。
- 运行态探针将两个门关节分别驱动到约 `1.135 rad` 和 `-1.255 rad`，确认铰链可实际
  转动，不只是视觉模型。
- 当前为模块化柜门尺寸，尚未替换为一扇完整房间门；门轴阻力、接触力和通过门框判定
  仍待补。

俯视：

![W2 DoorBrace 俯视](../recordings/scene_views/door_brace/door_brace_top_camera.png)

四视角：

![W2 DoorBrace 四视角](../recordings/scene_views/door_brace/door_brace_montage.png)

全景：

![W2 DoorBrace 全景](../recordings/scene_views/door_brace/door_brace_panorama.png)

近距离前视：

![W2 DoorBrace 近距离前视](../recordings/scene_views/door_brace/door_brace_front_camera.png)

---

## W5 HighLowServe

当前实现：

- 低位台、高位货架、两个 YCB 目标物体和托盘。
- 桌面和货架使用 Oak 纹理，支撑使用不锈钢纹理。
- 当前成功判定仍主要使用主目标，高低双目标联合评分待补。

俯视：

![W5 HighLowServe 俯视](../recordings/scene_views/high_low_serve/high_low_serve_top_camera.png)

四视角：

![W5 HighLowServe 四视角](../recordings/scene_views/high_low_serve/high_low_serve_montage.png)

全景：

![W5 HighLowServe 全景](../recordings/scene_views/high_low_serve/high_low_serve_panorama.png)

---

## L1 HomeMission

当前实现：

- Sektion 柜体、门框、门板、蓝色目标盒和目的地篮子。
- 门板使用 Oak 纹理，门框使用 matte paint 材质。
- 当前为组合场景工程基线，子目标 evaluator、语言目标解析和失败恢复逻辑待补。

俯视：

![L1 HomeMission 俯视](../recordings/scene_views/home_mission/home_mission_top_camera.png)

四视角：

![L1 HomeMission 四视角](../recordings/scene_views/home_mission/home_mission_montage.png)

全景：

![L1 HomeMission 全景](../recordings/scene_views/home_mission/home_mission_panorama.png)

---

## 复现命令

P1：

```powershell
conda run --no-capture-output -n env_isaac_lab python -u `
  scripts/data/capture_rear_pick_views.py `
  --headless --output_dir recordings/scene_views/rear_pick
```

P3：

```powershell
conda run --no-capture-output -n env_isaac_lab python -u `
  scripts/data/capture_peek_over_views.py `
  --headless --output_dir recordings/scene_views/peek_over
```

其余场景：

```powershell
conda run --no-capture-output -n env_isaac_lab python -u `
  scripts/data/capture_task_scene_views.py `
  --headless `
  --task Isaac-M20-Piper-DrawerLadder-Teleop-v0 `
  --output_dir recordings/scene_views/drawer_ladder
```

将 `--task` 和 `--output_dir` 替换为对应场景即可。

## 本轮验证结果

- P1 RearPick：headless quality 渲染通过，初始和重捕获截图已更新。
- P2 BlindIntercept：官方传送带和 PhysX surface velocity 运行通过，截图已更新。
- P3 PeekOver：headless quality 渲染通过，初始和侧向绕行截图已更新。
- W1 DrawerLadder：headless quality 渲染通过，截图已更新。
- W2 DoorBrace：官方 Sektion 铰链 articulation 运行通过，门关节可驱动，截图已更新。
- W5 HighLowServe：headless quality 渲染通过，截图已更新。
- L1 HomeMission：headless quality 渲染通过，截图已更新。
- 修改文件通过 `python -m py_compile`。
- 本轮没有运行完整 `pytest`；验证范围是场景构建、相机渲染和现有工程入口。

## 已知限制

- P2 当前已由官方传送带驱动，但只支持单向恒速，尚未接入变速、反向和启停命令。
- W2 当前复用官方模块化柜门 articulation，尚未替换为完整房间门资产，门轴阻力和
  接触力指标仍待补。
- W5 当前成功判定仍只使用主目标，尚未实现高低两个目标的联合评分。
- L1 已具备组合场景，但子目标 evaluator、语言目标和恢复逻辑尚未实现。
- 所有场景目前是 Teacher/Teleop/Collect 的工程基线，视觉学生和正式 benchmark 指标仍待补。
- 前视、腕部、workcell 的远裁剪虽然设为 `1.0e9 m`，实际训练和评测仍应关注相机深度精度与传感器噪声标定。
- 官方 ConveyorBelt USD 在 Isaac 5.1 下会提示 `Material%20Library/physics_material.usd`
  子层未加载；渲染和带面驱动正常，但传送带专用物理材质需要后续修复或本地覆盖。

## 后续环境主题任务

环境主题任务不使用 `Simple_Room`。接待厅、医院和厨房任务分别在 `Office`、
由官方 `Hospital` 裁剪的 `HospitalWard` 和独立 `KitchenRoom` 场景组合中制作对应
任务处理资产。设计见
[WBC-OmniBench 环境主题任务设计](WBC-OmniBench-环境主题任务设计.md)。

首轮环境主题任务已实现：

| 任务 | Gym task | 主题场景 | 验证 |
|---|---|---|---|
| R1 FrontDeskPickup | `Isaac-M20-Piper-FrontDeskPickup-*-v0` | `Office` | quality 四视角、全景、鱼眼通过 |
| H1 WardDelivery | `Isaac-M20-Piper-WardDelivery-*-v0` | `Hospital` | quality 四视角、全景、鱼眼通过 |
| K1 KitchenDrawerServe | `Isaac-M20-Piper-KitchenDrawerServe-*-v0` | 独立 `KitchenRoom` | quality 四视角、全景、鱼眼通过 |
| H2 SupplyRestock | `Isaac-M20-Piper-SupplyRestock-*-v0` | `Hospital` | quality 四视角、全景、鱼眼通过 |

实现细节、开源检索和场景坐标见
[WBC-OmniBench 环境主题任务实现记录](WBC-OmniBench-环境主题任务实现记录.md)。
