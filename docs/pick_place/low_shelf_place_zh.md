# M20 + Piper 低层托板取放任务制作流程

**任务分支**：`codex/low-shelf-pick-place`

**目标**：制作第一个可执行的 pick-and-place 场景，并让 M20 的机身高度与 pitch 成为完成任务的必要动作。

**当前状态**：第一版场景和状态教师任务已实现；RL 无头冒烟通过。相机数据采集仍待验证。

## 1. 任务定义

任务 ID：

```text
Isaac-M20-Piper-LowShelf-Place-v0
```

语言描述：

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
- 目标区使用独立、可观测的浅色放置垫。
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
 Δbody_height, Δbody_pitch, Δbody_roll]
```

夹爪继续使用独立的二值开合动作。

低层策略复用现有 `PreTrainedPickWBCAction` 和 WBC checkpoint。新任务只定义场景、物体、目标、观测、奖励和成功条件。

## 4. 观测与成功条件

状态教师观测包含：

- 机器人本体速度、姿态、关节位置和速度。
- 当前机身高度、pitch、roll。
- 当前末端位姿。
- 糖盒相对机器人和末端的位姿。
- 上层目标垫相对机器人和末端的位姿。
- 夹爪状态。
- 上一步动作。

成功条件：

- 糖盒至少被抬离下层一次，抬升高度超过 `0.04 m`。
- 糖盒中心进入上层目标垫的水平容差范围。
- 糖盒高度接近目标垫表面。
- 糖盒速度低于阈值并保持约 `0.5 s`。
- 夹爪已张开。
- 机器人没有倾倒或持续异常碰撞。

## 5. 实施步骤

- [x] 建立任务分支和流程文档。
- [x] 新增双层架、YCB 糖盒、目标垫和固定相机场景。
- [x] 新增目标垫位姿观测、机身姿态观测和夹爪状态观测。
- [x] 新增抓取、搬运、放置奖励及组合成功判定。
- [x] 注册 RL teacher 任务和 VR 遥操任务。
- [ ] 增加动作记录和 HDF5 元数据接口。
- [x] 做资产静置冒烟测试；抓取/放置成功率测试待做。
- [x] 做 1/8 环境无头 smoke test。
- [x] 更新本文档的实测结果和遗留问题。

## 6. 验收记录

2026-11-04 第一版验证：

- 场景创建时间：约 `1.2 s`（1/8 env，Windows + A4000）。
- 物体静置：稳定。糖盒中心最终高度约 `0.321 m`，下层架顶面 `0.300 m`。
- 物体初始位置：默认 `(0.68, -0.15, 0.34)`，随机偏移限制为 x `[-0.04, 0.02]`、y `[-0.03, 0.03]`、z `[0.0, 0.01]`。
- 目标垫：中心 `(0.78, 0.20, 0.555)`。
- 夹爪是否能稳定夹持：尚未做成功率测试。
- 2 iteration 训练：`1 env` 和 `8 env` 均 `EXIT=0`。
- 终止统计：8 env 冒烟中 `object_dropped=0`；随机策略下仍有 `bad_orientation`，属于未训练策略的正常现象。
- 遥操环境：关闭相机时可创建并运行；启用 `--enable_cameras` 的本地无头测试触发 Isaac Sim Vulkan/相机崩溃，待换图形会话或稳定驱动验证。
- 已知问题：相机朝向和固定视角画面尚未做视觉检查；HDF5 记录尚未接入。

## 7. 复现命令

配置解析：

```powershell
conda run --no-capture-output -n env_isaac_lab powershell -NoProfile -Command "& 'D:\nvidia-isaac-sim\IsaacLab-5.1.0\isaaclab.bat' -p scripts\reinforcement_learning\rsl_rl\probe_task_cfg.py --task Isaac-M20-Piper-LowShelf-Place-v0 --headless"
```

2 iteration 训练冒烟：

```powershell
conda run --no-capture-output -n env_isaac_lab powershell -NoProfile -Command "& 'D:\nvidia-isaac-sim\IsaacLab-5.1.0\isaaclab.bat' -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-M20-Piper-LowShelf-Place-v0 --headless --num_envs 8 --max_iterations 2"
```

场景与物体静置探针：

```powershell
conda run --no-capture-output -n env_isaac_lab powershell -NoProfile -Command "& 'D:\nvidia-isaac-sim\IsaacLab-5.1.0\isaaclab.bat' -p scripts\reinforcement_learning\rsl_rl\probe_low_shelf_scene.py --task Isaac-M20-Piper-LowShelf-Place-v0 --headless --num_envs 1 --steps 8"
```

遥操任务（无相机调试）：

```powershell
conda run --no-capture-output -n env_isaac_lab powershell -NoProfile -Command "& 'D:\nvidia-isaac-sim\IsaacLab-5.1.0\isaaclab.bat' -p scripts\reinforcement_learning\rsl_rl\probe_low_shelf_scene.py --task Isaac-M20-Piper-LowShelf-Teleop-v0 --headless --disable_workcell_camera --num_envs 1 --steps 1"
```

## 8. 维护约定

每次完成一个实施步骤，都要同步更新本文件的：

- 当前状态。
- 实施步骤复选框。
- 验收记录。
- 复现命令和已知问题。

