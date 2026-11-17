# 任务隔离的数据采集架构

数据采集入口与具体任务已经解耦。新增任务时不再复制采集循环，只新增任务环境和
collection profile。

## 分层

```text
scripts/data/collect_demonstrations.py
  └─ scripts/data/collection/cli.py
       ├─ 通用 CLI、输入设备、相机、鱼眼、HDF5 参数
       └─ scripts/data/collection/runner.py
            ├─ 输入设备与主循环
            ├─ 统一采样、异步 writer、UI、profiling
            └─ scripts/data/collection/profiles.py
                 ├─ builtin_profiles.py
                 │    ├─ low_shelf
                 │    └─ two_cylinder
                 └─ 外部 --profile-module 注册的任务 profile

source/rl_training/rl_training/recorders/hdf5.py
  └─ 任务无关 recorder，按 profile 声明采集实体和 action term

source/rl_training/rl_training/datasets/hdf5.py
  └─ 通用 teleop_manipulation/v1 流式 writer
```

任务目录只负责环境、场景、相机工厂、奖励和终止条件；采集循环不知道 low-shelf 或
two-cylinder 的具体对象名。

## 通用命令

自动根据 task id 选择 profile：

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts\data\collect_demonstrations.py `
  --task Isaac-M20-Piper-LowShelf-Collect-v0 `
  --input vr --num_envs 1 `
  --record-hz 10 --compression lzf --async-write `
  --output recordings\low_shelf_place\vr.hdf5
```

```powershell
python scripts\data\collect_demonstrations.py `
  --task Isaac-M20-Piper-TwoCylinder-Collect-v0 `
  --input keyboard --num_envs 1 `
  --record-hz 10 --compression lzf --async-write `
  --output recordings\two_cylinder_place\keyboard.hdf5
```

显式选择 profile：

```powershell
python scripts\data\collect_demonstrations.py `
  --task Isaac-M20-Piper-TwoCylinder-Collect-v0 `
  --profile two_cylinder
```

旧的任务专用采集入口已经移除。所有任务统一使用
`collect_demonstrations.py`。

## 新增任务

### 1. 任务环境

在任务 env cfg 中使用共享 recorder：

```python
from rl_training.recorders import HDF5RecorderManagerCfg


@configclass
class MyTaskCollectEnvCfg(MyTaskTeleopEnvCfg):
    recorders: HDF5RecorderManagerCfg = HDF5RecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        cfg = self.recorders.record_hdf5
        cfg.task_profile = "my_task"
        cfg.action_term_name = "pre_trained_pick_action"
        cfg.tracked_entity_names = ("object", "target_pad", "fixture")
        cfg.primary_object_name = "object"
        cfg.primary_target_name = "target_pad"
        cfg.success_term_name = "place_success"
        cfg.ee_body_name = "gripper_base"
        cfg.base_link_name = "base_link"
        cfg.gripper_joint_names = ("gripper_joint1", "gripper_joint2")
```

`object` 和 `target_pad` 是兼容别名；所有实体也会写入 `entities/<name>/...`。

### 2. 任务 profile

在任意可导入模块中注册：

```python
from collection.profiles import CollectionTaskProfile, register_collection_profile


register_collection_profile(
    CollectionTaskProfile(
        key="my_task",
        display_name="My task",
        task_ids=("Isaac-M20-Piper-MyTask-Collect-v0",),
        output_dir="my_task",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_my_top_camera,
            "wrist_camera": make_my_wrist_camera,
            "workcell_camera": make_my_workcell_camera,
        },
        tracked_entity_names=("object", "target_pad", "fixture"),
        primary_object_name="object",
        primary_target_name="target_pad",
        success_term_name="place_success",
    )
)
```

运行：

```powershell
python scripts\data\collect_demonstrations.py `
  --task Isaac-M20-Piper-MyTask-Collect-v0 `
  --profile my_task `
  --profile-module my_package.collection_profile
```

## HDF5 兼容性

新文件使用：

```text
schema = teleop_manipulation/v1
```

旧文件：

```text
schema = low_shelf_pick_place/v1
```

转换脚本同时接受两种 schema。旧 HDF5 的 `object`、`target_pad`、`robot` 和
`camera/<name>/rgb` 路径保持不变。

## 已实测

```text
low_shelf     -> profile=low_shelf, schema=teleop_manipulation/v1
two_cylinder  -> profile=two_cylinder, schema=teleop_manipulation/v1
```

两个任务均由同一个 `collect_demonstrations.py` 完成 headless HDF5 写入和正常关闭。

## Fast 数采路径

需要更低遥操延迟时，显式增加 `--fast-stepping`。它复用同一套 collection profile、
recorder 和异步 HDF5 writer，只替换 `env.step()`：

```text
process_action -> record_pre_step -> FastTeleopStepper.advance -> record_post_step
```

Fast 路径仍每个物理 tick 推进 action manager，低层策略保持训练语义的 50 Hz；
机械臂 target 默认每 10 ms 写一次，其他 scene 数据每 20 ms 写一次。控制周期结束会
推进 `episode_length_buf` 和 `common_step_counter`，因此 HDF5 的 `step_index` 与
`sim_time` 和图像/动作采样保持同一时间轴。

```powershell
python scripts\data\collect_demonstrations.py `
  --task Isaac-M20-Piper-TwoCylinder-Collect-v0 `
  --input keyboard --fast-stepping --fast-render-hz 15 `
  --record-hz 10 --output recordings\two_cylinder_place\fast.hdf5
```

Fast 模式会运行 termination manager。自动成功/失败触发后，先记录当前帧和
termination flags，再 reset 并切换下一个 episode；键盘 `K/L` 或 VR
success/failure 仍可手动覆盖。详细参数、profiling 和抖动 A/B 见
`docs/teleop_fast_path_zh.md`。

## 回放

回放同样与具体任务解耦：

```text
scripts/data/replay_hdf5.py
  └─ scripts/data/replay/hdf5_video.py
       ├─ 读取任意 teleop_manipulation/v1 HDF5
       ├─ 按 camera_names 元数据排序
       ├─ 支持 RGB、深度、单相机和多相机
       └─ 输出 MP4 / PNG
```

回放不加载 Isaac Sim，也不依赖 task id 或 collection profile。检查数据：

```powershell
python scripts\data\replay_hdf5.py `
  --input recordings\my_task\keyboard.hdf5 `
  --check-only
```

单相机：

```powershell
python scripts\data\replay_hdf5.py `
  --input recordings\my_task\keyboard.hdf5 `
  --cameras wrist_camera `
  --output recordings\replay\my_task_wrist.mp4
```

多相机：

```powershell
python scripts\data\replay_hdf5.py `
  --input recordings\my_task\keyboard.hdf5 `
  --cameras top_camera wrist_camera workcell_camera `
  --stride 2 --fps 10 `
  --output recordings\replay\my_task_views.mp4 `
  --frame recordings\replay\my_task_last.png
```

不同分辨率会按 `--tile-width` / `--tile-height` 统一后拼接。鱼眼相机缺少普通针孔
`intrinsics` 时仍可正常回放。
