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
