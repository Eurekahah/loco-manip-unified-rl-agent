# M20-Piper 鱼眼遥操与多相机采集优化

本文记录 Isaac Sim 5.1 / Isaac Lab 5.1 下的 VR 数据采集架构、实测数据和推荐命令。
工作环境为 Windows、RTX A4000 16 GB。

## 架构

```text
主线程
  ├─ VR / 键盘控制
  ├─ env.step
  ├─ 按 record_hz 统一采样 state/action/wrist/fisheye
  ├─ 按 wrist_display_hz 更新 omni.ui 腕部窗口
  └─ 按 panorama_display_hz 执行 GPU 拼接并更新全景窗口

有界队列
  └─ 后台 HDF5 writer 线程
       ├─ 第一帧创建可扩展数据集
       ├─ 后续逐帧 resize + append
       └─ reset/exit 时 finalize 并 flush
```

主 viewport 默认保持自由视角，并保留原来的自由相机位置。全景、腕部、前视三个预览窗口
直接读取已有传感器，不额外创建 camera render product；三者默认沿窗口底部自动排成一行，
并计入窗口标题/边框，因此彼此不重叠。`--camera-view chest/front/wrist/both` 只改变
主视口绑定的相机，不会再额外创建独立腕部 viewport。

## 新增参数

```text
--record-hz                         统一状态/action/相机采样率，默认 10 Hz
--preview-hz                        GUI 预览参考频率，默认 15 Hz
--wrist-display-hz                  腕部窗口刷新率，默认 20 Hz
--panorama-display-hz               全景窗口刷新率，默认 12 Hz
--front-display-hz                  前视窗口刷新率，默认 20 Hz
--show-wrist-camera                 遥操默认开启腕部 omni.ui 窗口
--show-panorama                     遥操默认开启全景 omni.ui 窗口
--show-front-camera                 遥操默认开启前视 omni.ui 窗口
--panorama-backend {auto,gpu,cpu}   默认 gpu，CUDA 不可用时自动回退 CPU
--record-panorama                   可选保存拼接全景，默认关闭
--compression {none,lzf,gzip1,gzip4}
--async-write / --no-async-write
--writer-queue-size                 默认 8 帧
--profile-json                      profiling JSON 输出路径
--profile-interval                  JSON 快照间隔，默认 5 秒
--ui-snapshot-dir                   保存主 viewport、腕部/前视/全景 UI 验证图
--wrist-window-width/height/pos     腕部 UI 尺寸和位置，默认 240x240 + 自动位置
--front-window-width/height/pos     前视 UI 尺寸和位置，默认 320x240 + 自动位置
--panorama-window-width/height/pos  全景 UI 尺寸和位置，默认 560x280 + 自动位置
```

键盘和 VR 遥操的默认鱼眼模式是 `four`；`--fisheye-rig two`（等价 `dual`）切换为前后
双路。`--fisheye-rig none` 会关闭鱼眼、全景和全景录制。数据采集仍由任务 profile 决定
默认鱼眼配置；headless 模式会自动关闭所有 UI 预览。

## GPU 全景

`scripts/tools/fisheye_gpu.py` 在启动时把现有 OpenCV remap 表转换为归一化
`grid_sample` grid：

```text
grid   [N, H_pano, W_pano, 2]
weight [N, 1, H_pano, W_pano]
```

四路 RGB 保持 CUDA tensor，组成 `[N, H, W, 3]` batch，通过
`torch.nn.functional.grid_sample(..., align_corners=True, padding_mode="zeros")`
采样，再按权重求和、裁剪到 `[0, 255]` 并转成 `uint8`。只有最终全景预览或可选保存帧
执行一次 CPU 回读。`--panorama-backend cpu` 可切回原有 `cv2.remap` 路径。

合成图 CPU/GPU 对照：

```text
dual  平均绝对像素差 0.48，最大 1
four  平均绝对像素差 0.23，99.9% 像素差 <= 12
```

## HDF5

writer 使用 `h5py.File(..., libver="latest")`，每个 episode 的第一帧决定数据集形状，
之后使用 `maxshape=(None, ...)` 逐帧扩展。默认图像压缩为 `lzf`，不再使用 `gzip=4`。
队列满时丢弃当前采样帧并计数，不阻塞 VR 主循环；finalize 控制命令不受帧队列背压影响。

新 HDF5 使用通用 `teleop_manipulation/v1`；转换脚本仍兼容旧
`low_shelf_pick_place/v1`。episode 中包含：

```text
episodes/<episode>/sim_time
episodes/<episode>/camera/<name>/sensor_frame
episodes/<episode>/camera/<name>/T_world_camera_world
episodes/<episode>/camera/<name>/T_world_camera_ros
episodes/<episode>/camera/<name>/projection_model
episodes/<episode>/camera/<name>/intrinsics_applicable
```

鱼眼组不再写普通针孔 `intrinsics`，而是记录
`fisheye_equidistant`、FOV、安装位置、镜头间距和投影模型。统一 `record_hz` 下，
state/action/所有相机帧长度一致。

## TiledCamera 评估

`scripts/tools/probe_tiled_fisheye_camera.py` 已在 Isaac Lab 5.1 实测通过：

```text
TiledCameraCfg + FisheyeCameraCfg -> RGB (1, 200, 320, 3), torch.uint8
```

因此 tiled render product 与鱼眼 spawn 兼容。但当前双路/四路 rig 的每个相机有不同安装
位姿；单个 `TiledCameraCfg` 只能 spawn 一个 offset。若要用一个 tiled product 合并四路，
需要额外手工创建四个不同 transform 的 camera prim，再让 tiled sensor 用 regex 绑定，
这会让当前双路、四路和单相机配置明显复杂化。本阶段保留独立 `CameraCfg`，已用上面的
探测脚本证明兼容性，后续可在此基础上做 tiled rig。

## 实测数据

测试均使用 A4000、Isaac Sim 5.1、`rendering_mode=balanced`、`record_hz=10`、
异步 writer、LZF。启动时间不计入 loop。

### 四路 `480x300 -> 960x480`

```text
env.step      203.06 ms
loop          205.33 ms
实际 FPS       5.12
hdf5 enqueue   0.19 ms
hdf5 write    40.07 ms（后台线程）
queue          0
dropped        0
```

### 四路 `320x200 -> 640x320`

```text
env.step      210.15 ms
loop          212.85 ms
实际 FPS       5.01
hdf5 enqueue   0.22 ms
hdf5 write    48.44 ms（后台线程）
queue          0
dropped        0
```

### GUI 三窗口（四路 `320x200` + 腕部 224x224）

```text
loop                225.75 ms
实际 FPS              4.78
camera capture        0.88 ms
GPU stitch            0.40 ms
GPU -> CPU            0.57 ms
panorama UI           4.45 ms
wrist UI              2.12 ms
hdf5 write           33.73 ms（后台线程）
```

腕部窗口和全景窗口各更新 60 次，均可见；相机 `sensor_frame` 为连续递增，没有重复采样。

优化前基线为四路 `480x300`、`performance`、CPU 全景：loop 约 277 ms、约 3.6 FPS。
当前 balanced 配置的 loop 约 205 ms、约 5.1 FPS，同时画质设置更高。

### 文件大小

```text
4 帧 四路 480x300，无全景      2.33 MB
4 帧 四路 320x200 + 全景        2.54 MB
60 帧 四路 320x200 + 腕部       20.49 MB，约 3.47 MB/s
```

## GUI 验证

验证产物位于：

```text
outputs/fisheye_smoke/gui_snapshots/main_viewport.png
outputs/fisheye_smoke/gui_snapshots/wrist_ui.png
outputs/fisheye_smoke/gui_snapshots/panorama_ui.png
outputs/fisheye_smoke/gui_snapshots/ui_status.json
```

`ui_status.json` 显示两个 UI 窗口均 `visible=true`，各更新 60 次，全景 backend 为
`gpu`。主 viewport 仍为自由视角；全景中心方向与机器人前向一致。

## 推荐命令

四路鱼眼 + 腕部 + 全景，VR 采集：

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts\data\collect_demonstrations.py `
  --task Isaac-M20-Piper-LowShelf-Collect-v0 `
  --input vr --num_envs 1 `
  --wrist_camera --no-top_camera --no-scene_camera `
  --fisheye-rig four `
  --fisheye-width 320 --fisheye-height 200 `
  --panorama-width 640 --panorama-height 320 `
  --record-hz 10 --preview-hz 15 `
  --wrist-display-hz 20 --panorama-display-hz 12 `
  --show-wrist-camera --show-panorama --panorama-backend gpu `
  --compression lzf --async-write `
  --profile-json outputs\collect_four_profile.json `
  --output recordings\low_shelf_place\four_fisheye_vr.hdf5
```

需要保存拼接全景时增加 `--record-panorama`。需要深度时显式增加 `--record_depth`。

headless smoke：

```powershell
python scripts\data\collect_demonstrations.py `
  --task Isaac-M20-Piper-LowShelf-Collect-v0 `
  --input keyboard --headless --max_steps 20 `
  --no-wrist_camera --no-top_camera --no-scene_camera `
  --fisheye-rig four `
  --fisheye-width 320 --fisheye-height 200 `
  --record-hz 10 --compression lzf `
  --output recordings\smoke\four_fisheye.hdf5 --overwrite
```

## 当前限制

- 四路相机仍对应四个独立 render product，尚未合并成单个 tiled product。
- 统一 `record_hz` 下所有相机帧长度一致；尚未实现每路相机独立时间戳和异步频率。
- 全景 UI 最终仍需要一次 GPU 到 CPU 回读，CPU 上传到 `omni.ui` 使用
  `set_data_array`，不再使用 `.tolist()`。
- 本机没有 `lerobot`，因此 HDF5 到 LeRobot 的最终写出未执行；转换脚本已更新以识别
  `fisheye_panorama`，普通三路和鱼眼默认映射保持兼容。
- Isaac Sim 启动时会报告包目录下 `user.config.json`、material cache 和 OptiX cache
  无写权限；这不影响本次运行，但会增加启动时间。
