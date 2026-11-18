# 鱼眼传感器：遥操与数据采集

双路和四路鱼眼传感器通过共享模块接入：

```text
scripts/tools/fisheye_cli.py       命令行参数与校验
scripts/tools/fisheye_teleop.py    相机创建、全景拼接、omni.ui 窗口
scripts/tools/fisheye_equirect.py  鱼眼投影与等距柱面 remap
scripts/tools/fisheye_gpu.py       GPU grid_sample 全景投影与融合
scripts/tools/fisheye_profiler.py  轻量 profiling
```

完整架构、实测性能和 GUI 验证见
[`fisheye_runtime_optimization_zh.md`](fisheye_runtime_optimization_zh.md)。
任务与采集入口的隔离方式见
[`demonstration_collection_architecture_zh.md`](demonstration_collection_architecture_zh.md)。

默认 `--fisheye-rig none`，不创建额外相机，原有脚本行为不变。可选：

```text
none   不启用鱼眼
dual   前后两路鱼眼，Scene key: fisheye_front, fisheye_back
four   前左后右四路鱼眼，Scene key: fisheye_cam0..fisheye_cam3
```

所有相机都是 `Robot/base_link` 的子 prim，会随 M20 机器人刚性移动。默认安装位置为
`(-0.44, 0.0, 0.34) m`，即 `base_link` 尾部上方；单路理想等距鱼眼 FOV 为 200 度。

## 遥操脚本

以下三个脚本支持 `--fisheye-rig`、独立全景窗口
`--show-panorama / --no-show-panorama` 和腕部 UI 窗口
`--show-wrist-camera / --no-show-wrist-camera`：

```text
scripts/teleoperation/teleop_m20_piper_keyboard.py
scripts/teleoperation/teleop_m20_piper_vr.py
scripts/teleoperation/teleop_m20_loco_keyboard.py
```

双路全景：

```powershell
python scripts\teleoperation\teleop_m20_piper_keyboard.py `
  --task Isaac-M20-Piper-Teleop-History-v0 --num_envs 1 `
  --fisheye-rig dual --show-panorama --show-wrist-camera `
  --fisheye-width 480 --fisheye-height 300 `
  --panorama-width 960 --panorama-height 480 `
  --panorama-backend gpu --panorama-display-hz 12 --wrist-display-hz 20
```

四路全景：

```powershell
python scripts\teleoperation\teleop_m20_piper_vr.py `
  --task Isaac-M20-Piper-Teleop-History-v0 --num_envs 1 `
  --fisheye-rig four --show-panorama --show-wrist-camera `
  --fisheye-width 480 --fisheye-height 300 `
  --panorama-width 960 --panorama-height 480 `
  --panorama-backend gpu --panorama-display-hz 12 --wrist-display-hz 20
```

只采集/渲染鱼眼，不打开拼接窗口时，去掉 `--show-panorama` 即可。启用任一鱼眼模式后，
脚本会自动设置 `enable_cameras=True`。`--rendering_mode` 默认就是 `balanced`；仍可用
`--rendering_mode performance/quality` 覆盖。`--show-panorama` 和
`--show-wrist-camera` 只能用于 GUI；headless 模式会报错。

## 数据采集脚本

```text
scripts/data/collect_demonstrations.py
```

支持相同的 `--fisheye-rig`、分辨率、FOV、安装位置和更新时间参数。默认只记录原始腕部
RGB 和原始鱼眼 RGB；`--record-panorama` 可显式追加拼接全景，
`--record_depth` 可显式请求深度。状态、action 和所有相机按 `--record-hz` 统一采样。
HDF5 由有界队列和后台 writer 逐帧扩展，默认 `lzf` 压缩。

四路采集示例：

```powershell
python scripts\data\collect_demonstrations.py `
  --task Isaac-M20-Piper-LowShelf-Collect-v0 `
  --input keyboard --num_envs 1 `
  --no-scene_camera --no-wrist_camera --no-top_camera `
  --fisheye-rig four `
  --fisheye-width 480 --fisheye-height 300 `
  --record-hz 10 --compression lzf --async-write `
  --profile-json outputs\four_fisheye_profile.json `
  --output recordings\low_shelf_place\four_fisheye.hdf5
```

HDF5 中对应组为：

```text
episodes/<episode>/camera/fisheye_front/rgb
episodes/<episode>/camera/fisheye_back/rgb

episodes/<episode>/camera/fisheye_cam0/rgb
episodes/<episode>/camera/fisheye_cam1/rgb
episodes/<episode>/camera/fisheye_cam2/rgb
episodes/<episode>/camera/fisheye_cam3/rgb
```

根节点还会写入 `fisheye_rig`、`fisheye_projection`、`fisheye_fov_deg`、
`fisheye_mount_pos_base_link`、`fisheye_lens_offset_m` 和
`fisheye_camera_names`，因此 HDF5 可以脱离启动命令独立描述鱼眼配置。

## 常用参数

```text
--fisheye-rig {none,dual,four}   默认 none
--fisheye-width                  单路宽度，默认 480
--fisheye-height                 单路高度，默认 300
--fisheye-fov-deg                理想鱼眼全 FOV，默认 200
--fisheye-mount-pos X Y Z        相对 base_link 的安装点
--fisheye-lens-offset            双路前后镜头半间距，默认 0.02 m
--fisheye-update-period          传感器更新周期；0 自动跟随预览/采样频率
--show-panorama                  遥操脚本显示独立等距柱面窗口
--panorama-width                 全景宽度，默认 960
--panorama-height                全景高度，默认 480
--panorama-backend               auto / gpu / cpu，默认 auto
--show-wrist-camera              显示已有腕部 sensor 的 omni.ui 窗口
--wrist-display-hz               腕部 UI 刷新率，默认 20
--panorama-display-hz            全景 UI 刷新率，默认 12
--preview-hz                     GUI 预览参考频率，默认 15
--record-hz                      统一采样率，默认 10
--record-panorama                可选保存全景，默认关闭
--compression                    none / lzf / gzip1 / gzip4，默认 lzf
--async-write / --no-async-write 默认后台 writer
--writer-queue-size              默认 8
--profile-json                   输出轻量性能 JSON
--ui-snapshot-dir                保存主 viewport 和 UI 验证图
```

## 转 LeRobot

`scripts/data/convert_hdf5_to_lerobot.py` 会自动识别鱼眼组和可选
`fisheye_panorama`。只有鱼眼时从 `camera1`
开始编号；与原有 top/wrist/workcell 混采时，鱼眼从 `camera4` 开始编号。不同相机允许
使用不同分辨率。

```powershell
python scripts\data\convert_hdf5_to_lerobot.py `
  --input recordings\low_shelf_place\four_fisheye.hdf5 `
  --output data\lerobot\low_shelf_four_fisheye `
  --overwrite
```

当前 HDF5 默认保存原始鱼眼图像和相机位姿，可选保存全景。鱼眼组明确记录
`projection_model=fisheye_equidistant`、FOV、安装位置和镜头间距，不写普通针孔
`intrinsics`；不要把它当成普通针孔相机使用。
