# 鱼眼传感器：遥操与数据采集

双路和四路鱼眼传感器通过共享模块接入：

```text
scripts/tools/fisheye_cli.py       命令行参数与校验
scripts/tools/fisheye_teleop.py    相机创建、全景拼接、omni.ui 窗口
scripts/tools/fisheye_equirect.py  鱼眼投影与等距柱面 remap
```

默认 `--fisheye-rig none`，不创建额外相机，原有脚本行为不变。可选：

```text
none   不启用鱼眼
dual   前后两路鱼眼，Scene key: fisheye_front, fisheye_back
four   前左后右四路鱼眼，Scene key: fisheye_cam0..fisheye_cam3
```

所有相机都是 `Robot/base_link` 的子 prim，会随 M20 机器人刚性移动。默认安装位置为
`(-0.44, 0.0, 0.34) m`，即 `base_link` 尾部上方；单路理想等距鱼眼 FOV 为 200 度。

## 遥操脚本

以下三个脚本支持 `--fisheye-rig` 和独立全景窗口开关
`--show-panorama / --no-show-panorama`：

```text
scripts/teleoperation/teleop_m20_piper_keyboard.py
scripts/teleoperation/teleop_m20_piper_vr.py
scripts/teleoperation/teleop_m20_loco_keyboard.py
```

双路全景：

```powershell
python scripts\teleoperation\teleop_m20_piper_keyboard.py `
  --task Isaac-M20-Piper-Teleop-History-v0 --num_envs 1 `
  --fisheye-rig dual --show-panorama `
  --fisheye-width 480 --fisheye-height 300 `
  --panorama-width 960 --panorama-height 480
```

四路全景：

```powershell
python scripts\teleoperation\teleop_m20_piper_vr.py `
  --task Isaac-M20-Piper-Teleop-History-v0 --num_envs 1 `
  --fisheye-rig four --show-panorama `
  --fisheye-width 480 --fisheye-height 300 `
  --panorama-width 960 --panorama-height 480
```

只采集/渲染鱼眼，不打开拼接窗口时，去掉 `--show-panorama` 即可。启用任一鱼眼模式后，
脚本会自动设置 `enable_cameras=True`。`--show-panorama` 只能用于 GUI；headless 模式会报错。

## 数据采集脚本

```text
scripts/data/collect_low_shelf_hdf5.py
```

支持相同的 `--fisheye-rig`、分辨率、FOV、安装位置和更新时间参数，但数据采集侧只记录
原始鱼眼 RGB，不计算拼接全景。`--record_depth` 开启时，鱼眼相机也会请求 depth。

四路采集示例：

```powershell
python scripts\data\collect_low_shelf_hdf5.py `
  --task Isaac-M20-Piper-LowShelf-Collect-v0 `
  --input keyboard --num_envs 1 `
  --no-scene_camera --no-wrist_camera --no-top_camera `
  --fisheye-rig four `
  --fisheye-width 480 --fisheye-height 300 `
  --fisheye-update-period 0.1 `
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
--fisheye-update-period          传感器更新周期；0 表示每步更新
--show-panorama                  遥操脚本显示独立等距柱面窗口
--panorama-width                 全景宽度，默认 960
--panorama-height                全景高度，默认 480
```

## 转 LeRobot

`scripts/data/convert_hdf5_to_lerobot.py` 会自动识别鱼眼组。只有鱼眼时从 `camera1`
开始编号；与原有 top/wrist/workcell 混采时，鱼眼从 `camera4` 开始编号。不同相机允许
使用不同分辨率。

```powershell
python scripts\data\convert_hdf5_to_lerobot.py `
  --input recordings\low_shelf_place\four_fisheye.hdf5 `
  --output data\lerobot\low_shelf_four_fisheye `
  --overwrite
```

当前 HDF5 保存的是原始鱼眼图像和相机位姿。鱼眼投影参数应以采集时的命令行配置为准，
不要把它当成普通针孔相机内参使用。
