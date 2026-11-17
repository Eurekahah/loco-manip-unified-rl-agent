# M20-Piper Insta360 双鱼眼全景探针

脚本：

```text
scripts/tools/render_m20_piper_fisheye_gui.py
```

默认使用项目里的稳定低层策略任务：

```text
Isaac-M20-Piper-Teleop-History-v0
```

该任务默认加载：

```text
logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/exported_deploy/policy.pt
```

它会在场景里创建：

1. `Robot/base_link/M20PanoFront`：朝前鱼眼；
2. `Robot/base_link/M20PanoBack`：朝后鱼眼；
3. 两个相机都是 Isaac Lab 官方 `CameraCfg + FisheyeCameraCfg`；
4. 两个相机是 `base_link` 的子 prim，会随机器人刚性移动；
5. GUI 中打开前后两个原生鱼眼视口，以及一个拼接后的全景窗口。

相机默认参数：

```text
rig 位置: (-0.44, 0.0, 0.34) m，相对 base_link
前后镜头偏移: +/- 0.02 m
单个鱼眼 FOV: 200 deg
拼接输出: 960x480 equirectangular
拼接物距: 4.0 m
接缝混合: 4 deg
```

默认 `pitch_deg=0`。前后两个 200° 鱼眼在水平放置时覆盖完整球面；如果同时向下俯，会损失天顶区域。

## 运行

```powershell
conda activate env_isaac_lab
python -u scripts\tools\render_m20_piper_fisheye_gui.py `
  --width 960 --height 960 `
  --equirect_width 960 --equirect_height 480 `
  --measure_frames 60 --warmup_frames 10 `
  --vx 0.25 --wz 0.15
```

GUI 模式计时结束后会继续实时运行。关闭 Isaac Sim 窗口即可退出。

验证相机跟随：

```powershell
python -u scripts\tools\render_m20_piper_fisheye_gui.py `
  --headless --enable_cameras `
  --width 320 --height 320 `
  --equirect_width 320 --equirect_height 160 `
  --measure_frames 1 --warmup_frames 0 `
  --validate_follow --exit_after_measure
```

跟随测试会把机器人根节点平移约 15 cm，并保存：

```text
outputs/fisheye_m20_piper/follow_before.png
outputs/fisheye_m20_piper/follow_after.png
```

同时打印机器人本体蓝色标记在图像中的像素位移。像素位移接近 0 说明相机图像随机器人移动。

## 拼接方法

`scripts/tools/fisheye_equirect.py` 里的双鱼眼拼接使用已知外参，而不是 `cv2.Stitcher` 特征匹配：

1. 为每个 equirectangular 像素生成球面方向；
2. 按 4 m 物距补偿前后镜头约 4 cm 的基线；
3. 分别投影到前后鱼眼；
4. 在左右两个 90° 接缝附近按角度权重混合；
5. 输出 RGB 全景和有效区域 mask。

当前 200° 前后鱼眼在 `pitch_deg=0` 时覆盖率为 100%。

## 计时口径

```text
dual_fisheye_native_viewports.motion_step_ms
    低层策略推理 + 物理步进 + 前后鱼眼 GUI 渲染的 env.step() 时间

dual_fisheye_native_viewports.readback_both_ms
    两路 RGB annotator 回读时间

dual_fisheye_plus_equirectangular_panorama.motion_step_ms
    低层策略推理 + 物理步进 + 鱼眼视口 + 全景 UI 的 env.step() 时间

dual_fisheye_plus_equirectangular_panorama.dual_stitch_ms
    双鱼眼 remap、权重混合的时间

panorama_ui_ms
    omni.ui.ByteImageProvider 上传全景图的时间

loop_ms
    渲染 + 双路回读 + 拼接 + UI 更新
```

## 输出

```text
outputs/fisheye_m20_piper/
  front_fisheye_rgb.png
  back_fisheye_rgb.png
  equirect_rgb.png
  equirect_valid_mask.png
  timings.json
```

## 关于固定关节

相机不是刚体，而是传感器 prim。Isaac Lab 中把相机 prim 作为 `base_link` 的子 prim，就是传感器层面的刚性固定，也是最稳妥的跟随方式。

不要在仿真启动后再给相机新建独立 `RigidBodyAPI` 和 `FixedJoint`。实测这样会让 PhysX articulation view 失效，随后访问 `robot.data` 会报 `Simulation view object is invalidated`。
