# M20-Piper 四路鱼眼：实机标定与理想水平环视

脚本：

```text
scripts/tools/render_m20_piper_four_fisheye_gui.py
```

拼接与投影：

```text
scripts/tools/fisheye_equirect.py
```

## 相机模型

用户给出的 `camera_model: omni`、`distortion_model: radtan` 按 Kalibr
全向相机模型解释：

```text
intrinsics: [fx, fy, cx, cy]
distortion_coeffs: [xi, k1, k2, p1, p2]
```

其中第一个畸变系数是 `xi`。如果把五个畸变系数全部当成普通
OpenCV pinhole-radtan 参数，单路有效半视场会错误地缩到约 15 度。

相机坐标系采用 OpenCV/ROS 约定：

```text
+X right, +Y down, +Z forward
```

`T_cn_cnm1` 按 Kalibr 约定解释为“把相机 n-1 坐标系变换到相机 n
坐标系”，所以生成相机 n 在 camera-0 中的位姿时使用其逆矩阵累积。

## 两种模式

`real_calibration`：

```text
四路各自使用实机标定的 fx/fy/cx/cy + xi/k1/k2/p1/p2
RTX 使用每路单独的 200 度 F-theta 拟合
全景拼接使用每路完整的 omni-radtan 模型
```

`ideal_horizontal_90`（当前默认）：

```text
完全忽略实机标定文件
四路相机共用同一个理想 200 度圆形等距鱼眼内参
四路光心重合，光轴在 base_link 水平面内依次相差 90 度
光轴方向依次为：前、左、后、右
全景拼接只使用理想等距投影和固定角度融合
```

该模式用于验证“完美相机”的拼接上限。因为四路光心完全重合，机器人
近景不会出现相机基线造成的视差；它不是实机四镜头机械结构的物理
仿真，而是理想虚拟传感器基线。

## 安装位置

四个相机都是 `Robot/base_link` 的子传感器 prim：

```text
/World/envs/env_0/Robot/base_link/M20FourCam0
/World/envs/env_0/Robot/base_link/M20FourCam1
/World/envs/env_0/Robot/base_link/M20FourCam2
/World/envs/env_0/Robot/base_link/M20FourCam3
```

默认 rig 原点相对 `base_link`：

```text
(-0.44, 0.0, 0.34) m
```

`X=-0.44 m` 位于机身尾部，`Z=0.34 m` 高于 base_link。相机作为
`base_link` 子 prim，会随机器人刚性移动，不需要在运行时创建
RigidBodyAPI 和 FixedJoint。

`ideal_horizontal_90` 中四路相机使用同一个安装点，因此没有机械
基线。这是为了得到与前后双鱼眼理想拼接一致的近景效果。

## 原生渲染与拼接

Isaac Sim 5.1 没有直接对应 Kalibr `omni-radtan` 的相机 schema。
实机标定模式使用 200 度 F-theta 模型做 RTX 原生渲染近似，拟合误差为：

```text
cam0: 0.039 deg RMS
cam1: 0.049 deg RMS
cam2: 0.022 deg RMS
cam3: 0.075 deg RMS
```

等距柱面拼接仍使用完整的 `xi + k1,k2,p1,p2` 投影，不依赖特征点。
每个全景像素先变成球面方向，再投影到四路相机，并在相邻镜头重叠区
按光轴夹角做高斯权重融合。

四个镜头水平环视，但竖直视场约 90 度。按当前标定，1280x640
等距柱面全景的真实有效覆盖率约 `56.1%`，天空顶部和机身下方会出现
黑色未覆盖区域。这是标定视场决定的，不能通过隐藏 mask 变成真实
360 度球面覆盖。

`ideal_horizontal_90` 使用重合光心和四个严格相差 90 度的理想光轴，
单路使用圆形 200 度鱼眼，几何覆盖率为 `100%`，不会产生相机基线
造成的近景视差重影。

## GUI 运行

实机标定：

```powershell
conda activate env_isaac_lab
python -u scripts\tools\render_m20_piper_four_fisheye_gui.py `
  --rig_mode real_calibration `
  --width 960 --height 600 `
  --equirect_width 1280 --equirect_height 640 `
  --measure_frames 10 --warmup_frames 2 `
  --vx 0.20 --wz 0.10
```

理想水平 90 度环视（当前默认）：

```powershell
python -u scripts\tools\render_m20_piper_four_fisheye_gui.py `
  --rig_mode ideal_horizontal_90 `
  --ideal_fov_deg 200 `
  --width 960 --height 600 `
  --equirect_width 1280 --equirect_height 640 `
  --measure_frames 10 --warmup_frames 2 `
  --vx 0.20 --wz 0.10
```

GUI 会打开四个原生鱼眼视口和一个 `omni.ui` 等距柱面全景窗口。计时
结束后继续实时运行，直到关闭 Isaac Sim。

## 跟随验证

```powershell
python -u scripts\tools\render_m20_piper_four_fisheye_gui.py `
  --headless --enable_cameras `
  --width 320 --height 200 `
  --equirect_width 320 --equirect_height 160 `
  --measure_frames 1 --warmup_frames 0 `
  --validate_follow --exit_after_measure
```

测试会同时打印机器人根节点和相机世界坐标的位移，用于确认相机随
`base_link` 移动。

## 实测耗时

RTX A4000、Isaac Sim 5.1、960x600 四路输入、1280x640 全景、
实机标定模式 10 个有效帧：

```text
motion_step_ms   mean 208.687 ms  p95 229.030 ms
readback_four_ms mean  26.771 ms  p95  28.718 ms
four_stitch_ms   mean  74.802 ms  p95  78.301 ms
loop_ms          mean 310.264 ms  p95 332.819 ms
```

理想水平 90 度环视模式 10 个有效帧：

```text
motion_step_ms   mean 223.733 ms  p95 275.451 ms
readback_four_ms mean  28.318 ms  p95  31.299 ms
four_stitch_ms   mean  72.535 ms  p95  82.776 ms
loop_ms          mean 324.588 ms  p95 372.056 ms
```

其中 `motion_step_ms` 包含低层策略推理、物理步进和四路 RTX 相机
渲染。`loop_ms` 还包含四路 RGB 回读、omni-radtan remap、融合和
全景 UI 更新。

同一配置在 GUI 模式下实测 5 帧：

```text
motion_step_ms   mean 316.332 ms  p95 377.920 ms
readback_four_ms mean  26.174 ms  p95  28.049 ms
four_stitch_ms   mean  74.799 ms  p95  77.121 ms
panorama_ui_ms   mean  78.737 ms  p95  81.073 ms
loop_ms          mean 500.497 ms  p95 564.107 ms
```

GUI 比无头测量多出 `ByteImageProvider` 上传全景图的约 79 ms，所以
当前 960x600 四路加 1280x640 全景的实时整环约为 2 FPS。需要更高
帧率时，优先把四路渲染分辨率降到 640x400，或降低全景分辨率。

## 输出

```text
outputs/four_fisheye_m20_piper_960x600/
  real_calibration/
    cam0_rgb.png
    cam1_rgb.png
    cam2_rgb.png
    cam3_rgb.png
    four_view_grid.png
    equirect_rgb.png
    equirect_valid_mask.png
    timings.json
  ideal_horizontal_90/
    cam0_rgb.png
    cam1_rgb.png
    cam2_rgb.png
    cam3_rgb.png
    four_view_grid.png
    equirect_rgb.png
    equirect_valid_mask.png
    timings.json
```

## 实现注意

不要在已运行的 articulation stage 上给相机 prim 动态调用
`OmniLensDistortion*API.ApplyAPI()`；本仓库环境中该调用会阻塞。
当前脚本使用 Isaac Lab 已稳定的 `CameraCfg + FisheyeCameraCfg`
路径。

四路 `T_cn_cnm1` 也不能直接顺序相乘。Kalibr 约定下必须使用逆变换
累积，否则相机姿态误差会沿链传递，后几路图像的地平线会出现明显
滚转。
