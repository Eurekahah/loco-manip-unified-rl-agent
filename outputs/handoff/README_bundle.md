# AS2-W / Piper X / AS2W_PiperX URDF 资源包

整理日期：2026-11-27
内容：三个模型的 URDF + 全部网格（STL / DAE），并额外提供 Piper X 的 RealSense D405 相机变体。

---

## 1. 目录结构

```text
AS2W_PiperX_urdf_bundle/
├── README.md
├── as2w_description/                 # 宇树 AS2-W 四轮足机器人（官方 unitree_ros 包）
│   ├── as2w.urdf
│   └── meshes/                       # 18 个 STL
├── piper_x_description/              # 松灵 Piper X 六轴机械臂
│   ├── package.xml
│   ├── CMakeLists.txt
│   ├── config/  launch/  rviz/
│   ├── urdf/
│   │   ├── piper_x_description.urdf
│   │   ├── piper_x_description_d405.urdf
│   │   └── PIPER-XURDF.csv
│   └── meshes/
│       ├── *.STL                     # 7 个 STL
│       └── dae/*.dae                 # 16 个 DAE
├── as2w_piperx/                      # AS2-W + Piper X 组装整机（本仓库生成）
│   ├── README.md
│   └── urdf/
│       ├── AS2W_PiperX.urdf
│       └── meshes/                   # 27 个 STL
├── realsense2_description/           # D405 网格依赖
│   ├── package.xml  CMakeLists.txt
│   └── meshes/d405.stl
├── piper_description/                # 相机支架网格依赖
│   ├── package.xml  CMakeLists.txt
│   └── meshes/dae/realsense_mid_stand.dae
└── previews/                         # Isaac Sim RTX 效果图
```

`piper_x_description_d405.urdf` 是**可直接加载的纯 URDF**，不需要运行 xacro。

---

## 2. 模型参数

| 模型 | Links | Joints | 总质量 | 关节类型 |
|---|---:|---:|---:|---|
| AS2-W | 18 | 17 | 26.000 kg | 12 revolute + 4 continuous + 1 fixed |
| Piper X | 9 | 8 | 1.550 kg | 6 revolute + 2 prismatic |
| Piper X + D405 | 21 | 20 | 1.622 kg | 6 revolute + 2 prismatic + 12 fixed |
| AS2W_PiperX | 28 | 27 | 27.561 kg | 18 revolute + 4 continuous + 3 fixed + 2 prismatic |

- AS2-W：`base_link` + 16 个腿部连杆（FL/FR/RL/RR × hip/thigh/calf/foot）+ `JT128` 激光雷达。
- Piper X：`base_link` + `Link1`…`Link6` + `Link7`/`Link8` 平行夹爪两指。
- Piper X + D405：在 Piper X 的 `Link6` 上增加相机支架、D405 本体和标准相机坐标系。
- AS2W_PiperX：在 AS2-W 的 `base_link` 上通过固定关节 `base_to_arm`（xyz = 0, 0, 0.0798）安装 Piper X。

---

## 3. D405 相机变体

文件：`piper_x_description/urdf/piper_x_description_d405.urdf`

参数来源：Intel RealSense `realsense2_description` 的 D405 模型。

- 外壳尺寸：0.042 m（宽）× 0.042 m（高）× 0.023 m（深）
- 质量：0.072 kg
- 网格：`meshes/d405.stl`，原始单位为毫米，URDF 中使用 `scale="0.001 0.001 0.001"`
- 本体坐标系：`camera_link`
- 标准坐标系：`d405_depth_frame`、`d405_depth_optical_frame`、`d405_infra1_*`、`d405_infra2_*`、`d405_color_*`

安装位置沿用原相机支架的位姿：

```xml
<joint name="camera_stand_joint" type="fixed">
  <origin xyz="0.001 -0.036 0.022" rpy="1.57 3.14 0.0"/>
  <parent link="Link6"/>
  <child link="camera_stand_link"/>
</joint>

<joint name="d405_camera_joint" type="fixed">
  <origin xyz="0.029 -0.069 0.022" rpy="0.0 -1.22 1.57"/>
  <parent link="Link6"/>
  <child link="d405_camera_link"/>
</joint>
```

如果实际 D405 支架的螺纹孔位置与旧支架不同，只需修改 `d405_camera_joint` 的 `xyz/rpy`，相机本体尺寸和网格不需要改。

---

## 4. 来源

| 模型 | 来源 |
|---|---|
| as2w_description | https://github.com/unitreerobotics/unitree_ros/tree/master/robots/as2w_description |
| piper_x_description | 松灵官方描述包（agilexrobotics/piper_isaac_sim 同源） |
| D405 mesh / 参数 | Intel RealSense `realsense2_description` |
| as2w_piperx | 本仓库 `deep_robotics_model/AS2W_PiperX`，由 `scripts/tools/build_as2w_piperx_urdf.py` 生成 |

AS2-W 的 18 个 STL 与本仓库组装件中的同名网格做过 MD5 比对，完全一致。

---

## 5. 使用方式

### 5.1 ROS 2

建议把包复制到工作空间的 `src/`：

```bash
mkdir -p ws/src
cp -r as2w_description piper_x_description as2w_piperx \
      realsense2_description piper_description ws/src/
cd ws
colcon build
source install/setup.bash
ros2 launch piper_x_description display_urdf.launch.py
```

`display_urdf.launch.py` 已默认指向 D405 变体。`as2w.urdf` 与 `AS2W_PiperX.urdf` 没有 launch 文件，可直接用 `robot_state_publisher` 加载。

### 5.2 Isaac Sim / Isaac Lab

Isaac Sim 的 URDF importer 对网格路径的解析规则：

- `package://<pkg>/...` → 相对包目录的上一级解析
- 相对路径（`meshes/...`）→ 相对当前工作目录解析

```python
import os
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
import omni.kit.commands

BUNDLE = r"<解压目录>/AS2W_PiperX_urdf_bundle"

# Piper X + D405（package://）
os.chdir(BUNDLE)
urdf = os.path.join(BUNDLE, "piper_x_description", "urdf",
                    "piper_x_description_d405.urdf")

# 纯 Piper X
# urdf = os.path.join(BUNDLE, "piper_x_description", "urdf",
#                     "piper_x_description.urdf")

ok, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
cfg.merge_fixed_joints = False
cfg.convex_decomp = False
cfg.fix_base = False
ok, root = omni.kit.commands.execute(
    "URDFParseAndImportFile", urdf_path=urdf,
    import_config=cfg, get_articulation_root=True)
print(root)
```

AS2-W 与组装整机使用相对网格路径，导入前分别 `chdir` 到：

- `AS2W_PiperX_urdf_bundle/as2w_description`
- `AS2W_PiperX_urdf_bundle/as2w_piperx/urdf`

### 5.3 MuJoCo

MuJoCo 不直接读取 URDF，需要先转换。STL 与 DAE 均可被常用 URDF→MJCF 工具识别。

---

## 6. 本包内修复

原始 Piper X 描述包有两处会导致基座显示异常，这里已修正（源文件未改动）：

1. **文件名大小写**：URDF 中为 `meshes/dae/base_Link.dae`，实际文件为 `base_link.dae`。Windows 下不报错，Linux 下会丢网格。
2. **DAE 单位元数据错误**：`base_link.dae` 由 Rhino 导出，`<unit meter="0.001" name="Millimeters"/>`，但坐标数值实际是米。Isaac Sim 会再缩小 1000 倍，基座看起来像没有渲染。已改为 `<unit meter="1" name="Meters"/>`。

---

## 7. 校验

构建脚本逐个解析所有 URDF 的 `<mesh filename>`，并按大小写敏感方式检查文件存在：

| URDF | 引用数 | 找到 | 缺失 |
|---|---:|---:|---:|
| as2w_description/as2w.urdf | 36 | 36 | 0 |
| piper_x_description/urdf/piper_x_description.urdf | 18 | 18 | 0 |
| piper_x_description/urdf/piper_x_description_d405.urdf | 21 | 21 | 0 |
| as2w_piperx/urdf/AS2W_PiperX.urdf | 54 | 54 | 0 |

完整结果见包内 `_validate.txt`。

---

## 8. 效果图

`previews/` 下为 Isaac Sim 5.1 RTX 渲染：

- `as2w.png` — AS2-W 整机
- `piper_x.png` — Piper X 单臂（基座网格已修复）
- `piper_x_d405.png` — Piper X + D405 变体
- `as2w_piperx.png` — AS2W_PiperX 组装整机

---

## 9. STEP / STP

本包只包含上游公开发布的 URDF 与网格。AS2-W 和 Piper X 的上游描述包未提供 STEP/STP 文件，因此包内不包含 CAD 源文件。
