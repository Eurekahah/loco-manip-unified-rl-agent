# WBC-OmniBench 环境主题任务实现记录

版本：v0.2  
日期：2026-11-18  
验证：Isaac Sim 5.1，headless quality 渲染

## 1. 本轮范围

本轮实现环境主题设计中的 R1、H1、K1、H2 四个任务，目标是形成可遥操、可采集、可复现的
场景工程基线，不扩展新的训练 MDP：

| 任务 | Gym task | 主题场景 | 主要能力 |
|---|---|---|---|
| R1 FrontDeskPickup | `Isaac-M20-Piper-FrontDeskPickup-*-v0` | Isaac `Office` | 全景搜索、接待台高度取物、访客侧放置 |
| H1 WardDelivery | `Isaac-M20-Piper-WardDelivery-*-v0` | Isaac `Hospital` | 全景搜索、窄道、高位药架、床头柜放置 |
| K1 KitchenDrawerServe | `Isaac-M20-Piper-KitchenDrawerServe-*-v0` | 独立 `KitchenRoom` | 开抽屉、低姿取物、上层柜门约束、台面放置 |
| H2 SupplyRestock | `Isaac-M20-Piper-SupplyRestock-*-v0` | Isaac `Hospital` | 高位耗材取件、绕障、地面低位补货 |

每个任务均注册 `Teacher-v0`、`Teleop-v0` 和 `Collect-v0`。Teacher 只提供状态观测；
Teleop 使用现有 13 维高层接口；Collect 打开前视、腕部、workcell、顶部相机和 HDF5
recorder。所有任务继续使用现有机器人、动作空间、奖励和终止结构。

## 2. 场景实现

### 2.1 R1 FrontDeskPickup

- 加载 Isaac 官方 `Environments/Office/office.usd`，没有加载 `Simple_Room`。
- 在接待厅工作区加入带碰撞的接待台体、木纹台面、隐私挡板和访客侧矮桌。
- 目标位于机器人初始前视方向之外，起始需要尾部高位全景相机发现。
- 台面放置糖盒目标及 cracker box、tomato soup can 两个干扰物。
- 访客侧使用官方咖啡桌，目标托盘位于桌面上方；等候椅和沙发限定接近路径。
- 在前视、腕部和全景之外，额外提供 workcell 与顶部相机。

验证状态：quality 渲染通过。静置 40 个环境步后，目标糖盒仍位于接待台面
`z≈1.069 m`，未发生掉落或穿透。

### 2.2 H1 WardDelivery

- 从 Isaac 官方 `Environments/Hospital/hospital.usd` 裁剪当前病房单元，生成
  `source/rl_training/assets/hospital_ward.usda`，不再把整栋多翼医院放进每个环境。
- 裁剪区域为原始 `x=20.60..27.10, y=-4.60..3.10, z=-0.10..3.60`，并平移
  `(-23.81, 0.80, 0.0) m` 居中；只保留完全落在该区域内的顶层 prim。
- 复用该单元自带的医疗柜和床头柜纹理/结构，并加入任务药架、病床、轮椅和目标托盘。
- 任务目标改用医院官方 `SM_PillBottleSet_01b.usd` 刚体包装，干扰物使用官方
  `SM_BottleA.usd`，不再使用 YCB 糖盒/芥末瓶冒充医疗物品。
- 药品包装位于大约 `1.32 m` 高的药架上；目标容器位于病床侧约 `0.90 m` 高的柜台。
- 轮椅和病床形成局部窄道，任务需要底盘重新站位以及机械臂与机身高度调整。
- 目标在初始前视/腕部视图中不可见，全景相机保持其全局可见性。

验证状态：quality 渲染通过。静置 40 个环境步后，医院药品包装稳定在药架
`z≈1.372 m`，目标托盘稳定在床头柜 `z≈0.912 m`。

### 2.3 K1 KitchenDrawerServe

- 不复用 `Simple_Room`。`KitchenRoom` 由独立墙、地板、天花板、踢脚环境和灯光配置
  构成，并使用木纹、墙体漆面、不锈钢材质。
- 底层使用 Isaac 官方 `Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd`。
- 上层另放一个 Sektion 柜体，柜门初始打开约 `0.72 rad`，压缩低位抽屉上方的
  机械臂抬升空间。
- 目标抽屉初始打开 `0.18 m`，罐体代理位于抽屉内；台面放糖盒干扰物。
- 台面高度为 `0.85 m`，托盘位于台面右侧。
- 当前使用已安装 YCB 集合中的 `005_tomato_soup_can.usd` 作为抽屉内目标代理，
  因为本机 YCB 资产包没有设计文档中的 mug。

验证状态：quality 渲染通过。静置 120 个环境步后，罐体仍位于目标抽屉内，
位置约为 `z≈0.709 m`，线速度和角速度接近零。

### 2.4 H2 SupplyRestock

- 复用与 H1 相同的 `HospitalWard` 裁剪场景。
- 新增 `SM_SupplyCabinet_01c.usd` 耗材柜，并在约 `1.40 m` 高度设置带碰撞的药架。
- 目标使用医院官方 `SM_PillBottleSet_01b.usd` 刚体包装，干扰物使用官方
  `SM_BottleA.usd`；两者位于同一任务层，要求选择正确目标。
- 目的地改为病床侧地面托盘，距离地面约 `1.1 cm`，形成约 `1.4 m → 0.01 m` 的
  高到低连续放置跨度。
- 通过新增双层耗材推车、病床和轮椅形成连续绕障；任务允许一次重抓取和恢复。
- 遥操时长首版为 `180 s`，与设计中的 `150 s` 成功判定预算分开，方便采集完整恢复过程。

验证状态：quality 渲染通过。医院药品包装静置后稳定在任务药架 `z≈1.452 m`；
出生点修正后机器人根高度约为 `z≈0.537 m`，未再落在 Hospital 自带床头柜或新增推车上。

Hospital USD 实测 1507 个顶层 prim，且原场景没有动态 `RigidBodyAPI`。裁剪后病房
只保留 91 个顶层引用，文件约 69 KB；Teleop quality 场景的 simulation setup
从整栋医院约 `60–70 s` 降到约 `34 s`。原医院小物件本身带碰撞网格，但默认是静态
场景几何，因此任务物体通过薄刚体包装复用原始医院 USD 网格，并显式使用凸包碰撞近似。

### 2.5 医院场景裁剪与官方任务物体

检查 `Hospital/hospital.usd` 得到以下事实：

- 1507 个顶层 prim，主要是墙体、楼层、楼梯和重复 Props 引用；
- 原医院场景没有 `RigidBodyAPI` 或 `ArticulationRootAPI`；
- 医院 Props 大量使用 `./Props/*.usd` 外部引用，可按顶层 prim 的世界包围盒裁剪；
- 小物件自身带 `PhysicsCollisionAPI`，但作为背景引用时仍保持静态。

因此当前实现：

1. 用 `scripts/tools/crop_isaac_hospital_room.py` 选择完全落在病房屋界内的顶层 prim；
2. 把 `./Props` 引用重映射到稳定的 Isaac Nucleus URL；
3. 把整个裁剪场景平移到机器人原点，输出 `source/rl_training/assets/hospital_ward.usda`；
4. 对需要的官方医院小物件建立带 `RigidBodyAPI` 的薄包装，保留原 USD 网格与碰撞，
   并设置 `convexHull` 动态碰撞近似。

当前 H1/H2 使用官方医院 `SM_PillBottleSet_01b.usd` 作为目标，
`SM_BottleA.usd` 作为干扰物。可用的小型医院道具还包括：

| 资产 | 约尺寸（长 × 宽 × 高） | 适用性 |
|---|---|---|
| `SM_PillBottle_01t.usd` | 2.9 × 2.9 × 5.1 cm | 精细抓取/小目标压力测试，偏小 |
| `SM_PillBottleSet_01b.usd` | 10.6 × 17.0 × 10.3 cm | 药品包/药瓶组合，当前目标 |
| `SM_Container_01a.usd` | 10.8 × 10.8 × 19.9 cm | 圆柱容器，适合替代 mug/can |
| `SM_BottleA.usd` | 6.2 × 6.2 × 25.0 cm | 医院瓶体，当前干扰物 |
| `SM_FirstAidKit_01a.usd` | 36.4 × 11.9 × 33.6 cm | 较大急救包，适合双臂或更宽夹爪 |
| `SM_UtilitiesTray_01e.usd` | 53.2 × 30.6 × 8.6 cm | 扁平托盘，适合非抓取式推拉 |

重新生成当前病房：

```powershell
conda run --no-capture-output -n env_isaac_lab python `
  scripts/tools/crop_isaac_hospital_room.py `
  --headless `
  --input <Isaac-Assets>/Isaac/Environments/Hospital/hospital.usd `
  --output source/rl_training/assets/hospital_ward.usda `
  --bounds 20.60 27.10 -4.60 3.10 -0.10 3.60 `
  --translate -23.81 0.80 0.0 `
  --selection contained `
  --overwrite
```

该工具按完整顶层 prim 选择，不做几何布尔切割。扩大裁剪范围时，应检查边界墙、门框、
地板和天花板是否仍完整；需要精确裁剪跨边界网格时，应在 DCC/USD API 中另行处理。

## 3. 相机与采集约定

- 前视：继续复用 `base_link` 前方 `(0.58, 0.0, 0.12)` 的 RGB-D 安装。
- 腕部：继续复用 Piper `gripper_base` 安装。
- 键盘/VR Teleop 默认 `--camera-view free`：主视口保持原来的自由相机位置；全景、腕部、
  前视在底部自动排成一行。需要主视口跟随前视时再显式使用 `--camera-view chest/front/both`。
- 全景：四个主题任务 profile 均将四路鱼眼 rig 提高到 `base_link=(-0.44, 0.0, 0.58) m`，
  避免低安装位被病床、轮椅或接待台遮挡。
- 全景默认记录原始鱼眼 RGB；需要拼接图时加 `--record-panorama`。
- HDF5 统一记录机器人状态、动作、相机帧、任务目标、托盘和主题干扰物；医院任务目标
  当前使用官方医院药品包装，而不是外来 YCB 物体。

已注册采集 profile：

- `front_desk_pickup`
- `ward_delivery`
- `kitchen_drawer_serve`
- `supply_restock`

四个主题 profile 默认启用前视和四路鱼眼；显式传参仍可覆盖：

```powershell
--no-front_camera
--fisheye-rig none
--record-panorama
```

桌面 Collect 默认让主视口绑定机器人前视相机并跟随机器人，同时打开腕部预览；
可用 `--no-follow-robot` 和 `--no-show-wrist-camera` 关闭。

默认四路原始鱼眼会写入 `camera/fisheye_cam0..3`；需要拼接图时再加
`--record-panorama`，拼接图写入
`camera/fisheye_panorama`。已用 R1 完成 25 步 smoke：记录 5 个样本，
俯视、腕部、workcell、4 路鱼眼和等距柱面全景均写入 HDF5，鱼眼安装位置在元数据中为
`base_link=(-0.44, 0.0, 0.58) m`。

首轮 quality 渲染：

- 接待厅四视角：
  ![FrontDeskPickup 四视角](../recordings/scene_views/front_desk_pickup/front_desk_pickup_montage.png)
- 接待厅全景：
  ![FrontDeskPickup 全景](../recordings/scene_views/front_desk_pickup/front_desk_pickup_panorama.png)
- 病房四视角：
  ![WardDelivery 四视角](../recordings/scene_views/ward_delivery/ward_delivery_montage.png)
- 病房全景：
  ![WardDelivery 全景](../recordings/scene_views/ward_delivery/ward_delivery_panorama.png)
- 厨房四视角：
  ![KitchenDrawerServe 四视角](../recordings/scene_views/kitchen_drawer_serve/kitchen_drawer_serve_montage.png)
- 厨房全景：
  ![KitchenDrawerServe 全景](../recordings/scene_views/kitchen_drawer_serve/kitchen_drawer_serve_panorama.png)
- 补货四视角：
  ![SupplyRestock 四视角](../recordings/scene_views/supply_restock/supply_restock_montage.png)
- 补货全景：
  ![SupplyRestock 全景](../recordings/scene_views/supply_restock/supply_restock_panorama.png)

## 4. 开源实现与资产检索结论

本轮按“先检索、能复用就复用”的原则检查了任务设计和参考文献对应的公开实现：

| 来源 | 检查结果 | 本轮处理 |
|---|---|---|
| [soom1017/isaaclab_door_open](https://github.com/soom1017/isaaclab_door_open) | Isaac Lab 开门任务模板；机器人和门 USD 依赖仓库外私有路径；未提供 LICENSE | 只参考任务组织，不复制代码/资产 |
| [PatrizioAcquadro/AlexDoor-XAS](https://github.com/PatrizioAcquadro/AlexDoor-XAS) | 固定底座人形开门研究；README 明确为 proprietary；门资产含 local-only/private 条目 | 不复制 |
| [javokhirajabov/quadwbg](https://github.com/javokhirajabov/quadwbg) | GitHub 仓库未提供可用实现细节和 LICENSE | 不复制 |
| [MasterHow/OneOcc](https://github.com/MasterHow/OneOcc) | 全景占用论文项目；截至检查时训练/推理代码和 checkpoint 尚未发布 | 仅参考全景任务动机 |
| [haosulab/ManiSkill](https://github.com/haosulab/ManiSkill) | Apache-2.0，SAPIEN 任务结构和关节物体设计成熟 | 参考关节物体任务语义；本任务继续复用 Isaac 官方 Sektion 资产 |
| Isaac 官方资产 | Office、Hospital、YCB、Sektion 资产与当前 Isaac Sim 5.1 兼容 | 直接复用 |

因此本轮没有把外部仓库代码直接拷入工程。可以直接复用的部分是 Isaac 官方场景、
家具、关节柜体和 YCB 资产；任务注册、相机工厂和 HDF5 profile 继续沿用本仓库已有
实现。

## 5. 足臂/轮臂协作定位

当前平台不提供单腿末端笛卡尔控制，因此本轮不把“腿作为第五只手”作为成功必要条件。
已实现任务中的全身协作由现有动作空间承担：

- H1：高位药架到低位床头柜、病床/轮椅窄道、机身高度与底盘的联合调整。
- H2：约 `1.4 m` 耗材架到地面托盘，经病床、轮椅和双层推车的连续绕障。
- K1：低位抽屉与打开的上层柜门形成垂直约束，需要调整机身高度/pitch 和底盘站位。
- R1：标准接待台高度到访客侧低桌，需要底盘重新站位、机身高度调整和连续抓取。

对于“机械臂单独拉门会拖动底盘”的强因果门任务，建议在获得可再分发的铰链门 USD
后单独实现，并使用门轴阻尼、轮制动和地面摩擦随机化。当前开源检索到的门资产许可和
运行框架均不适合直接并入本仓库。

## 6. 复现命令

接待厅：

```powershell
conda run --no-capture-output -n env_isaac_lab python -u `
  scripts/data/capture_task_scene_views.py `
  --headless `
  --task Isaac-M20-Piper-FrontDeskPickup-Teleop-v0 `
  --output_dir recordings/scene_views/front_desk_pickup
```

医院：

```powershell
conda run --no-capture-output -n env_isaac_lab python -u `
  scripts/data/capture_task_scene_views.py `
  --headless `
  --task Isaac-M20-Piper-WardDelivery-Teleop-v0 `
  --output_dir recordings/scene_views/ward_delivery
```

厨房：

```powershell
conda run --no-capture-output -n env_isaac_lab python -u `
  scripts/data/capture_task_scene_views.py `
  --headless `
  --task Isaac-M20-Piper-KitchenDrawerServe-Teleop-v0 `
  --output_dir recordings/scene_views/kitchen_drawer_serve
```

医疗补货：

```powershell
conda run --no-capture-output -n env_isaac_lab python -u `
  scripts/data/capture_task_scene_views.py `
  --headless `
  --task Isaac-M20-Piper-SupplyRestock-Teleop-v0 `
  --output_dir recordings/scene_views/supply_restock
```

## 7. 当前限制

- 本轮是遥操采集场景基线，不新增抽屉复位、门开角、窄道净空或干扰物未碰落等完整 MDP
  成功判定。
- K1 的上层柜门和抽屉关节已在场景中可动，但 HDF5 当前没有增加关节物体专用时间序列；
  相机数据和任务物体状态可直接采集。
- H1/H2 当前使用由官方 Hospital 派生并重新居中到原点的 `hospital_ward.usda`。
  该资产仍引用官方 Hospital Props URL，需要 Nucleus/云资产可用；如离线部署，应将
  所需 Props 与纹理缓存到本地 Nucleus，并保持相同的引用前缀。
- 当前 `KitchenRoom` 是任务配置层面的独立场景组合，不是单独发布并引用的 USD 文件；
  后续如需要跨任务共享，可将其烘焙为独立 USD 资产而不改任务注册接口。
- 独立强因果开门压力测试继续暂缓，等待许可清晰、可再分发的铰链门 USD；当前不会用
  不可分发资产或低阻尼柜门冒充该任务。
