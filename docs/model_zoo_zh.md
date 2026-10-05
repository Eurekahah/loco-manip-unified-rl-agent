# 模型清单 / 最佳策略索引（可更新）

**用途**：一眼找到"现在该用哪个 checkpoint、怎么跑起来看、性能什么水平"。
换模型 / 出新指标时**只改这一个文件**（表里加一行 + 在 `docs/review/DEFECT_LOG_zh.md` 记一条）。

**用法前提**：在 cmd 里先激活 conda 环境（`conda activate env_isaac_lab`），
**下面所有命令都是 `python …` 开头**，在仓库根目录执行；跑 Isaac 前建议先
`set PYTHONIOENCODING=utf-8`。`--headless` 是不开窗口（出图/出表），去掉它才会弹 Isaac 窗口。

**指标口径（不可混比）**

* 平地：`eval_fixed_command.py`，**512 envs / 1100 步 / seed 42 / train task**，三档固定命令 `(0,0,0)/(0.5,0,0)/(1.0,0,0)`。
* 多地形：`policy_report.py`，**云端 256 envs / 500 步（10 s/档）/ `--terrain-grid keep`**（本机跑不了训练级网格，见 DEF-031）。

---

## 一、低层运动策略（平地，`History-Adaptation-Deeprobotics-M20-v0`）

| 推荐度 | 模型（run / 文件） | 是什么 | 指标（三档 = `(0,0,0)/(0.5,0,0)/(1.0,0,0)`） |
|---|---|---|---|
| ⭐**首选** | `logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/model_19999.pt` | **cap1.2 定稿版**（静止专项软化权重 + 加强扰动 + 镜像修复 + ⑫ 修复 + `max_noise_std=1.2`） | `err_vel_xy` **0.1014 / 0.1301 / 0.1594**；摔倒率 **0.000 / 0.002 / 0.000**；回合长度 1000 / 999.4 / 1000；回报 37.6 / 52.2 / 50.6；训练回报 35.9、噪声平台 1.132、膝镜像 RMS 0.303 |
| ⭐次选 | `.../2026-09-30_00-09-25_cloud_soft20k/model_19999.pt` | 静止专项软化版 20k（cap=0），用来对比 cap 的增益 | 0.1067 / 0.1332 / 0.1900；摔倒率 0.0254 / 0.0156 / 0.0195 |
| 对照 | `.../2026-09-20_00-50-31/model_19999.pt` | **部署基线**（旧代码：有"右后腿撇"+静止漂移） | 0.1644 / 0.1620 / 0.1903；摔倒率 0.0370 / 0.0312 / 0.0254；膝镜像 RMS 1.114 |
| 诊断 | `.../2026-10-02_00-45-34_abl_legacyall_10k/model_9999.pt` | 只保留 ⑫ 的消融（= main 行为 + ⑫）：步态最对称 | 0.0916 / 0.0978 / 0.1150；摔倒率 0/0/0；hl~hr 膝差 **0.008 rad** |

**GUI 看它跑**

```bat
python scripts\reinforcement_learning\rsl_rl\play.py --task=History-Adaptation-Deeprobotics-M20-play-v0 --checkpoint=logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/model_19999.pt --num_envs=16 --real-time
```

**键盘交互看"静止还漂不漂"**（不按键 = 命令 (0,0,0)；`↑/↓` 前后、`←/→` 左右、`Z/X` 偏航）

```bat
python scripts\reinforcement_learning\rsl_rl\play.py --task=History-Adaptation-Deeprobotics-M20-play-v0 --checkpoint=logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/model_19999.pt --keyboard --real-time
```

**一页式体检报告（13 个角度，含 A/B / 抗扰扫描 / 每地形高度图）**

```bat
python scripts\reinforcement_learning\rsl_rl\policy_report.py --headless --task History-Adaptation-Deeprobotics-M20-play-v0 --checkpoint logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/model_19999.pt --compare logs/rsl_rl/history_adaptation/2026-09-20_00-50-31/model_19999.pt --label cap12_20k --label-b oldcode_20k --num_envs 64 --steps 400 --push-sweep "1,1;2,2;3,3" --out-dir logs/smoke/report_cap12_vs_old
```

不用再手写"指令切换"那套：默认就会在**一条 16 s 连续轨迹**里把
`vx/vy/wz`（13 段）与 `height/pitch/roll`（7 段）依次切换一遍
（`--seg-s` 改每段秒数、`--schedule none` 关掉）。

各图看什么（2026-10-05 第四轮改版后）：

| 图 | 内容 |
|---|---|
| `fig01` | 切换轨迹上的 vx/vy/wz 指令 vs 实际 + 逐轴误差（灰竖线 = 切换时刻） |
| `fig02` | 逐段稳态误差柱状（vx/vy/wz 各一格）+ 三张 cmd-vs-actual 散点 |
| `fig03` | height / pitch / roll **各一行**：左列时程、右列逐段稳态误差 |
| `fig04` | 每档速度指令一列的四足触地条带 + 占空比（分"滚动"与"迈步"） |
| `fig05/06/07` | 关节轨迹 / 力矩（含膝盖+轮子时程）/ 对称性（含不对称度、轮距时程） |
| `fig08` | 臂：EE 位置/姿态误差（均值虚线+数值）、**臂关节力矩** |
| `fig09` | 每个子地形一张**稠密高度热力图** + 该 env 真实轨迹 |
| `fig10/11/12` | A/B 对比 / 分地形指标 / 指令切换数字表 |
| `fig13` | **push 抗扰扫描**：力度×频率分档 ⇒ 生还率 / 尖刺频次 / 恢复时间（`--push-sweep` 才有） |

> 图内文字 2026-10-05 起全部改英文；数据仍全在 `data.npz` / `summary.json`。
> `--from-npz` 可以"云端采集 + 本机画图"（npz 分了 cmd/sched/arm/push 四组键）。

**抗扰能力（push 扫描，2026-10-05 云端实测；`logs/smoke/report_flatAB_new/`）**

`--push-sweep "力度倍数,频率倍数"`：1x1 = 训练口径（间隔 5~10 s、±2/±1/yaw±0.52），
>1 是**训练分布之外**的外推。64 envs × 16 s/档。

| 档位 | cap12_20k 终止 / env / 分钟 | cap12 最大瞬时误差 | oldcode_20k 终止 / env / 分钟 | oldcode 最大瞬时误差 |
|---|---|---|---|---|
| 1x1（训练口径） | **0.06**（恢复 0.31 s） | **1.83 m/s** | 0.12（恢复 0.54 s） | 1.84 m/s |
| 2x2 | **0.41** | 3.93 | 2.11 | 3.80 |
| 3x3（外推） | **3.23** | **4.98** | 7.10 | 6.40 |

⇒ `cap12_20k` 在外推区间里摔倒率只有旧代码的 **45%**、恢复时间 57% ⇒ 抗扰训练有效。

---

## 二、多地形策略（`Rough-Slopes-*`）

**本机能不能跑？——能看、能测，别用它做多环境训练。**（2026-10-05 更正）

* `play.py` 会把地形**自动压成 5×5 并关掉地形课程**（见 `play.py:101-104`），所以**任何多地形任务**
  在本机小环境数下都能跑，例如下面这条（本机实测可用）：

  ```bat
  python scripts\reinforcement_learning\rsl_rl\play.py --task=Rough-Slopes-SlowVx-History-Adaptation-Deeprobotics-M20-play-v0 --checkpoint=logs/rsl_rl/history_adaptation/2026-10-02_00-44-42_cloud_slowvx20k/model_19999.pt --num_envs=4
  ```
* 本机**会卡死**的只有一种组合：**训练任务 + 训练级网格（10 行 × 20 列 = 200 块） + 环境数 ≥48**
  （在 env 创建阶段挂住，CPU 近 0）。用 `policy_report.py` 时加 `--terrain-grid auto`（≤16 envs 时自动压 5×5）
  或 `--terrain-grid 5x5` 就能避掉；要做**多环境地形训练/大样本分地形统计**才去云端。

**v_x 命令范围（`commands.base_velocity.ranges.lin_vel_x`）**

| 场景 | 范围 | 什么时候到 ±5 |
|---|---|---|
| 平地 `History-Adaptation-*`（训练，`WBCCurriculumCfg`） | 台阶 **±2→±3→±4→±5** | 75k/100k/125k/**150k 环境步** ≈ **3125/4167/5208/6250 iter**（`num_steps_per_env=24`） |
| 多地形 `Rough-Slopes-*`（训练） | 同上（`RoughEnvWBCConfig` 里本来是 None，RoughSlopes 重新打开） | 同上 |
| 多地形 `Rough-Slopes-SlowVx-*`（训练） | 台阶**整体 ×2**：150k/200k/250k/**300k** | ±5 推到 **12500 iter**（终值仍是 ±5） |
| 任意 `-play-v0` | **固定 (-1, 1)**（`lin_vel_x/y/z` 都设 -1~1，四个台阶全部置 None） | 不会扩 |

> 所以：**训练时会跑到 ±5 m/s**（这也是 `error_vel_xy` 后期看着变大的原因之一，见 DEF-023）；
> **`-play-v0` 下只会采到 ±1**。固定命令 eval / `policy_report.py` 是**直接写死命令缓冲并关重采样**的，
> 与上面的范围无关。

| 推荐度 | 模型 | 是什么 |
|---|---|---|
| ⭐**首选** | `logs/rsl_rl/history_adaptation/2026-10-02_00-44-42_cloud_slowvx20k/model_19999.pt` | **SlowVx**（v_x 课程台阶 ×2 + cap1.2）——目前多地形最好 |
| 对照 | `.../2026-09-30_00-14-44_cloud_roughslopes20k/model_19999.pt` | 原多地形 20k（cap=0、原 v_x 课程） |

**训练期指标（同一 seed / 4096 envs / 20k iter，`summarize_run.py`）**

| 指标（末 1000 iter） | 原多地形 20k | **SlowVx 20k** |
|---|---|---|
| `Curriculum/terrain_levels` | 3.706（斜率 **−0.091/1k**，在退） | **4.74（+0.095/1k，还在涨）** |
| `Episode_Termination/bad_orientation_2` | 0.2065 | **0.1006（−51%）** |
| `Train/mean_reward` | 19.71 | **26.13（+33%）** |
| `ep_len` / `time_out` | ~901 / 0.80 | 同量级 |

**固定命令指标（云端 256 envs / 10 s/档 / 保持训练网格）** —— 原多地形 20k：

| 命令 | `err_vel_xy` | `err_vel_yaw` | 机身高度均值 / std | 俯仰均值 | 后腿左右不对称 | `hl~hr` 膝RMS | 终止 |
|---|---|---|---|---|---|---|---|
| (0,0,0) | **0.0479** | 0.0241 | 0.338 / 0.0172 | −3.05° | −2.38 cm | 0.133 | bad_orientation_2 ×11 |
| (0.5,0,0) | 0.1817 | 0.1207 | 0.468 / 0.0296 | +15.59° | −5.85 cm | 0.394 | bad_orientation_2 ×13 |
| (1.0,0,0) | **0.2467** | 0.1199 | 0.482 / 0.0173 | +11.27° | +11.36 cm | 0.344 | bad_orientation_2 ×10 |

**分地形（同一份 256-env 报告，按每个 env 落在哪种地形上分组）**

| 地形（env 数） | `err_vel_xy` @ (0,0,0) / (0.5) / (1.0) | 高度std | 力矩RMS最大 | 平均终止次数 | 最差腿触地占比 |
|---|---|---|---|---|---|
| `random_rough` 粗糙（103） | **0.0717** / 0.1430 / 0.1850 | 0.0141 | 43.7 | 0.03 | 0.307 |
| `hf_pyramid_slope` 上坡（64） | 0.0984 / 0.1595 / 0.2065 | 0.0148 | 45.9 | **0.10** | 0.441 |
| `hf_pyramid_slope_inv` 下坡（64） | 0.1118 / 0.1666 / **0.2326** | 0.0138 | 42.1 | 0.01 | 0.361 |
| `flat` 平地（25） | 0.0966 / 0.1139 / 0.1902 | 0.0130 | 42.5 | 0.03 | 0.291 |

⇒ **粗糙地形并不比平地差**；短板是**下坡的速度跟踪**（1.0 m/s 档 0.2326）与**上坡的稳定性**
（每 env 0.10 次终止）。

> ⚠️ 这两张表来自**旧版报告（已随 2026-10-05 第四轮一起清掉）**——口径是"3 档纯 vx 命令"。
> 新工具（第四轮）已把这些重跑成 9 档（含 vy/wz）+ push 扫描，产物在
> `logs/smoke/report_terrain_new/`：
>
> * 逐档 `err_vel_xy`（256 envs / 400 步/档）：`(0,0,0)` **0.0898**、`(0.3)` 0.1928、
>   `(0.8)` 0.2515、`(1.5)` 0.1930、`(0,0.4)` 0.3928、`(0,-0.4)` 0.3417、
>   `(0,0,0.6)` 0.1672、`(0,0,-0.6)` 0.1933、`(0.8,0.4,0.3)` 0.3264；
> * 分地形（`(0,0,0)` 档，**地形 × 命令分档**的新表格结构）：粗糙 **0.0916** ＜ 下坡 0.1117
>   ≈ 平地 0.1128 ＜ 上坡 0.1228；平均终止 上坡 **0.17** ＞ 平地/粗糙 0.04 ＞ 下坡 0.00；
>   最差腿触地占比：平地 0.750、上坡 0.125、粗糙 0.100、**下坡 0.043**（有一条腿几乎不触地）；
> * push 扫描：`1x1` 0.15 → `2x2` 0.69 → **`3x3` 4.88** 次/env/分钟（地形上的抗扰余量
>   比平地小：平地 cap12 在 3x3 是 3.23）。

**SlowVx 的同口径数字（256 envs / 10 s/档；旧版报告，同上）**

逐档：`(0,0,0)` err_xy **0.1016**、`(0.5)` **0.1438**、`(1.0)` **0.1760**；`bad_orientation_2` 各 9 / 9 / 10 次。

| 地形（env 数） | `err_vel_xy` @ (0,0,0) / (0.5) / (1.0) | 高度std | 力矩RMS最大 | 平均终止次数 | 最差腿触地占比 |
|---|---|---|---|---|---|
| `random_rough` 粗糙（103） | 0.0983 / **0.1961** / 0.1839 | 0.0145 | 43.4 | 0.03 | **0.182** |
| `hf_pyramid_slope` 上坡（64） | 0.1330 / 0.1951 / 0.1782 | 0.0161 | 45.2 | **0.08** | 0.251 |
| `hf_pyramid_slope_inv` 下坡（64） | **0.1495** / 0.1719 / 0.1943 | 0.0148 | 41.8 | **0.00** | 0.389 |
| `flat` 平地（25） | 0.1112 / 0.1606 / 0.1765 | 0.0148 | 42.2 | 0.04 | 0.574 |

⇒ 与原版相比：**低速档（0.5 / 1.0 m/s）全面更好**（0.1438 vs 0.1817、0.1760 vs 0.2467），
**静止档略差**（0.1016 vs 0.0479，因为 SlowVx 长期在 ±5 的大命令范围里训练），终止次数少 20~30%；
**两版的短板一致：下坡/平地上"最差那条腿的触地占比"偏低（0.39 / 0.57）**。

**它能爬到什么地形等级（训练的 v_x 是 ±5 的，所以这就是"±5 情况下"的答案）**

`terrain_levels`（0~9 共 10 行，难度 = 行号 / 9）：

| iter | 0 | 2000 | 4000 | 6000 | 8000 | 10000 | 12000 | 14000 | 16000 | 末 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| **SlowVx 20k** | 3.49 | 3.29 | 3.15 | 3.19 | 5.15 | 5.62 | **5.75（峰值）** | 5.67 | 5.28 | **4.74** |
| 原多地形 20k | 3.49 | 3.40 | 5.25 | **5.80（峰值）** | 2.84 | 2.91 | 3.06 | 3.26 | 3.52 | 3.71 |

* **原版**：±5 的 v_x 在 6250 iter 打开后地形等级**从 5.8 崩到 ~2.8**（末段 3.71，斜率 −0.09/1k）；
  同期 `ep_len` 掉到 ~780。
* **SlowVx**（把 ±5 推迟到 12500 iter）：**峰值 5.75、末段 4.74、斜率 +0.095/1k（还在涨）**，
  而且 8000 iter 之后再没掉到 4.6 以下，`ep_len` 稳定在 ~940。
* **换算成实际地形难度**（难度 = level/9）：末段 level 4.74 ⇒ **坡度 ≈ 0.21（≈12°）、粗糙噪声 ≈ 0.031 m**；
  峰值 5.75 ⇒ **≈14.4° 坡、≈0.036 m 粗糙**。也就是这套配置目前稳定在"**12° 斜坡 + 3 cm 随机粗糙**"这一档，
  离最难（0.4 = 21.8° 坡 / 0.05 m 粗糙）还有距离。

**在云端看它跑 / 出报告**（想要 200 块训练网格、或 256 envs 分地形统计时才需要）

```bat
:: 先 ssh 进实例（本机 cmd）：ssh -p 1237 root@10.60.144.11
:: 然后在实例上（Linux）：
python scripts/reinforcement_learning/rsl_rl/play.py --task=Rough-Slopes-SlowVx-History-Adaptation-Deeprobotics-M20-play-v0 --checkpoint=logs/rsl_rl/history_adaptation/2026-10-02_00-44-42_cloud_slowvx20k/model_19999.pt --num_envs=4
POLICY_REPORT_FONT=/root/fonts/simhei.ttf python scripts/reinforcement_learning/rsl_rl/policy_report.py --headless --task Rough-Slopes-SlowVx-History-Adaptation-Deeprobotics-M20-play-v0 --checkpoint logs/rsl_rl/history_adaptation/2026-10-02_00-44-42_cloud_slowvx20k/model_19999.pt --commands "0,0,0;0.5,0,0;1.0,0,0" --num_envs 256 --steps 500 --terrain-grid keep --out-dir /root/report_slowvx
```

> `-play-v0` 变体（2026-10-05 补）：`Rough-Slopes-SlowVx-History-Adaptation-Deeprobotics-M20-play-v0`
> （完整难度、关课程）；原来的 `Rough-Slopes-History-Adaptation-…-play-v0` 仍在，两个都能用。

---

## 三、部署态策略（高层 / 遥操 / sim2sim 用这个，不是 `model_*.pt`）

| 用途 | 路径 | 说明 |
|---|---|---|
| 高层四条任务的**默认**低层策略 | `logs/rsl_rl/history_adaptation/2026-09-20_00-50-31/exported_deploy/policy.pt` | 部署基线（旧代码） |
| **推荐换成** | `logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/exported_deploy/policy.pt` | cap1.2 版 |
| 消融 / 多地形对照 | `.../2026-10-02_00-45-34_abl_legacyall_10k/exported_deploy/policy.pt`、`.../2026-09-30_00-14-44_cloud_roughslopes20k/exported_deploy/policy.pt` | ⑫-only / 多地形 |

```bat
:: 遥操（VR）里看新低层策略
set RL_TRAINING_LOW_LEVEL_POLICY_TELEOP_HISTORY=logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/exported_deploy/policy.pt
python scripts\teleoperation\teleop_vr_m20_piper.py --task Isaac-M20-Piper-Teleop-History-v0 --num_envs 1
:: 纯回放（不开 VR）
python scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-M20-Piper-Teleop-History-v0 --num_envs=4 --real-time
:: 重新导出部署态策略（换模型后必做；自检"与 rsl_rl act_inference 的最大误差"应为 0.000e+00）
python scripts\reinforcement_learning\rsl_rl\export_deploy_policy.py --run logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k --checkpoint model_19999.pt
```

---

## 四、消融 / 对照模型（只在查"哪个改动起作用"时用）

| 模型 | 是什么 | 一句话结论 |
|---|---|---|
| `.../2026-10-02_00-45-34_abl_legacyall_10k/model_9999.pt` | 三项改动全退、只留 ⑫ | **⑫ 是"右后腿撇"的根因**（膝差 1.156→0.008） |
| `.../2026-09-30_11-20-59_abl_pushonly_10k/model_9999.pt` | 只加强扰动 | `(0,0,0)` err 0.0968 |
| `.../2026-09-30_19-33-26_abl_rewardonly_10k/model_9999.pt` | 只改静止奖励+镜像 | 零速略好但高速变差（0.1361/0.2018/0.2267） |
| `.../2026-09-30_00-09-25_cloud_soft20k/model_10000.pt` | 两样都改 @10k | 0.1074/0.1429/0.2337 |

---

## 五、盘点 / 出新指标的模板

```bat
:: 列出所有 run：模型数 / 最大迭代 / 体积 / 是否已导出部署态
powershell -Command "Get-ChildItem logs\rsl_rl -Directory | ForEach-Object { Get-ChildItem $_.FullName -Directory } | ForEach-Object { $m=@(Get-ChildItem $_.FullName -Filter 'model_*.pt'); $it = if ($m) { ($m | %% { [int]($_.BaseName -replace 'model_','') } | Measure-Object -Maximum).Maximum } else { -1 }; '{0,-46} n={1,-4} maxIter={2,-7} export={3}' -f $_.Name, $m.Count, $it, (Test-Path (Join-Path $_.FullName 'exported_deploy')) }"

:: 新模型出指标（固定命令 eval）
python scripts\reinforcement_learning\rsl_rl\eval_fixed_command.py --headless --num_envs 512 --steps 1100 --seed 42 --commands "0,0,0;0.5,0,0;1.0,0,0" --task History-Adaptation-Deeprobotics-M20-v0 --checkpoint <新模型> --label <标签> --out logs/smoke/eval_<标签>.json

:: 步态对称性（镜像 RMS / 左右不对称 / 前后轮距）
python scripts\reinforcement_learning\rsl_rl\probe_gait_symmetry.py --headless --num_envs 256 --steps 600 --warmup 150 --commands "1.0,0,0" --checkpoint <新模型> --label <标签> --out logs/smoke/gait_<标签>.json

:: 训练曲线（阶段均值 + 首末段窗口 + 两 run 对比）
python scripts\reinforcement_learning\rsl_rl\summarize_run.py --run <runA> --baseline <runB> --tags "terrain_levels|mean_reward|bad_orientation_2"
```

> 相关文档：性能变化的来龙去脉在 `docs/review/DONE_zh.md`（第七 / 十二~十六节）与
> `docs/review/DEFECT_LOG_zh.md`（DEF-026~045）；训练/部署说明见 `docs/train_history_flat_zh.md`、
> `docs/deploy_sim2sim_sim2real_zh.md`。
