# 模型清单（可更新）—— 哪个 checkpoint 好用、怎么跑起来

**用途**：`logs/` 下现在有 **171 个 run 目录**（2026-10-05 统计），其中绝大多数是
2-iteration 的冒烟/探针 run（只有 `model_0.pt` / `model_1.pt`），**不能当模型用**。
这份文档把"值得用的模型 + 对应 play 命令 + 实测指标"整理成一张表，避免每次翻目录。

**维护约定**：① 新增可用模型时在下面表 A 加一行（写在最上面）；② 指标一律注明口径
（不同 env 数/步数不可混比）；③ 每次更新在文末"更新记录"加一行。

---

## 0. 先学会分辨"可用模型"和"冒烟 run"

| 特征（看 run 目录里 `model_*.pt` 的数量 / 最大迭代号） | 结论 |
|---|---|
| `#model ≥ 9` 且 `maxIter ≥ 999`（例：`41 / 19999`、`9 / 3999`、`2 / 999`） | **可用**（训练到位的策略） |
| `#model ≤ 4` 且 `maxIter ≤ 100` | **冒烟/探针 run**（`smoke_regression.py`、`probe_*.py`、timing 测试留下的），**不要用** |
| `#model = 1~2` 且 `maxIter = 0~1` | 同上（最常见的噪音来源，`high_level_pick_flat_teacher/` 下有 100+ 个） |

一键统计（本机，不启动 Isaac）：

```powershell
& $PY -c "import os,re,glob;  [print(f'{len(glob.glob(os.path.join(d,\"model_*.pt\"))):>4} {max([int(re.search(r\"model_(\d+)\",os.path.basename(p)).group(1)) for p in glob.glob(os.path.join(d,\"model_*.pt\"))]) if glob.glob(os.path.join(d,\"model_*.pt\")) else -1:>7} {d}') for d in sorted(glob.glob('logs/rsl_rl/*/*')) if glob.glob(os.path.join(d,'model_*.pt'))]"
```

---

## A. 推荐使用的模型（性能最好，按用途）

指标口径统一为：**固定命令 eval，512 envs × 1100 步 / seed 42**（`eval_fixed_command.py`），
三档命令 `(0,0,0) / (0.5,0,0) / (1.0,0,0)` 的 `err_vel_xy`；"摔倒"= 终止构成里的
`bad_orientation_2 + root_height_below_minimum`。**play 命令只是用来看/录，不是指标口径。**

| # | 用途 | checkpoint | 关键指标（近似） | play 命令（`--num_envs=1`） |
|---|---|---|---|---|
| **1** | **平地低层（首选）** | `logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/model_19999.pt` | `err_xy` **0.1014 / 0.1301 / 0.1594**、摔倒 **0.000/0.002/0.000**、`hl~hr` 膝镜像 RMS **0.303**、噪声平台 1.13、训练回报 35.9 | `python scripts/reinforcement_learning/rsl_rl/play.py --task=History-Adaptation-Deeprobotics-M20-play-v0 --checkpoint=logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/model_19999.pt --num_envs=1` |
| **2** | 平地低层（无 noise-cap 对照，用来做 A/B） | `.../2026-09-30_00-09-25_cloud_soft20k/model_19999.pt` | `err_xy` 0.1067 / 0.1332 / 0.1900、摔倒 0.025/0.016/0.020、膝镜像 RMS 0.374 | 同上，换 `--checkpoint` |
| **3** | **多地形（首选）** | `.../2026-10-02_00-44-42_cloud_slowvx20k/model_19999.pt` | `terrain_levels` 末 1000 **4.74（+0.095/1k，仍在涨）**、`bad_orientation_2` 0.1006、回报 26.1；（地形固定命令 eval 还没做） | `... --task=Rough-Slopes-SlowVx-History-Adaptation-Deeprobotics-M20-v0 --checkpoint=... --num_envs=1`（**本机跑不了生成地形，要在云端**） |
| **4** | 多地形（原版对照） | `.../2026-09-30_00-14-44_cloud_roughslopes20k/model_19999.pt` | `terrain_levels` 末 1000 3.706（斜率 −0.09/1k）、0.1006→0.2065 摔倒、回报 19.7；**分地形**：粗糙 0.0717/0.1430/0.1850、上坡 0.0984/0.1595/0.2065、下坡 0.1118/0.1666/**0.2326**、平地 0.0966/0.1139/0.1902 | 同上，`--task=Rough-Slopes-History-Adaptation-Deeprobotics-M20-play-v0` |
| **5** | **步态最对称 + 零速最好**（"⑫ only"消融） | `.../2026-10-02_00-45-34_abl_legacyall_10k/model_9999.pt` | `err_xy` **0.0916 / 0.0978 / 0.1150**、摔倒 **0/0/0**、hl/hr 膝差 **0.008 rad**、膝镜像 RMS 0.334 | `--task=History-Ablation-LegacyAll-Deeprobotics-M20-v0 --num_envs=1` |
| 6 | 消融：只加强扰动 | `.../2026-09-30_11-20-59_abl_pushonly_10k/model_9999.pt` | 0.0968 / 0.1097 / 0.1401、膝差 0.065 | `--task=History-Ablation-PushOnly-Deeprobotics-M20-v0` |
| 7 | 消融：只改奖励 | `.../2026-09-30_19-33-26_abl_rewardonly_10k/model_9999.pt` | 0.1361 / 0.2018 / 0.2267、`fl~hr` 膝 RMS 0.749（四格最好） | `--task=History-Ablation-RewardOnly-Deeprobotics-M20-v0` |
| 8 | **旧代码 20k**（所有对照的基线） | `.../2026-09-20_00-50-31/model_19999.pt` | 0.1644 / 0.1620 / 0.1903、hl/hr 膝差 0.957（"右后腿撇"） | `--task=History-Adaptation-Deeprobotics-M20-play-v0` |
| 9 | **部署基线（部署/ sim2sim 口径）** | `.../2026-09-20_00-50-31/exported_deploy/{policy.pt,policy.onnx,policy_layout.json}` | 与上面同一策略，但导出成推理图（torchscript 自检 0.000e+00） | 高层任务用（见下） |

### 高层 / 遥操任务要用的**低层部署态策略**

高层任务不是加载 `.pt` 训练权重，而是加载**导出态** `exported_deploy/policy.pt`（含 `policy_layout.json`），
用环境变量指过去即可：

| 高层任务 | 默认低层策略 | 换成"新配方"的命令 |
|---|---|---|
| `Isaac-M20-Piper-Teleop-History-v0`（VR 遥操） | `logs/rsl_rl/history_adaptation/2026-09-20_00-50-31/exported_deploy/policy.pt` | `$env:RL_TRAINING_LOW_LEVEL_POLICY_TELEOP_HISTORY='logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/exported_deploy/policy.pt'` 然后 `--task=Isaac-M20-Piper-Teleop-History-v0 --num_envs=1` |
| `Isaac-Deeprobotics-High-Level-Pick-Flat-Teacher-v0` | `logs/rsl_rl/deeprobotics_m20_flat/2026-04-21_00-02-23/exported/policy.pt` | `$env:RL_TRAINING_LOW_LEVEL_POLICY_FLAT='<其他>/exported_deploy/policy.pt'` |
| `...-Pick-WBC-Flat-Teacher-v0` | `logs/rsl_rl/deeprobotics_m20_wbc_flat/2026-09-18_01-31-58/exported/policy.pt` | `$env:RL_TRAINING_LOW_LEVEL_POLICY_WBC='<其他>/exported_deploy/policy.pt'` |

常用 play 命令模板（本机，单环境，肉眼看）：

```powershell
cd D:\nvidia-isaac-sim\loco-manip-unified-rl-agent
$env:PYTHONIOENCODING='utf-8'
$PY='C:\Users\autolab\miniconda3\envs\env_isaac_lab\python.exe'

# ① 键盘交互看静止漂移（不按键 = 零速命令）
& $PY scripts\reinforcement_learning\rsl_rl\play.py --task=History-Adaptation-Deeprobotics-M20-play-v0 `
    --checkpoint=<上面的 checkpoint> --num_envs=1 --keyboard --real-time
# ② 多环境并排看步态（--num_envs=16 --real-time）
# ③ 录 mp4：加 --video --video_length=400 --headless（视频落在 <run>/videos/play/）
# ④ 遥操：& $PY scripts\teleoperation\teleop_vr_m20_piper.py --task=Isaac-M20-Piper-Teleop-History-v0 --num_envs=1
```

---

## B. 冒烟/探针 run（**不要拿来当模型**）

特征：`#model ≤ 4` 且 `maxIter ≤ 100`。2026-10-05 统计约 **100+ 个**，主要来源与目录：

| 来源 | 典型目录 | 说明 |
|---|---|---|
| 回归矩阵（每次改动都跑） | `high_level_pick_flat_teacher/2026-09-3x_*`、`history_adaptation/2026-09-3x_*`（时间戳形式，如 `2026-09-30_15-04-37`） | 2 iter 冒烟，用来验证"建环境 + 观测维度"没问题 |
| 步态/EE/历史窗口等探针 | `history_adaptation/*_timing_probe*`、`history_adaptation/2026-09-19_14-16-49` | probe 脚本自己建的 env |
| 遥操计时测试 | `high_level_pick_flat_teacher/*teleop_timing*` | 只有 11 步 |

**清理建议**：只想省磁盘的话，直接删这些 run 的 `model_*.pt`（保留 `params/` 与 `events*` 便于回溯），
或用"`#model ≤ 4 且 maxIter ≤ 100`"这条规则挑——**不要删表 A 里那 9 个**。

---

## C. 怎么更新这张表（指标怎么重算）

```powershell
# 固定命令 eval（表 A 的主口径；跨 checkpoint 唯一合法口径）
& $PY scripts\reinforcement_learning\rsl_rl\eval_fixed_command.py --headless --num_envs=512 --steps=1100 --seed=42 `
  --commands "0,0,0;0.5,0,0;1.0,0,0" --task=History-Adaptation-Deeprobotics-M20-v0 `
  --checkpoint=<ckpt> --label=<名字> --out=logs/smoke/eval_<名字>.json

# 步态对称性（膝差 / 镜像 RMS / 足端俯视图 / 轮距）
& $PY scripts\reinforcement_learning\rsl_rl\probe_gait_symmetry.py --headless --num_envs=256 --steps=800 --commands "1.0,0,0" `
  --checkpoint=<ckpt> --label=<名字> --out=logs/smoke/gait_<名字>.json

# 训练曲线（阶段均值；不启动 Isaac）
& $PY scripts\reinforcement_learning\rsl_rl\summarize_run.py --run=<run 目录> --baseline=<对照 run> --tags "mean_reward|terrain_levels"

# 一站式体检报告（10+ 个角度，含分地形/指令切换；地形任务建议在云端跑）
& $PY scripts\reinforcement_learning\rsl_rl\policy_report.py --headless --task=<task> --checkpoint=<ckpt> `
  --commands "0,0,0;0.5,0,0;1.0,0,0" --num_envs=64 --steps=500 --out-dir=logs/smoke/report_<名字>
```

> 口径提醒：`eval_fixed_command.py` **不关 push 事件** ⇒ 各任务"摔倒率"列不可跨任务比
> （加强扰动的变体考核更狠）；噪声上限/课程等差异也要一起说。详见 `DEFECT_LOG_zh.md` DEF-040 §5。

---

## 更新记录

| 日期 | 更新内容 |
|---|---|
| 2026-10-05 | 初版：扫 `logs/` 全部 171 个 run；列出 9 个推荐模型（含多地形 SlowVx、⑫-only 消融）+ 高层/遥操的低层部署态策略映射 + 冒烟 run 的识别与清理建议 + 指标重算命令 |
