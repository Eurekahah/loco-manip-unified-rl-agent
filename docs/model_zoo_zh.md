# 模型清单 / 最佳策略索引（可更新）

**用途**：一眼找到"现在该用哪个 checkpoint、怎么在本地把它跑起来看、它的性能大概什么水平"。
新增/替换模型时**只改这一个文件**（表格里加一行 + 在 `DEFECT_LOG_zh.md` 记一条）。

**维护约定**

* 只收录**验收过**的模型（有数字、有对照）；"最佳"按固定命令 eval 的 `err_vel_xy` 与摔倒率排。
* 所有指标口径：`eval_fixed_command.py`（512 envs / 1100 steps / seed 42 / train task）或
  `policy_report.py`（云端 64 envs / 500 steps = 10 s/档）；**跨口径不要混着比**。
* 命令模板里的 `$PY` = `C:\Users\autolab\miniconda3\envs\env_isaac_lab\python.exe`，
  跑前先 `$env:PYTHONIOENCODING='utf-8'`，仓库根目录下执行。
* 反查某个 run 的模型数量/最大迭代：见文末"盘点命令"。

---

## 一、低层运动策略（平地，`History-Adaptation-Deeprobotics-M20-v0`）

| 推荐度 | 模型（run / 文件） | 是什么 | 关键指标（固定命令 eval，三档 `(0,0,0)/(0.5,0,0)/(1.0,0,0)`） |
|---|---|---|---|
| ⭐**首选** | `logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/model_19999.pt` | **cap1.2 定稿版**（`max_noise_std=1.2`，静止专项软化权重 + 加强扰动 + 镜像修复 + ⑫ 修复） | `err_vel_xy` **0.1014 / 0.1301 / 0.1594**；摔倒率 **0.000 / 0.002 / 0.000**；回合长度 1000/999.4/1000；训练回报 35.9、噪声平台 1.132 |
| ⭐次选 | `.../2026-09-30_00-09-25_cloud_soft20k/model_19999.pt` | 静止专项软化版 20k（cap=0），用来对比"cap 带来的增益" | 0.1067 / 0.1332 / 0.1900；摔倒率 0.0254 / 0.0156 / 0.0195 |
| 对照 | `.../2026-09-20_00-50-31/model_19999.pt` | **部署基线**（旧代码，含"右后腿撇"与静止漂移） | 0.1644 / 0.1620 / 0.1903；摔倒率 0.0370 / 0.0312 / 0.0254 |
| 诊断 | `.../2026-10-02_00-45-34_abl_legacyall_10k/model_9999.pt` | 只保留 ⑫ 的消融（= main 行为 + ⑫）：步态最对称 | 0.0916 / 0.0978 / 0.1150；摔倒率 0/0/0；hl~hr 膝差 **0.008 rad** |

**本地看它跑（GUI）**

```powershell
$PY scripts\reinforcement_learning\rsl_rl\play.py `
  --task=History-Adaptation-Deeprobotics-M20-play-v0 `
  --checkpoint=logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/model_19999.pt `
  --num_envs=16 --real-time
```

**键盘交互看"静止还漂不漂"**（不按键 = 命令 (0,0,0)；`↑/↓` 前后、`←/→` 左右、`Z/X` 偏航）：

```powershell
$PY scripts\reinforcement_learning\rsl_rl\play.py `
  --task=History-Adaptation-Deeprobotics-M20-play-v0 `
  --checkpoint=logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/model_19999.pt `
  --keyboard --real-time
```

**一页式体检报告（10+ 个角度，含 A/B）**：

```powershell
$PY scripts\reinforcement_learning\rsl_rl\policy_report.py --headless `
  --task History-Adaptation-Deeprobotics-M20-play-v0 `
  --checkpoint logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/model_19999.pt `
  --compare    logs/rsl_rl/history_adaptation/2026-09-20_00-50-31/model_19999.pt `
  --label cap12_20k --label-b oldcode_20k `
  --commands "0,0,0;0.5,0,0;1.0,0,0" --num_envs 32 --steps 500 --out-dir logs/smoke/report_cap12_vs_old
```

---

## 二、多地形策略（`Rough-Slopes-*`，本机跑不了 ⇒ 云端跑）

| 推荐度 | 模型 | 是什么 | 关键指标 |
|---|---|---|---|
| ⭐**首选** | `logs/rsl_rl/history_adaptation/2026-10-02_00-44-42_cloud_slowvx20k/model_19999.pt` | **SlowVx**（v_x 课程台阶 ×2 + cap1.2）——目前多地形最好 | `terrain_levels` 末 1000 **4.74**（斜率 +0.095/1k，还在涨）；`bad_orientation_2` 末 1000 **0.1006**；回报末 1000 **26.13** |
| 对照 | `.../2026-09-30_00-14-44_cloud_roughslopes20k/model_19999.pt` | 原多地形 20k（cap=0、原 v_x 课程） | `terrain_levels` 末 1000 3.706（**斜率 −0.091/1k，在退**）；`bad_orientation_2` 0.2065；回报 19.71 |
| 分地形诊断 | 同上（`cloud_slowvx20k` 的分地形表见 `logs/smoke/report_terrain256/report.md`） | 256 envs 分地形：粗糙 0.0717 ＜ 平地 0.0966；**下坡最差 0.2326、上坡最易摔 0.10 次/env** | 说明短板在坡面（不是粗糙度） |

**云端看它跑**（本机 `terrain_type=generator` 在训练级网格会卡住，见 DEF-031）：

```bash
# 在 autodl 实例上（ssh -p <port> root@10.60.144.11）
python scripts/reinforcement_learning/rsl_rl/play.py --task=Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0 \
  --checkpoint logs/rsl_rl/history_adaptation/2026-10-02_00-44-42_cloud_slowvx20k/model_19999.pt --num_envs=4
# 或者只跑报告（云端画中文需先传字体）：
POLICY_REPORT_FONT=/root/fonts/simhei.ttf python scripts/reinforcement_learning/rsl_rl/policy_report.py --headless \
  --task Rough-Slopes-History-Adaptation-Deeprobotics-M20-play-v0 --checkpoint <ckpt> \
  --commands "0,0,0;0.5,0,0;1.0,0,0" --num_envs 256 --steps 500 --terrain-grid keep --out-dir /root/report_x
```

---

## 三、部署态策略（高层任务 / 遥操 / sim2sim 用这个，不是 `model_*.pt`）

| 用途 | 路径 | 说明 |
|---|---|---|
| 高层四条任务的**默认**低层策略 | `logs/rsl_rl/history_adaptation/2026-09-20_00-50-31/exported_deploy/policy.pt` | 部署基线（旧代码） |
| **推荐换成** | `logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/exported_deploy/policy.pt` | cap1.2 版；用法：`$env:RL_TRAINING_LOW_LEVEL_POLICY_TELEOP_HISTORY='<此路径>'` |
| 消融对照 | `.../2026-10-02_00-45-34_abl_legacyall_10k/exported_deploy/policy.pt`；`.../2026-09-30_00-14-44_cloud_roughslopes20k/exported_deploy/policy.pt` | ⑫-only / 多地形 |

```powershell
# 遥操里看新低层策略（VR）
$env:RL_TRAINING_LOW_LEVEL_POLICY_TELEOP_HISTORY='logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k/exported_deploy/policy.pt'
$PY scripts\teleoperation\teleop_vr_m20_piper.py --task Isaac-M20-Piper-Teleop-History-v0 --num_envs 1
# 或纯回放
$PY scripts\reinforcement_learning\rsl_rl\play.py --task=Isaac-M20-Piper-Teleop-History-v0 --num_envs=4 --real-time
```

> 部署态策略要**重新导出**时：`$PY scripts\reinforcement_learning\rsl_rl\export_deploy_policy.py --run <run> --checkpoint model_19999.pt`
> （自检 `与 rsl_rl act_inference 的最大误差` 应为 `0.000e+00`）。

---

## 四、消融/对照模型（只在查"哪个改动起作用"时用）

| 模型 | 是什么 | 一句话结论 |
|---|---|---|
| `.../2026-10-02_00-45-34_abl_legacyall_10k/model_9999.pt` | 三项改动全退、只留 ⑫ | **⑫ 是"右后腿撇"的根因**（膝差 1.156→0.008） |
| `.../2026-09-30_11-20-59_abl_pushonly_10k/model_9999.pt` | 只加强扰动 | `(0,0,0)` err 0.0968（与"⑫ only"同档） |
| `.../2026-09-30_19-33-26_abl_rewardonly_10k/model_9999.pt` | 只改静止奖励+镜像 | 零速略好但高速变差（0.1361/0.2018/0.2267） |
| `.../2026-09-30_00-09-25_cloud_soft20k/model_10000.pt` | 两样都改 @10k | 0.1074/0.1429/0.2337 |

---

## 五、盘点命令（想再加一行时用）

```powershell
# 列出所有 run：模型数 / 最大迭代 / 体积 / 是否有部署态导出
Get-ChildItem logs\rsl_rl -Directory | ForEach-Object { Get-ChildItem $_.FullName -Directory } | ForEach-Object {
  $m = @(Get-ChildItem $_.FullName -Filter 'model_*.pt'); $it = if ($m) { ($m | % { [int]($_.BaseName -replace 'model_','') } | Measure-Object -Maximum).Maximum } else { -1 }
  "{0,-46} n={1,-4} maxIter={2,-7} export={3}" -f $_.Name, $m.Count, $it, (Test-Path (Join-Path $_.FullName 'exported_deploy'))
}

# 新模型出指标（固定命令 eval；跨口径不可比）
$PY scripts\reinforcement_learning\rsl_rl\eval_fixed_command.py --headless --num_envs 512 --steps 1100 --seed 42 `
  --commands "0,0,0;0.5,0,0;1.0,0,0" --task History-Adaptation-Deeprobotics-M20-v0 --checkpoint <新模型> --label <标签> --out logs/smoke/eval_<标签>.json

# 步态对称性（镜像 RMS / 左右不对称 / 前后轮距）
$PY scripts\reinforcement_learning\rsl_rl\probe_gait_symmetry.py --headless --num_envs 256 --steps 600 --warmup 150 `
  --commands "1.0,0,0" --checkpoint <新模型> --label <标签> --out logs/smoke/gait_<标签>.json
```

> 相关文档：性能变化的来龙去脉在 `docs/review/DONE_zh.md`（第七 / 十二~十五节）与
> `docs/review/DEFECT_LOG_zh.md`（DEF-026~044）；训练/部署说明见 `docs/train_history_flat_zh.md`、
> `docs/deploy_sim2sim_sim2real_zh.md`。
