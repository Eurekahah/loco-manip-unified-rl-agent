# 已完成清单（DONE）—— 按主题

**文档职责**：记录"已经做完并且有实测验收"的事情（含 commit 与关键数字）。
未完成的在 `TODO_zh.md`；每条缺陷/特性的现象→原因→修正→结果在 `DEFECT_LOG_zh.md`。

**维护约定**：见 `templates/DOC_TEMPLATE_zh.md`。完成任务时从 TODO 迁到这里，
**保留日期与 commit**；只写结论与验收数字，过程细节写进 DEFECT_LOG。

## 更新记录

| 日期 | 更新内容 | 相关 commit / 分支 |
|---|---|---|
| 2026-09-20 | 初版：合并 6 份旧文档里的"已修"条目；记录低层内容并入 main | `main @ 7ff5b86` |
| 2026-09-20 | 新增第五节：高层链并入 `main`（4 个高层任务 2 iter 全 EXIT=0）+ 导出部署态策略固化成流程；第二节标题去掉"未合并 main" | `codex/hl-merge-p0`（`129848e`/`af4602d`/`07601e9`/`0772757`） |
| 2026-09-20 | 新增第六节 **部署基线**：`main @ 2d49f47`（tag `deploy-baseline-2026-09-20`）= 部署口径代码，对应 run `2026-09-20_00-50-31` 的 `exported_deploy/*`（含 sha256 与"训练代码 vs main"的差异核对）+ **基线可运行性验收**（8 任务冒烟回归 8/8 EXIT=0） | 基线 `2d49f47`；记录 `eb22401` |
| 2026-09-22 | 新增第一节 **P1-1 修复**：探索噪声上界 `max_noise_std=1.2` 的 A/B 实测通过（**建议作为默认**）；同批对照点 `entropy_coef=0.002`（通过但略逊）与 `entropy_coef=0.0`（意外点，s3 摔倒反而 +15%） | 开关代码 `b75c596`；实测回填见 DEF-024 §4（2026-09-22） |
| 2026-09-29 | 新增第七节 **训练细节专项**（分支 `codex/ll-train-detail-fix`）：静止伫立（DEF-026）、镜像符号（DEF-027）、扰动加强（DEF-028）、多地形任务（DEF-029）、遥操 history 任务（DEF-030）；另记录本机跑不了生成地形任务的平台问题（DEF-031） | `codex/ll-train-detail-fix` |
| 2026-09-30 | 新增第八节 **本机收尾批**：回归矩阵脚本化（13 任务 11 OK / 1 SKIP / 0 FAIL）+ known_issues ⑧⑩⑪⑫⑱ + 三处工程债（DEF-034） | `codex/ll-train-detail-fix` |
| 2026-09-30 | 新增第九节 **P2：高层 action term 收敛**：3 个 action term 继承 `LowLevelPolicyActionBase`（减 ~600 行重复机械）+ 删 `PreTrainedPolicyAction` + 复位时机探针（DEF-035） | `codex/ll-train-detail-fix` |
| 2026-09-30 | 新增第十节 **P3：EE 锚点对照一键化**：新脚本 `sweep_ee_anchor.py`（4 组 `full/default/low/cfg`，逐组子进程 + 对比表 + 汇总 JSON）+ 修掉旧探针 `none ≡ low` 的语义歧义（DEF-036） | `codex/ll-train-detail-fix` |

---

## 七、训练细节专项（2026-09-29，分支 `codex/ll-train-detail-fix`）

一次性处理"训出来的模型细节不对"的 5 件事。代码全部落地，冒烟全部 EXIT=0；
训练类结论见下面每条的"验收"。

| 需求 | 内容 | 关键改动 | 验收 |
|---|---|---|---|
| ① 静止时轮足仍有前向速度 | 基线实测命令 (0,0,0) 时 `err_vel_xy = **0.148 m/s**`（`eval_fixed_command.py`，512 envs / seed 42 / 1100 steps） | 新增 `stand_still_vel_l2`（−8.0）+ `stand_still_wheel_vel_l2`（−0.01）两项**只按命令门控**的惩罚；`rel_standing_envs` 0.02→**0.15**；三条 25k 步爬升课程（`DEFECT_LOG_zh.md` DEF-026） | 2 iter EXIT=0（奖励 23 项 / 课程 15 项）；2000-iter 训练的固定命令 eval 见下 |
| ② 右后腿往右前方撇 | 判定为**奖励项符号 bug**，不是策略调不出来 | `joint_mirror` 的对角对要求 `θ_fl=θ_hr`，而本机型真实镜像关系是 `θ_fl=−θ_hr`（MJCF 轴 / 关节限位 / 默认姿态三重证据）；新增 `joint_mirror_signed`（按关节名后缀给符号）+ 扩到 4 对（含左右对），权重 −0.03→−0.06（DEF-027） | 2 iter EXIT=0；`joint_mirror −0.06` 在奖励表里 |
| ③ 多地形（只随机粗糙+正反斜坡+平地） | 新增 `ROUGH_SLOPES_FLAT_TERRAINS_CFG`（粗糙 0.40 噪声 **0.01~0.05**、上坡 0.25、下坡 0.25、平地 0.10） | 新任务 `Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0`（+`-play-v0`），继承 `RoughEnvWBCConfig`，显式开地形课程并按 `RoughWOStairs` 的步骤重开 v_x 课程（DEF-029） | cfg 级验证通过；**端到端冒烟被平台问题挡住**（DEF-031：本机跑任何生成地形任务都会在 env 创建期死锁，原始代码同样复现） |
| ④ 加强扰动 | push 间隔 5~10 s、vx±2 / vy±1 / **yaw±0.52** | `EventCfg.randomize_push_robot` 改幅度与间隔；`disturbance_ramp` 的 base 同步改、`start_scale` 0.3→0.2、`num_steps` 25k→**50k**（DEF-028） | 2 iter EXIT=0（平地 history） |
| ⑤ 遥操缺 history 版 | `TeleopLLAction` 本体早已支持 history 回放，缺的只是注册 | 新增 `Isaac-M20-Piper-Teleop-History-v0`（`TeleopHistoryActionsCfg`/`TeleopHistoryEnvCfg`，低层默认指向 history 部署态策略，可用 `RL_TRAINING_LOW_LEVEL_POLICY_TELEOP_HISTORY` 覆盖）（DEF-030） | 2 iter EXIT=0，日志 `低层 obs 维度=83 (checkpoint 期望 83)` + `history 窗口: 10 × 70 = 700` |

### 冒烟回归（2026-09-29，`--headless --num_envs 64 --max_iterations 2`）

| 任务 | EXIT | 备注 |
|---|---|---|
| `History-Adaptation-Deeprobotics-M20-v0` | 0 | 奖励 21→**23** 项、课程 12→**15** 项；两条新惩罚与三条课程都在爬升 |
| `Flat-Deeprobotics-M20-Piper-WBC-v0` | 0 | 共享同一批奖励/课程类 ⇒ 回归通过 |
| `Isaac-M20-Piper-Teleop-v0` | 0 | 未受影响 |
| `Isaac-M20-Piper-Teleop-History-v0` | 0 | **新增**；replay 打印 `history 窗口: 10 × 70 = 700` |
| `Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0` | **SKIP** | 卡在平台问题 DEF-031（原始代码 + `Rough-WO-Stairs` 同样复现） |
| `Rough-WO-Stairs-History-Adaptation-Deeprobotics-M20-v0` | **SKIP** | 同上 |

> 复现：`logs/smoke/2026-09-29_<task>.log`；平台问题的对照实验见 DEF-031 §2。

### 静止伫立训练验收（需求 ①）—— 两轮，结论是"漂移修好了，但第一版权重太狠"

**怎么做对照**：训练日志里的 `Train/mean_reward` 带**命令课程**，跨 run 不可比；
而且新增的惩罚项本身就改变了总回报的尺度。所以判定只用两件东西：
① 固定命令探针 `eval_fixed_command.py`；② **同代对照**（同一个 run 自己的 `model_<iter>.pt`），
因为 `2026-09-20_00-50-31` 这个基线 run 是按 500 iter 存盘的，可以直接取它的 `model_1500` 当"旧代码同代"。

#### 1. A/B：新代码 1500 iter vs 旧代码同代（seed 42 / 4096 envs / 每档 1100 步 / 512 envs）

**训练任务**（`History-Adaptation-Deeprobotics-M20-v0`，s0 难度、弱扰动）：

| 命令 | 指标 | 旧代码 @1500 | **新代码 @1500**（-8.0/-0.01） | 旧代码 @20000 |
|---|---|---|---|---|
| (0,0,0) | `err_vel_xy` (m/s) | 0.1153 | **0.0886**（−23%） | 0.1644 |
| (0,0,0) | 回合长度 | 903.6 | 597.5 | 981.9 |
| (0,0,0) | 摔倒率 | 0.1780 | **0.7080** | 0.0370 |
| (0.5,0,0) | `err_vel_xy` | 0.1204 | 0.1505 | 0.1619 |
| (0.5,0,0) | 摔倒率 | 0.1686 | 0.3679 | 0.0312 |

**-play- 任务**（完整难度 + 全量扰动）：

| 命令 | 指标 | 旧代码 @1500 | 新代码 @1500 | 旧代码 @20000 |
|---|---|---|---|---|
| (0,0,0) | `err_vel_xy` (m/s) | 0.1475 | **0.1157**（−22%） | 0.1478 |
| (0,0,0) | 摔倒率 | 0.6287 | 0.6991 | 0.1332 |
| (0.5,0,0) | `err_vel_xy` | 0.1466 | 0.1742 | 0.1583 |
| (1.0,0,0) | `err_vel_xy` | — | 0.2253 | 0.1920 |

**读法**：目标指标（命令为 0 时的残余速度）**两档难度都稳定降 22~23%**，方向完全对；
但**摔倒率明显变差**，而且终止构成几乎全是 `bad_orientation_2`（翻倒）：
训练任务 cmd=0 上，新代码 685 个回合里 473 次翻倒 + 12 次塌陷，旧代码同代只有 76 + 18。
（-play- 上同代差距小得多：0.6287 → 0.6991；旧代码能到 0.133 是因为它多训了 13 倍。）

#### 2. 为什么摔倒变多：惩罚量级**超了总回报**

iter 1234 的实测（`Episode_Reward/*` 是**每秒速率**）：

| 项 | weight | 实测 rate | 反解出的物理量（只算有零速命令的 env） |
|---|---|---|---|
| `stand_still_vel` | −8.0 | −0.2603/s | `E\|v\|²` ≈ 0.217 ⇒ **\|v\| ≈ 0.47 m/s** |
| `stand_still_wheel_vel` | −0.01 | −0.2230/s | `E[Σω²]` ≈ 149 ⇒ **ω ≈ 6.1 rad/s/轮** |
| **两项合计** | | **−0.483/s** | 而同一步全 batch `Σ Episode_Reward` 只有 **+0.396/s** |

⇒ 惩罚 = 总回报的 **122%**。折算到"静止 env"上约 −3.2/s，是它们正回报的 ~2 倍。
策略于是学到"把轮子彻底冻住"—— 而轮式倒立摆恰恰靠轮子微动平衡 ⇒ 翻倒。
（`is_terminated` 只有 −5 的一次性代价，摊到 1000 步里约 −0.005/s，拦不住这个交换。）

**修正**：按"两项合计 ≈ 总回报的 10~15%"重新定标 ⇒
`stand_still_vel` −0.2 → **−2.0**、`stand_still_wheel_vel` −5e-05 → **−5e-04**
（预估合计约 −0.076/s，~19%）。分工也调整成：**底盘速度项管"用户看到的前向漂移"**，
轮速项只留一个很小的"别空转"信号，不去跟平衡用的轮子微动作对打。

#### 3. 第二版（软化版）验收：**三方同代对比，软化版全面胜出**

第二个 run：`logs/rsl_rl/history_adaptation/2026-09-29_21-38-50_stand_still_soft`
（1000 iter，seed 42，4096 envs，权重 −2.0 / −5e-04；末 checkpoint `model_999.pt`）。
三方都是 **1000 iter / seed 42 / 512 envs / 每档 1100 步**：

| 命令 (vx,vy,wz) | 指标 | 旧代码 @1000 | 第一版 @1000（−8/−0.01） | **软化版 @1000（−2/−5e-4）** |
|---|---|---|---|---|
| (0,0,0) | `err_vel_xy` (m/s) | 0.1030 | 0.0860 | **0.0857**（−17%） |
| (0,0,0) | 摔倒率 | 0.2481 | 0.3401 | **0.1065**（−57%） |
| (0,0,0) | 回合长度 | 890.0 | 845.6 | **920.1** |
| (0,0,0) | 每回合回报 | 40.83 | 38.44 | **46.28** |
| (0.5,0,0) | `err_vel_xy` | **0.1195** | 0.1981 | 0.1625 |
| (0.5,0,0) | 摔倒率 | 0.2443 | 0.2472 | **0.1203** |
| (1.0,0,0) | `err_vel_xy` | **0.1547** | 0.1921 | 0.1856 |
| (1.0,0,0) | 摔倒率 | 0.2713 | 0.1509 | **0.1049** |

读法：**静止漂移降 17%，摔倒率反而降一半以上，回合更长、回报更高** ——
软化版是三项都赢；第一版（重惩罚）是"漂移降了但摔得更狠"。
唯一的小代价：0.5 / 1.0 m/s 两档的速度跟踪误差比旧代码略差（0.16~0.19 vs 0.12~0.15），
1000 iter 还早，需要全长 run 才能判断是不是会被训回来（进 TODO P1-1'）。

顺带回答"权重到底有没有起作用"：**训练日志里的 `Episode_Termination/bad_orientation_2`
在两版之间几乎重合**（iter44 峰值都 0.55、iter175 都 0.33），因为那个指标是所有命令混在一起的
聚合量、且"要求静止"的 env 只占 15%；而**固定命令 eval 把 100% 的 env 都钉在零速命令上**，
于是第一版的坏策略（冻轮子）立刻显形（摔倒 0.3401 vs 软化版 0.1065，而且到 1500 iter 时
第一版继续恶化到 0.7080 ⇒ 它是"越训越会冻"）。

#### 4. 步态对称性（需求 ②）的量化验证

新工具 `scripts/reinforcement_learning/rsl_rl/probe_gait_symmetry.py`（把"撇腿"变成数字）。
旧代码 20k 基线在 -play- 任务、命令 (1.0,0,0) 下：

| 腿 | hipx | hipy | knee | foot x_b | foot y_b |
|---|---|---|---|---|---|
| fl | −0.081 | −0.389 | +1.276 | +0.390 | +0.254 |
| fr | +0.016 | −0.227 | +1.227 | +0.441 | −0.225 |
| hl | −0.031 | +0.444 | −1.347 | −0.385 | +0.230 |
| **hr** | **+0.178** | **+0.769** | **−0.375** | **−0.055** | **−0.289** |

⇒ 和 hl 比，**右后腿的膝几乎是直的（−0.375 vs −1.347）、足端往前 0.33 m、往外多 0.04 m**，
后轮距 0.519 m 比前轮距 0.479 m 宽 —— 就是用户说的"**右后腿往右前方撇**"。
镜像误差 RMS（rad）也给出同一结论：`hl~hr` hipy 0.454 / knee 1.127（knee 那一项接近两者差值）。

同代（1000 iter）三方对照（训练任务 / 命令 (1.0,0,0) / 256 envs / 400 步，`probe_gait_symmetry.py`）：

| 指标 | 旧代码 @1000 | 第一版 @1000 | **软化版 @1000** |
|---|---|---|---|
| 后腿左右不对称 `y_hl+y_hr` (m) | −0.0702 | +0.0287 | **−0.0099**（改善 86%） |
| 前/后轮距 (m) | 0.503 / 0.462（差 0.041） | 0.415 / 0.476 | **0.459 / 0.458**（差 0.001） |
| `hl~hr` 镜像 RMS（hipx/hipy/knee） | 0.252/0.154/0.522 | 0.316/0.207/0.487 | **0.172/0.115/0.325** |
| `fr~hl` 镜像 RMS（hipx/hipy/knee） | 0.172/0.104/0.464 | 0.149/0.224/0.365 | **0.108/0.105/0.220** |
| hl / hr 膝 (rad) | −1.144 / −0.869 | −1.201 / −1.301 | **−1.161 / −1.192** |

⇒ 软化版：**后腿左右不对称从 7.0 cm 降到 1.0 cm、前后轮距差从 4.1 cm 降到 0.1 cm、
4 个镜像对的 RMS 全线下滑**（`hl~hr` knee 0.522→0.325、`fr~hl` knee 0.464→0.220）。
即"右后腿往右前方撇"这条被量化确认、并且在符号修正后**基本消失**。
（注：`fl~hr` 的 knee 项软化版略高于旧代码 0.307 vs 0.286，属同量级抖动；其余项都更好。）

#### 5. 云端 20k 主线的**中途验收（iter = 10000）**—— 2026-09-30

run `logs/rsl_rl/history_adaptation/2026-09-30_00-09-25_cloud_soft20k`（就是 DEF-032 里
`bbc64d91a6-99f1820e` 那条 20k 主线；2026-09-30 上午已把整个 run 目录（含 model_10000.pt
与 tensorboard 事件）`scp` 回本机）。跑完 20k 前的**中途读数**：

**a) 固定命令 eval（-play- 任务 / 512 envs / seed 42 / 每档 1100 步）**

| 命令 (vx,vy,wz) | 指标 | 旧代码 @10000 | **云端软化版 @10000** | 旧代码 @20000 |
|---|---|---|---|---|
| (0,0,0) | `err_vel_xy` (m/s) | 0.1390 | **0.1177**（−15%） | 0.1478 |
| (0,0,0) | 摔倒率 | 0.0907 | 0.1071 | 0.1332 |
| (0,0,0) | 回合长度 | 942.8 | 913.1 | 894.5 |
| (0.5,0,0) | `err_vel_xy` | 0.1526 | **0.1332**（−13%） | 0.1583 |
| (0.5,0,0) | 摔倒率 | 0.1051 | **0.0917** | 0.0901 |
| (1.0,0,0) | `err_vel_xy` | 0.1945 | **0.1871**（−4%） | 0.1920 |
| (1.0,0,0) | 摔倒率 | **0.0814** | 0.1214 | 0.1055 |

读法：**三档命令的"静止/巡航速度误差"全部优于同代旧代码**（−4%~−15%），
摔倒率在 0.5 m/s 更好、在 0/1.0 m/s 略差（每档约 530 个回合 ⇒ 摔倒率的标准误 ~1.3%，
所以 0.09 vs 0.11 只能算边缘差异，0.081 vs 0.121 是显著的）。

**b) 训练侧（同迭代对比，用 `summarize_run.py --grid`）**

| iter | 指标 | 旧代码 | **云端软化版** |
|---|---|---|---|
| 10000 | `Train/mean_reward` | 21.32 | 14.70（其中新增两项惩罚约 −0.17/s 是"口径差"，见 §2） |
| 10000 | `mean_episode_length` | 934.9 | 799.1 |
| 10000 | `Episode_Termination/bad_orientation_2` | 0.0118 | **0.0084** |
| 10000 | `Episode_Termination/root_height_below_minimum` | 0.0876 | **0.280** ⚠️ |
| 10000 | `Metrics/base_velocity/error_vel_xy` | 0.879 | **0.789** |
| 10000 | `Episode_Reward/joint_mirror` | −0.162 | **−0.024** |

⚠️ **要注意的一条**：训练期的 `root_height_below_minimum`（塌陷类终止）明显高于旧代码
（s3 段均值 0.198 vs 0.118）。但**固定命令 eval 里同一项反而是好的**
（命令 (0,0,0)：`root_height` 触发 55/551 = 10.0%，旧 20k 是 74/557 = 13.3%）。
两点原因：① 训练期是**混合命令 + 加强后的 push（±2 m/s / 5~10 s）**，被推趴的比例天然更高；
② 站姿占比 0.15 让"零速命令"的样本多了 7 倍，"站着不动"本身比"慢慢滚"更容易触发高度下限。
⇒ **判据一律看固定命令 eval，不看训练期的聚合终止率**（这是 §2 已经踩过一次的坑）。

**c) 步态对称性（-play- / 命令 (1.0,0,0)，`probe_gait_symmetry.py`）**

| 指标 | 旧代码 @20000 | **云端软化版 @10000** |
|---|---|---|
| hr / hl 膝 (rad) | **−0.375 / −1.347**（右后腿膝几乎不弯） | **−1.560 / −1.505**（对称） |
| hr 足端 body 系 x | −0.055（比 hl 前 0.33 m） | −0.370（hl −0.367） |
| 后腿左右不对称 `y_hl+y_hr` | −0.0591 m | **+0.0076 m** |
| 前后轮距 (m) | 0.479 / 0.519（差 4.0 cm） | **0.427 / 0.444**（差 1.7 cm） |
| `hl~hr` 镜像 RMS（hipx/hipy/knee） | 0.299/0.454/1.127 | **0.326/0.143/0.380** |
| `fl~hr` 镜像 RMS knee | 1.238 | **0.762** |

⇒ **"右后腿往右前方撇"在 10000 iter 的策略上已经基本消失**（后腿膝角对称、足端对称、
镜像 RMS 的 hipy/knee 降 66~68%）。

> 结论（中途）：**静止漂移与步态对称两条需求在 10k 上就已经达标**，稳定性是"各有胜负"；
> 20k 跑完（约 2026-09-30 19:30）再复核一次并定稿。

## 八、本机收尾批（2026-09-30，分支 `codex/ll-train-detail-fix`）

这一批**不需要训练**、全部在本机验证；来龙去脉见 `DEFECT_LOG_zh.md` **DEF-034**。

| 项 | 内容 | 验收 |
|---|---|---|
| **回归矩阵脚本化**（TODO P1-4 + P3） | 新增 `scripts/reinforcement_learning/rsl_rl/smoke_regression.py`：逐任务独立进程 + 日志落 `logs/smoke/<日期>_<task>.log` + **"日志 N 秒无增长 ⇒ SKIP 并杀进程树"** + Markdown 汇总 + FAIL 才非零退出 | **13 任务：11 OK / 1 SKIP / 0 FAIL**（SKIP = 生成地形在本机死锁，DEF-031，157s 被杀）；完整表 `logs/smoke/2026-09-30_regression.md` |
| **known_issues ⑫ 修复**（EE 命令 reset 首帧） | `HeightInvariantEECommand.reset()` 覆写：`super().reset()` 后补"命令 ← 插值起点" | 新探针 `probe_ee_command_init.py`：修前 `env.reset()` 后命令 = 父类初值、与真实 EE 位姿差 **0.4327 m**；修后 `pose_command_b = (0.3492, 0, 0.4327, …)`、差 **0.000e+00** |
| **known_issues ⑪ 澄清** | 父类 `_update_command()` 本来就是 `pass`；`pose_command_w` 在 `_update_metrics()` 里每步更新（只比 `pose_command_b` 晚一拍，实测 6e-3~1.7e-2） | 探针实测 ⇒ "漏调 super() 导致 `pose_command_w` 永不更新"**不成立** |
| **known_issues ⑧ 删埋雷** | 删除 `action_mirror` / `action_sync`（两 term + 两函数）：用 articulation 关节 id 索引动作向量 + 关节名是 Go1 风格（本机型不存在）⇒ 打开就炸；状态层 `joint_mirror_signed` 已覆盖 | 回归 11 任务 EXIT=0 |
| **known_issues ⑩** | `grasp_success` / `ee_approach_object` 加 `_require_scene_entity`：明确报"需要 object 实体 / 当前场景有哪些 / 本奖励是给高层用的" | 未被任何任务引用，回归不受影响 |
| **known_issues ⑱** | 启动期布局自检增加接触传感器行序打印（body 数、是否与 articulation 同序、`名字#行号` 前 6 个）+ "传感器 body 名能否在 articulation 里找到"检查 | 回归日志里可见该打印 |
| **工程债 ×3** | `setup.py` → `find_packages`；`encoder.py` 的 frozen 缓存 key → `(name, device)`；`mp4-png-composition.py` 4 处裸 `except:` → `except Exception:` | `py_compile` 通过；回归（前两项不涉及训练路径） |

---

## 九、P2：高层 action term 收进一个基类（2026-09-30，分支 `codex/ll-train-detail-fix`）

用户要求"P2 里尽量不要那么多重复的 action term，能实现为一个基类最好"。来龙去脉见
`DEFECT_LOG_zh.md` **DEF-035**。

| 项 | 内容 | 验收 |
|---|---|---|
| **3 个 action term 收敛** | `PreTrainedPickAction` / `PreTrainedPickWBCAction` / `TeleopLLAction` 从 `ActionTerm` 改为继承 `LowLevelPolicyActionBase`：`__init__` 只留"`_raw_actions` 分配（`super()` 之前）→ `super().__init__` → 任务专属状态"，`apply_actions` 全部删除，改用基类的 `_on_low_level_tick()` / `_on_reset(env_ids)` / `_extra_cache_tensors()` | 回归矩阵 **11 OK / 1 SKIP / 0 FAIL**（`logs/smoke/2026-09-30_regression_p2.md`） |
| **删除 openvla 时代遗留** | `mdp/pre_trained_policy_action.py`（371 行，无人注册，只在一段 `#` 注释里被提到）+ `mdp/__init__.py` 的 star-import + `high_level_env_cfg.py` 的两处注释残留 | 全局 `grep`：仓库内已无 `PreTrainedPolicyAction` 引用 |
| **"低层布局打印逐字节一致"** | 用改动前后两批 `logs/smoke/*.log` 只比 `[ll-replay:*]` 行（**不改任何 cfg，只动代码结构**，所以这是最直接的行为中性证据） | Pick-Flat **83(83)**/action 23、Pick-WBC **76(76)**/16、Teleop **76(76)**/16、Teleop-History **83(83)** + history 10×70、Nav-Flat 76(76) —— **5/5 完全一致** |
| **复位钩子时机验证** | 新增 `scripts/reinforcement_learning/rsl_rl/probe_reset_anchor_timing.py`，把 `_on_reset` / `_reset_target_to_current_ee` / `_capture_default_ee_pose` / `recalibrate` 包起来打印 `episode_length_buf / root_z / ee_z` | 复位前 `root_z=0.5371`、`_on_reset` 里已是 **0.5500**（复位位姿）⇒ 钩子读到的是复位后状态；两个 episode 重锚出的 `target_ee_pos_b` 一致到 **1e-7** |

> 唯一的（有利的）行为差异：Pick/WBC 的"重锚 EE 目标"从复位后**第 2 个 env step**
> 提前到**第 1 个**（旧实现在 `apply_actions` 用 `episode_length_buf == 0` 检测，晚一步），
> 语义更贴合"把目标锚在 episode 起点的位姿"。

---

## 十、P3：EE 锚点对照一键化（2026-09-30，分支 `codex/ll-train-detail-fix`）

**"EE 锚点"是什么**：训练早期（课程 s0）把机械臂的目标位姿**锁死在一个固定点**上，让机械臂先
别乱动、底盘专心学平衡，之后再逐步放开到完整工作空间（课程 s1→s3）。这个"锁死的固定点"就是
锚点。来龙去脉见 `DEFECT_LOG_zh.md` **DEF-036**。

| 项 | 内容 | 验收 |
|---|---|---|
| **一键脚本** | 新增 `scripts/reinforcement_learning/rsl_rl/sweep_ee_anchor.py`：默认 4 组锚点（`full` = 课程 s3 全分布 / `default` = 举臂 / `low` = 低位前伸 / `cfg` = 不改 cfg），逐组起**独立 Isaac 子进程**（一个进程只能建一个 env）、收 JSON、打 Markdown 对比表 + 汇总 JSON，任一失败即非零退出 | 4/4 OK（EXIT=0），汇总 `logs/smoke/2026-09-30_15-33-06_ee_anchor_sweep.json` |
| **修掉语义歧义** | 旧探针 `probe_root_height_termination.py --freeze_ee_preset none` 的 "none = 不改 cfg" 在当前代码里**就是 low 锚点**（`FlatEnvWBCConfig.__post_init__` 已把 `p_l` 锁成 0.41）⇒ 原来 `none` 与 `low` 必然逐位相同；现补 `full`（真正的"无课程"）+ 默认值改 `cfg` + 加 `ranges:`/`--json_out` 打印 | `cfg` 与 `low` 两组**逐位相同** ⇒ 机器验证"cfg 默认 == low 锚点" |
| **回归脚本透传** | `smoke_regression.py` 新增 `--env KEY=VALUE`（高层四条换低层 checkpoint 用 `RL_TRAINING_LOW_LEVEL_POLICY_*`）与 `--hydra OVERRIDE`，都可重复 | 负向用例（注入不存在的 path）子进程日志里出现该 path + `FileNotFoundError`、脚本 FAIL 且 EXIT=1；正向用例 2/2 OK |

实测（本机，`512 envs × 1000 steps` = 20 s 窗口，策略 = 旧 20k 部署态）：

| 锚点 | 说明 | root_z p05 (m) | height_error 均值 (m) | 20 s 内 `root_z<0.30` | 倾角>45.8° |
|---|---|---|---|---|---|
| `full` | 全任务分布（= 无课程） | 0.3582 | +0.0159 | **7.0%** | 0.0% |
| `default` | 锁默认姿态（举臂） | 0.3615 | +0.0124 | **1.8%** | 0.0% |
| `low` | 锁低位前伸 | 0.3607 | +0.0114 | **0.8%** | 0.2% |
| `cfg` | 不改 cfg（= 当前默认） | 0.3607 | +0.0114 | **0.8%** | 0.2% |

⇒ 排序与 DEF-006 一致（`full` > `default` > `low`），**课程继续用 `low` 锚点是对的**。
绝对值比 DEF-006 低一大截是因为这次用的是**已训好的 20k 策略**（DEF-006 是 2026-09-19 的早期策略）
⇒ 两组数字不可横向比较，只能比组内排序。

---

## 一、低层训练（本轮主线）

| 日期 | 内容 | 关键实测 | commit |
|---|---|---|---|
| 2026-09-22 | **P1-1 修复：探索噪声上界 `max_noise_std=1.2`**（A/B 实测通过 ⇒ 建议作为低层训练默认；同时证明"噪声压太狠反而更不稳"） | 4000 iter / seed 42 / 4096 envs，**统一窗口 iter 3125–3999**：`mean_noise_std` 1.405→**1.052**、`mean_reward` 23.93→**38.52**、`mean_episode_length` 858→**905**、s3 合计摔倒 0.194→**0.132**；对照 `entropy_coef=0.002`：36.92 / 878 / 0.183，`entropy_coef=0`：34.36 / 857 / 0.222 | 开关 + 冒烟 `b75c596`；回填见 `DEFECT_LOG_zh.md` DEF-024 §4 |
| 2026-09-20 | 低层内容并入 `main`（训练配置 + EE 课程 + slerp + root_height 专项 + known_issues ⑤⑥⑦⑯ + 导出脚本 + 训练说明） | 4 个低层任务 `--num_envs 64 --max_iterations 2` 全 EXIT=0；`env.yaml` 里 `ee_goal_stages`/`disturbance_ramp`/`steady_error_clip=0.15`/`limit_angle=0.8` 均生效 | `26584e9`(merge) `469fbd4` `fef34a7` `7ff5b86` |
| 2026-09-20 | `bad_orientation_2` 改成旋转不变量 + 阈值 0.8 rad；**保留** `ee_goal`（policy 76→83） | History-Adaptation 2 iter exit 0；`policy 83 / history 700 / privileged 89` | `45f9e74` / `26584e9` |
| 2026-09-19 | **EE 目标课程**：s0 锚点 + s1/s2/s3 区间阶梯（`mdp.apply_range_stages`）；EE 姿态命令改 slerp | 课程 4 阶段自检全过；slerp 与官方单样本最大分量偏差 1.19e-07；姿态单步跳变 155.7°→3.12° | `469fbd4`（原 `9ccb8ec`/`96e1b66`） |
| 2026-09-20 | **root_height 专项**：s0 锚点从"默认（举起）位姿"改成**低位锚点**、`body_pose.height_range` 上界 0.60→0.55、扰动课程（push/外力 30%→100%，25k 步）、新增 `height_error_bias_steady` | 机制对照：锁低位 **1.0%** vs 锁默认位姿 **55.5%** vs 无课程 **25.8%**（20s 高度终止率）；阈值反事实证明"单独降阈值无效"（0.30→0.26 只 25.8%→24.4%） | `7ff5b86`（原 `96e1b66`） |
| 2026-09-20 | **训练结果**：run `2026-09-20_00-50-31`（4096 envs，15k iter，本分支代码） | `root_height_below_minimum` 0.361→**0.122**、合计摔倒 0.371→**0.132**、ep_len 805→**883**、reward 15.4→**22.7**、`height_error_bias_steady` 1.1 cm（iter=14000 对比旧 run） | 代码 `96e1b66` |
| 2026-09-20 | 部署态导出（`model_15000.pt`）—— actor-only 陷阱已避开 | `exported_deploy/{policy.pt,policy_layout.json}`，自检 scripted↔eager 与 `act_inference` 均 **0.000e+00**；layout `history/83/10×70/latent32/action16` | 脚本 `4276970` |
| 2026-09-19 | 低层 `known_issues` ⑤⑥⑦ + ⑯ 剩余：占位符 → `None`、`disable_zero_weight_rewards` 对 None 容错 + `term_names`、`stance_width` 改数值、启动期布局打印/断言（`mdp.check_policy_layout`）、`joint_pos_rel_without_wheel` 列序断言 | Arm/WBC/History 三任务 2 iter EXIT=0；启动打印 `policy 86 = … + actions22`、`history 700` | `fef34a7`（原 `6d22006`） |
| 2026-09-18 | 手臂奖励坐标系修复（root 系统一）+ EE body 索引缓存；privileged 观测缓存 reset 感知；手臂隔离 env | 见 `DEFECT_LOG_zh.md` DEF-017 | `b3496a5` / `905c2df` |

## 二、高层 replay（2026-09-20 起**已并入 main**，合并过程见第五节）

| 日期 | 内容 | 关键实测 | commit |
|---|---|---|---|
| 2026-09-19 | **①** `PreTrainedPickAction` 补 `ll_command`/`ll_command_w`；8 处世界系奖励项改用 `ll_command_world()` | 三任务 2 iter EXIT=0、reward 与改前逐位一致；坐标系与独立复算 0.000e+00 | `e064bc6` |
| 2026-09-19 | **②** IK 目标写 `pose_command_b`（并同步 `pose_start_b/pose_end_b`）；**③** flat 的 `ee_goal` 改 root 系 | `\|pose_command_b − 高层目标\| = 0.000e+00`；三任务 2 iter EXIT=0 | `7a22759` |
| 2026-09-19 | **清单外 A**：`actions` 观测少 7 维（IK 改 CommandManager 驱动后）→ 按布局拼接；**清单外 B**：轮关节掩码按**列**置零 | flat 83=83、WBC/teleop 86=86；正确列下标 `[12,13,14,15]` | `dc45d0e` |
| 2026-09-19 | **④** 低层动作 scale 不再硬编码（实测原写法是**死代码**：`JointAction.__init__` 已把 scale 编译成内部张量）；**⑥** 观测组模板 deepcopy，不再就地改 cfg；**⑨** `ee_pose_commands` → `ee_goal` | 三任务 2 iter EXIT=0 | `dc45d0e` |
| 2026-09-19 | **L2 布局推导**（`ee_action_dim=-1` 默认从低层 cfg 推导）+ 按 `policy_layout.json` 匹配 obs 维度（`ee_goal` 自动取舍） | teleop 83=83、WBC 83=83、flat(L1) 83=83 | `1f56b6e` |
| 2026-09-19 | **history 低层策略回放支持**：10 步窗口（复用 IsaacLab `CircularBuffer`）+ 单/双输入调用 + `last_action` 用低层 16 维动作 + 复位判定改"跳变检测" | 单步 vs 训练函数独立复算 `0.000e+00`；整窗顺序 40 次 tick `0.000e+00`；teleop/Pick-WBC 用 history 策略 2 iter EXIT=0 | `0d37c99` |
| 2026-09-19 | **⑦** checkpoint 路径参数化（环境变量 `RL_TRAINING_LOW_LEVEL_POLICY_<KEY>` / hydra）+ 统一加载报错；**⑧** `LOW_LEVEL_ENV_CFG` 改懒加载 + `render_interval` 修成不重复渲染 | 无效路径给出三种修法、有效覆盖 83=83；`Rendering step-size` 0.02→0.2、警告消失、reward 不回归 | `1b6d5c8` |
| 2026-09-19 | **⑤ 的 R1**：抽 `LowLevelPolicyActionBase`（载入/布局/观测/history/tick/`ll_command` 单一来源）+ 迁移 nav；顺带修 nav 奖励项读不存在的 `ll_command` | nav 2 iter EXIT=0、`低层 obs 69 = checkpoint 69`；`lateral_velocity_penalty` 从 AttributeError 变成 -0.0750/-0.0275 | `2c5a85a` |
| 2026-09-19 | 探针工具：`probe_ee_default_pose.py`、`probe_ee_curriculum.py`、`probe_history_window.py`、`probe_root_height_termination.py` | 见各自 commit | `9ccb8ec` / `0d37c99` / `96e1b66` |

## 三、更早的修复（来自原 `known_issues.md` 一、1-4/16/17 与三、）

| 日期 | 内容 | 关键实测 | commit |
|---|---|---|---|
| 2026-09-18 | `joint_pos_rel_without_wheel` 索引空间混用（清错关节）→ policy 的 `joint_pos/joint_vel` 回 `[".*"]` 原生序（24 维） | 被清掉的关节从 `hr_wheel + arm_joint1/2/3` 纠正为 `fl/fr/hl/hr_wheel` | `b3496a5`（另见 replay 侧 `dc45d0e`、断言 `6d22006`） |
| 2026-09-18 | privileged 观测缓存对 reset 模式随机化不敏感 → 抽出 `_PrivilegedCachedTerm`（`update_on_reset` 开关）；顺带发现 `ObsTerm.params` 会原样透传 | @4096 envs 每次 reset 净增 1.92 ms（摊销 0.0019 ms/step），缓存路径不变 | `905c2df` |
| 2026-09-18 | 手臂奖励坐标系不一致（root vs world）→ 统一 root 系 + EE body 索引缓存 | 旧算法把奖励算成 0.0000，正确值 0.0010 | `b3496a5` |
| 2026-09-18 | legacy 手臂奖励权重不可用（零动作稳态 `Στ²≈2.4e4`）→ `torque=-1e-5 / vel=-1e-3 / acc=-1e-8` | 总奖励从 -86.2 回到 +0.53 | `b3496a5` |
| 2026-09-18 | `[已确认：不是 bug]`"臂/夹爪持续接触 90 N"是归因错误（sensor 与 articulation body 顺序不同） | 按 `sensor.body_names` 归因：轮子 85~115 N（地面支撑），臂/夹爪 1.3~4.1 N | `1fb76e4` |

## 四、基础设施

* `export_deploy_policy.py`：把带 history encoder 的策略导出成部署态
  `forward(policy_obs, history_flat)` + `policy_layout.json`（纯 torch，不起 Isaac）。
* `low_level_replay.py`：低层 replay 的单一来源（关节分组/布局/观测组装/启动期断言/
  history 窗口/`ll_command` 辅助）。
* `mdp.check_policy_layout`（低层 startup 事件）：打印观测/动作布局并断言
  "`actions` 槽位宽度 == 动作总维度"。
* `summarize_run.py`（2026-09-20 新增）：把 run 的 tensorboard 标量压成
  "关键指标 × 课程阶段 / × 迭代采样"表，支持两 run 对齐对比；`--derive` 可把终止项
  求和（"合计摔倒"）；首次解析 70 MB 事件文件 30~50 s，之后走 `<run>/.summary_cache.npz`。
  用它做完了 P1-1 归因与 P1-2 证据（`DEFECT_LOG_zh.md` DEF-023）。
* `probe_gait_symmetry.py`（2026-09-29 新增）：把"撇腿/步态不对称"变成**可比的数字** ——
  固定命令滚动若干步，输出 ① 每腿 hipx/hipy/knee 均值 ② 4 个镜像对的误差 RMS（rad）
  ③ 4 个轮的 body 系 (x, y) 与两个左右不对称度（`y_fl+y_fr`、`y_hl+y_hr`）。
  用它量化了"右后腿往右前方撇"（`DONE_zh.md` 第七节 §4）。
* `eval_fixed_command.py`（2026-09-22 落库）：把 vx/vy/wz 钉死后滚动，给出**同命令**下的
  回合回报 / 速度误差 / 终止构成；新增的静止专项验收就靠它（`Train/mean_reward` 带命令课程、
  跨 run 不可比）。

## 五、高层链并入 `main`（P0-1）+ 导出流程固化（P0-2）

| 日期 | 内容 | 关键实测 | commit |
|---|---|---|---|
| 2026-09-20 | **P0-1 高层 replay 链合并进 `main`**（replay 布局：`actions` 观测少 7 维 + 轮关节掩码 → L2 布局推导 → ① `ll_command` → ② IK 目标写 `pose_command_b` + ③ `ee_goal` 用 root 系 → history 回放 → ⑦ checkpoint 参数化 + ⑧ 懒加载低层 cfg → ⑤ 的 R1 抽 `LowLevelPolicyActionBase` + 迁移 nav） | 4 个高层任务 `--headless --num_envs 64 --max_iterations 2` **全 EXIT=0**：Pick-Flat-Teacher reward **1.11**、Pick-WBC-Flat **1.28**、Teleop **0.15**、Nav-Flat-Teacher **10.25**；启动打印 "低层 obs 维度 == checkpoint 期望"（83/76/76/69）、动作分块 `leg12+wheel4+ee_ik{0,7}`；4 份日志均无 Traceback | `129848e`（第一步）、`af4602d`（⑦⑧）、`07601e9`（R1） |
| 2026-09-20 | 合并冲突按 TODO 约定解决：R1 保留基类写法，基类里的内联加载换成 ⑦ 的 `load_low_level_policy(...)`，并给 `verify_low_level_layout` 补传 `policy_layout_json` | 见 `DEFECT_LOG_zh.md` DEF-018 | `07601e9` |
| 2026-09-20 | 删掉高层分支带来的旧 `docs/review/*.md`（其中 `next_session_prompt.md` 与 main 的 `NEXT_SESSION_PROMPT.md` 只差大小写） | `docs/review/` 现在只剩 TODO/DONE/DEFECT_LOG/NEXT_SESSION_PROMPT + `templates/`；`git status` 干净 | `30d5411`、`0772757` |
| 2026-09-20 | **P0-2 导出流程固化**：`export_deploy_policy.py` 默认输出目录 `<run>/exported` → **`<run>/exported_deploy`**（并把 actor-only 陷阱写成显式警告）；`train.py` 训练结束打印可直接复制的导出命令；`docs/train_history_flat_zh.md` 补"训练完成后"章节 + 修正"高层 replay 还不支持 history"的过时说明 | `2026-09-20_00-50-31/model_15000.pt` 实测导出 `exported_deploy/{policy.pt 1001636 B, policy_layout.json}`：scripted vs eager **0.000e+00**、与 `ActorCriticHistory.act_inference` 交叉校验 **0.000e+00**、`kind=history / policy_obs_dim=83 / 10×70 / latent32 / action16` | `498e847` |
| 2026-09-20 | **ONNX 导出**：`--onnx/--no-onnx`（默认开）+ `--opset`（默认 17）；自检 = `onnx.checker` + onnxruntime↔TorchScript **相对**误差 + batch 1/5 动态维验证；`policy_layout.json` 增补 `onnx` 段与 `history_order/history_note`（见 DEF-020） | run `2026-09-20_00-50-31` **重新导出最新 checkpoint `model_19999.pt`**：`exported_deploy/{policy.pt, policy.onnx 981 KB, policy_layout.json}`；相对误差 **1.87e-07**（绝对 3.43e-05 / 幅值 183.6）；独立复核 B=1/3/8 相对误差 1.5e-07~2.7e-07；图 = opset17、`policy_obs['batch',83]`+`history_flat['batch',700]` → `action['batch',16]` | `708ca53` |
| 2026-09-20 | **部署交接**：`probe_deploy_layout.py`（一次性打印关节序/默认角/限位/动作增益/观测逐项 scale·clip·noise/关键 body/默认姿态几何）+ `docs/deploy_sim2sim_sim2real_zh.md`（接口契约、IK 复刻、MuJoCo 参数、七步上线顺序、失败模式表） | 探针实测：**动作增益 hipx 0.125 / 其余腿 0.25 / 轮速 5.0**；原生关节序 wheel=15..18；观测 policy **83**/critic 86/history **700**/privileged 89；MuJoCo 默认姿态 `gripper_base` 相对 base `(0.3492,0,0.4326)` vs Isaac `(0.3492,0,0.4327)` | `7458672` |

## 六、部署基线（deploy baseline）

**当前拿去部署的基线 = `main @ 2d49f47`**（`git rev-list --left-right --count origin/main...main` = `0 0`，
即与 `origin/main` 完全一致、已推送）。基线一旦记录就不再"漂"：后续代码/文档提交只在
`main` 上往前走，**要部署就 checkout 这个 commit（或用 tag）**，不要用"当时的 main"。

> 已打 **annotated tag `deploy-baseline-2026-09-20` → `2d49f47`**（已 push）。
> 部署机取代码：`git fetch --tags && git checkout deploy-baseline-2026-09-20`。

| 项 | 值 / 位置 |
|---|---|
| 部署口径代码 | `main @ 2d49f47`（含低层 root_height 专项 + 高层链 + 导出/部署工具链） |
| 对应训练 run | `logs/rsl_rl/history_adaptation/2026-09-20_00-50-31`（4096 envs，iter 0 → 19999） |
| checkpoint | `<run>/model_19999.pt` |
| 部署态策略 | `<run>/exported_deploy/{policy.pt, policy.onnx, policy_layout.json}` |
| 接口契约 | `policy_obs 83` + `history_flat 700`（10×70，**最旧→最新**）→ `action 16`；ONNX opset 17，与 TorchScript 相对误差 **1.87e-07** |

导出物指纹（`Get-FileHash -Algorithm SHA256`，2026-09-20 记录；换 checkpoint/改环境
重导出后**必须**重新记录）：

| 文件 | 字节 | sha256 |
|---|---|---|
| `exported_deploy/policy.pt` | 1001636 | `43C63D19…4522F6510` |
| `exported_deploy/policy.onnx` | 981002 | `77757542…B2FF53B8D1` |
| `exported_deploy/policy_layout.json` | 1640 | `7E3B11EB…B9E6A7947` |
| `model_19999.pt`（源头） | — | `592A50D6…89A87AC14` |

### 训练代码 vs 部署代码（这条必须能回溯）

run 目录里的 `<run>/git/loco-manip-unified-rl-agent.diff`（rsl_rl 训练启动时自动 dump）显示：
该 run 训练时在分支 `codex/ll-height-stability @ 96e1b66`，工作区唯一改动是
**注释掉 `FKReachableEECommand`**（该类没有任何 task/配置引用，`rough_env_cfg.py` 里只有一行
注释指向它 ⇒ 死代码，注释掉不影响训练）。

`main` 与它在这条低层链上的差异（`git diff 96e1b66 main -- source/rl_training/.../velocity ...`）
只有以下四类，**没有动力学 / 观测布局变化**：

1. `SceneEntityCfg(body_names="")` / `joint_names=""` → `None` 的占位符清理（weight=0 的死配置）；
2. `stance_width=float`（类型对象）→ 数值（同样是 weight=0）；
3. 新增启动期自检事件 `mdp.check_policy_layout`（只打印 + 断言，不改动力学）；
4. `observations.py` 的索引空间断言与文档注释、`deeprobotics.py` 删掉一段标定注释、
   `flat_env_wbc_cfg.py` 里旧文档路径改指 `DEFECT_LOG_zh.md`。

⇒ **结论**：在当前口径下"main 的代码 = 训出这个策略的代码"成立。要复现旧 checkpoint 时
只需注意别改 `ee_ik` 这类 action 维度（DEF-013：布局一变旧 checkpoint 静默失效）。
详见 `DEFECT_LOG_zh.md` DEF-022。

### 基线可运行性验收（2026-09-20，8 任务冒烟回归）

冻结基线前把 TODO P3 的"回归矩阵"整体跑了一遍（每个任务单独进程，
`python scripts/reinforcement_learning/rsl_rl/train.py --task <task> --headless --num_envs 64 --max_iterations 2`，
日志在 `logs/smoke/2026-09-20_<task>.log`）—— **8/8 EXIT=0，无 Traceback**：

| 任务 | EXIT | 关键打印 |
|---|---|---|
| `History-Adaptation-Deeprobotics-M20-v0` | 0 | 启动自检 `policy 83 / critic 86 / history 700 / privileged 89`；reward 0.02 / -0.01 |
| `Flat-Deeprobotics-M20-Piper-WBC-v0` | 0 | — |
| `Flat-Deeprobotics-M20-Piper-v0` | 0 | — |
| `Flat-Deeprobotics-M20-Piper-Arm-v0` | 0 | — |
| `Isaac-Deeprobotics-High-Level-Pick-Flat-Teacher-v0` | 0 | `PreTrainedPickAction: action_dim=23, 低层 obs 83（期望 83）`；reward **1.11** |
| `Isaac-Deeprobotics-High-Level-Pick-WBC-Flat-Teacher-v0` | 0 | `PreTrainedPickWBCAction: 16 / 76（期望 76）`；reward **1.28** |
| `Isaac-M20-Piper-Teleop-v0` | 0 | `TeleopLLAction: 16 / 76（期望 76）`；reward **0.15** |
| `Isaac-Deeprobotics-High-Level-Nav-Flat-Teacher-v0` | 0 | `PreTrainedNavAction: 16 / 69（期望 69）`；reward **10.25** |

四个高层 reward 与合并高层链时的记录（1.11 / 1.28 / 0.15 / 10.25）逐位一致 ⇒ 基线没有回归。
日志里只有 Isaac 自带的 `failed to open .../kit/.../user.config.json` 警告（只读安装目录，无害）。
