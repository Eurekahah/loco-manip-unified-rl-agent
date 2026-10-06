# 总待办清单（TODO）—— 按优先级

**文档职责**：这是**唯一**的"未完成任务"清单。已完成的事情在 `DONE_zh.md`，
每个缺陷/特性的来龙去脉在 `DEFECT_LOG_zh.md`。旧的分主题清单
（`high_level_todo.md` / `known_issues.md` / `history_low_level_policy_todo.md` /
`bad_orientation_analysis_zh.md` / `progress_summary_zh.md` / `todo_master_zh.md`）
已并入这两份文档，原文可在分支上查到（见文末"旧文档去哪了"）。

**维护约定**：见 `templates/DOC_TEMPLATE_zh.md`。每次更新在下面"更新记录"加一行；
条目用 `- [ ]`/`- [x]`；完成即迁到 `DONE_zh.md`。

## 更新记录

| 日期 | 更新内容 | 相关 commit / 分支 |
|---|---|---|
| 2026-09-20 | 初版：把 6 份旧清单合并成 TODO/DONE/DEFECT_LOG 三份；低层内容并入 `main`，P0 变成"高层链合并 + 导出流程固化" | `main @ 7ff5b86` |
| 2026-09-20 | **P0 清空**：高层链合并进 `main`（3 个 merge commit）+ 导出部署态策略固化（脚本默认目录/训练收尾提示/训练说明）；新增 DEF-018/019；P3"文档收尾"整条完成（旧 `docs/review/*.md` 已删 + 全部悬空引用改指新文档） | `codex/hl-merge-p0`（`129848e`/`af4602d`/`07601e9`/`30d5411`/`0772757`） |
| 2026-09-20 | 导出增加 **ONNX**（`--onnx/--opset`，含 onnxruntime 自检；DEF-020）；run `2026-09-20_00-50-31` 用最新 `model_19999.pt` 重新导出 | `708ca53` |
| 2026-09-20 | 新增**部署交接**：`docs/deploy_sim2sim_sim2real_zh.md` + `probe_deploy_layout.py`（DEF-021）；P1/P2/P3 待办不变 | `7458672` |
| 2026-09-20 | **部署基线固化**（`main @ 2d49f47` ↔ run `2026-09-20_00-50-31`，含产物 sha256 + tag `deploy-baseline-2026-09-20`，DONE 第六节 / DEF-022）；新增 `summarize_run.py`（P3 提前做，DEF-023）；**P1-1 改为"口径 + 配置"两条修法**、P1-2 补臂相关证据 | 基线 `2d49f47`；工具 `dc3a5b9`；文档 `eb22401` |
| 2026-09-20 | P1-1 进入实测：新增**探索噪声上界** `max_noise_std`（默认 0 = 不限制，投影梯度实现，DEF-024），并启动 4000-iter 的 cap=1.2 / entropy_coef=0.002 A/B（结果待回填） | `docs: P1-1 A/B` 提交 |
| 2026-09-29 | 新分支 `codex/ll-train-detail-fix`：**静止伫立专项**（DEF-026，两项惩罚 + 三条课程）、**镜像符号修复**（DEF-027，"右后腿往右前方撇"）、**扰动加强**（DEF-028）、**多地形任务**（DEF-029）、**遥操 history 任务**（DEF-030）；新增 P1-1'~P1-4 四条待办 | `codex/ll-train-detail-fix` |
| 2026-09-30 | **本机收尾批（DEF-034）**：回归矩阵脚本化（P1-4/P3 完成，11 OK / 1 SKIP / 0 FAIL）+ known_issues ⑧⑩⑪⑫⑱ + 三处工程债；11 条能跑的任务全部 EXIT=0 | `codex/ll-train-detail-fix` |
| 2026-09-30 | 第二批（按用户答复）：**cusrl 全删**（9 处注册字段 + setup 依赖）、**视觉编码器本地权重优先/默认不联网**（新增 `RL_TRAINING_ENCODER_DIR` / `RL_TRAINING_ALLOW_ENCODER_DOWNLOAD`）、**openvla 分支删除**（含 cfg）、**vr_extented 模块级 print 改成调试图**；sim2sim 因"已在另一个仓库实现"**移出待办**；P2 action term 收敛成一条带做法的待办 | `codex/ll-train-detail-fix` |
| 2026-09-30 | **P2 action term 收敛完成（DEF-035）**：`PreTrainedPickAction` / `PreTrainedPickWBCAction` / `TeleopLLAction` 全部继承 `LowLevelPolicyActionBase`（共减 ~600 行重复机械）+ 删除无人注册的 `PreTrainedPolicyAction` + 新增复位时机探针；回归 11 OK / 1 SKIP / 0 FAIL、低层布局打印逐字节一致 | `codex/ll-train-detail-fix` |
| 2026-09-30 | **P3 EE 锚点一键化完成（DEF-036）**：新脚本 `sweep_ee_anchor.py`（4 组 `full/default/low/cfg` 逐组子进程 + Markdown 对比表 + 汇总 JSON）+ 修掉 `probe_root_height_termination.py` 里 `--freeze_ee_preset none` 的语义歧义（当前 cfg 默认已经是 low 锚点，故原 `none ≡ low`）；实测 `full 7.0% / default 1.8% / low 0.8% / cfg 0.8%` | `codex/ll-train-detail-fix` |
| 2026-09-30 | **星号导入遮蔽核实并收口（DEF-037）**：7 个奖励函数是故意覆盖（写进注释）、2 个事件函数不是遮蔽（官方没这两个名字）、地形 cfg 同名冲突 → 本仓库那份改名 `MIXED_TERRAINS_CFG`；4 任务冒烟全 OK | `codex/ll-train-detail-fix` |
| 2026-09-30 | 低层 known_issues **⑬ 作废**（当前实现是 `> 0.1`，原本的 "`> 0.0` 空操作" 已不成立）；`vr_extented` 的"无超时线程"**评估后降级**（daemon 线程 + 服务器主循环本不该有超时，唯一 UDP connect 不阻塞；盲改风险大于收益）；IsaacLab 本地魔改条目补注"不属于本仓库" | `codex/ll-train-detail-fix` |
| 2026-09-30 | **云端收割 + P1-1''' 主线定稿（DEF-038）**：`cloud_soft20k` 跑完并拉回本机（43 文件/303 MB），与同代旧代码 20k 对照 —— 静止漂移 **−35%**、三档摔倒率全降、步态"右后腿撇"消失；另拉回 `abl_pushonly_10k`（已完成）与 `cloud_roughslopes20k@10500`；新增观察项"前轮距收窄" | `codex/ll-train-detail-fix` |
| 2026-10-02 | **缺陷根因修正（DEF-040 §3/§5）**："右后腿往右前方撇"的根因是 **⑫**（`HeightInvariantEECommand.reset()`），不是镜像符号 bug —— 补跑第四根轴 `abl_legacyall_10k`（= main + ⑫）膝差 **1.156→0.008 rad**；`(0,0,0)` 速度误差 **0.1543→0.0916**。DEF-027 归因降级；顺带记下 hydra 传 `0`（int）被类型校验拒、要写 `0.0` 的坑 | `codex/ll-train-detail-fix` |
| 2026-10-02 | **两条新长跑已上云**：① `cloud_slowvx20k`（多地形 + v_x 课程台阶 ×2，验证地形等级能否不回退）② `abl_legacyall_10k`（第四根消融轴，**已跑完并拉回**，见上）；新增任务 `Rough-Slopes-SlowVx-History-Adaptation-Deeprobotics-M20-v0` / `History-Ablation-LegacyAll-Deeprobotics-M20-v0` | `codex/ll-train-detail-fix` |
| 2026-10-02 | **可视化脚本收敛（DEF-041）**：`gait_test`/`torque_test`/`tracking_test`/`test` 4 个老脚本 → 统一 `policy_report.py`（10 个角度 + `--compare` A/B + `report.md`/`summary.json`/`data.npz`）；出平地（cap12 vs 旧代码）与多地形 20k 两份实测报告；旧脚本加"已被取代"说明（保留待删） | `codex/ll-train-detail-fix` |
| 2026-10-03/04 | **可视化工具二三轮 + 云端跑通（DEF-042/043/044）**：分地形统计（fig11）、指令切换/变换能力（fig12）、root_z 图改散点、时长 10 s、`--from-npz`、`POLICY_REPORT_FONT`；修 3 个崩溃 bug；**云端跑通平地 A/B@10 s** 与**多地形分地形（256 envs）**；顺带结案"SlowVx 治住地形等级回落" | `codex/ll-train-detail-fix` |
| 2026-10-05 | **DEF-048**：`policy_report.py` 第四轮（schedule 驱动的多指令组合 / `--push-sweep` 抗扰扫描 / 每地形稠密高度图 / 图内英文 / npz 分组）+ **修掉"扰动课程一直是空操作"的实质 bug**（`EventManager.active_terms` 是 dict）；删掉 4 个已被取代的老可视化脚本；旧报告目录清空后用新工具重跑平地 A/B + 多地形 | `codex/ll-train-detail-fix` |
| 2026-10-06 | **DEF-049**：fig03 pitch 符号 bug（画反）+ 无姿态指令段误差留空 → 已修并用**云端重采**验证；限幅元数据改从 actuator 实例读；fig08/报告新增臂"饱和 vs 顶限位"诊断。**这一轮发现的未处理项（A 工具 5 条 + B 训练 5 条）已整段写进 `NEXT_SESSION_PROMPT.md`**，本文件不再重复 | `codex/ll-train-detail-fix` |
| 2026-10-06 | **`NEXT_SESSION_PROMPT.md` A 组 4 条全部做完（DEF-050~053）**：A5 臂负载表改**全 env 口径**（`arm_pop`，同一份 64 envs 数据里 env0 单独看 joint2 顶限位 0% vs 全 env 21.3%）、A6 删掉 IsaacLab 依赖里的 `[IK DEBUG]` 刷屏、A7 新增 `--reset-grace`（默认 25 步）剔除复位瞬态、A8 fig04/fig11 支持 A/B 双 label。**B 组（训练侧 5 条）仍未动，B1 最优先** | `codex/ll-train-detail-fix` |
| 2026-10-06 | **B 组开工**：B1 云端 20k 已开跑（`2026-10-06_15-32-05_cloud_ramp20k`，**扰动课程首次生效** `step=0 → 0.20×`；结果待回填）；**B2-②/⑤ 完成**（IK 关节保护：位置 clamp 默认开、目标限速默认关，A/B/C/D 消融见 DEF-054）；**B2-①③④ + B3/B4/B5 仍未做** | `codex/ll-train-detail-fix` |
| 2026-10-06 | **B2-①完成**（DEF-055）：EE 目标加 FK 可达性过滤（20 万次关节采样 → 1.5 cm 体素栅格）⇒ joint4 饱和 47.7%→10.7%、超速 93.4%→17.0%、`\|tau\|` 均值 71.8→28.4 N·m；`reach_joint_margin` 消融证明不需要。**剩余：B2-③（臂跟踪奖励，需你定方向）/ B2-④（夹爪刚度）/ B3（坡面）/ B4（SlowVx 落默认）/ B5 老账** | `codex/ll-train-detail-fix` |

**优先级定义**：P0 = 挡在"能部署/能继续训练"前面；P1 = 决定训练质量上限；
P2 = 高层 replay 与工程债；P3 = 验证工具与文档。

---

## P0 —— 挡在主线前面

**已清空**（2026-09-20）。原先两条都已完成，见 `DONE_zh.md` 第五节：

- 高层 P0/P1 链合并 → 已并入 `main`（`codex/hl-merge-p0`：`129848e`/`af4602d`/`07601e9`），
  4 个高层任务 2 iter 全 EXIT=0，启动打印 obs 维度与 checkpoint 一致；
- "训练完必须导出部署态策略" → 已固化（`export_deploy_policy.py` 默认 `exported_deploy/` +
  训练收尾打印命令 + `docs/train_history_flat_zh.md` 补章节），验收实测自检 0.000e+00。

> 仍**不要合** `codex/docs-review`（旧清单来源，内容已并入三份新文档）。

---

## P1 —— 训练质量（决定上限）

- [x] **P1-1' 静止伫立专项：软化版权重的 A/B（两轮都跑完，软化版胜出）** → 结果见 DONE 第七节；
      剩余"全长 20k 定稿"另立 P1-1'''（下面）
  - 依据：DEFECT_LOG_zh.md DEF-026。基线（旧代码 20k）命令 (0,0,0) 时 `err_vel_xy = 0.148 m/s`、摔倒 0.133。
  - 已做：`stand_still_vel_l2` / `stand_still_wheel_vel_l2` 两项惩罚 + `rel_standing_envs` 0.02→0.15，
    三条 25k 步爬升课程（DEF-026 §3）。
  - **第一轮 A/B（权重 −8.0 / −0.01，1500 iter，与同代旧代码 `model_1500` 对比）已完成**：
    目标指标修好 —— 漂移**两档难度都降 22~23%**（s0 0.1153→0.0886；play 0.1475→0.1157）；
    但同代摔倒率 s0 从 0.178 涨到 **0.708**（终止几乎全是翻倒）⇒ 惩罚量级 = 总回报的 122%，太重。
    数字/复现命令见 `DONE_zh.md` 第七节，机理见 DEF-026 §4。
  - **软化版 run 也跑完并验收通过**（`2026-09-29_21-38-50_stand_still_soft`，1000 iter / seed 42）：
    同代三方对比（旧代码 / 第一版 −8/−0.01 / 软化版 −2/−5e-4）命令 (0,0,0)：
    `err_vel_xy` 0.1030 / 0.0860 / **0.0857**；摔倒率 0.2481 / 0.3401 / **0.1065**；
    回合长度 890.0 / 845.6 / **920.1**；回报 40.83 / 38.44 / **46.28** ⇒ **软化版三项全赢**。
  - 步态对称性（同代 1000 iter / 命令 1.0,0,0）：后腿左右不对称 −0.0702 / +0.0287 / **−0.0099 m**；
    前/后轮距 0.503/0.462 → 0.459/**0.458**；`hl~hr` 镜像 RMS（hipx/hipy/knee）
    0.252/0.154/0.522 → **0.172/0.115/0.325** ⇒ "右后腿往右前方撇"基本消失。
  - **唯一待观察**：0.5 / 1.0 m/s 两档的跟踪误差比旧代码略差（0.16~0.19 vs 0.12~0.15），
    1000 iter 还早，需全长 run 判断。
  - **下一步（P1-1'''）**：与 P1-1''（`max_noise_std=1.2`）一起跑**全长 20k**
    （4 台 autodl 3090 并行：① 软化版 20k ② 第一版/重惩罚 20k 对照
    ③ `Rough-Slopes-*` 多地形多点 ④ 分量消融/基线复核）。判据沿用上面的三方口径。

- [ ] **P1-1''' 全长 20k 定稿（**已上云开跑**，见 DEFECT_LOG_zh.md DEF-032 §4）**
  - 云端口径：4096 envs / seed 42 / 20k iter（与历史基线一致），阈值与判据沿用 DONE 第七节。
  - **已启动 ①**：`bbc64d91a6-99f1820e`（4UGPU 3090，ssh 10.60.144.11:1237）跑
    `History-Adaptation-Deeprobotics-M20-v0` + 软化版静止惩罚，3.2 s/iter ≈ 18 h，
    `--run_name cloud_soft20k`，日志 `/root/run_soft20k.log`。
  - **已启动 ②**：`686346b9c6-b16aa8d9`（planner 3090，ssh 10.60.144.11:291）跑
    `Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0`（**多地形，本机跑不了**），
    6.3 s/iter ≈ 35 h，`--run_name cloud_roughslopes20k`，日志 `/root/run_roughslopes.log`。
  - **中途验收（iter=10000）已完成并回填**（2026-09-30）：run 目录已 `scp` 回本机
    （`logs/rsl_rl/history_adaptation/2026-09-30_00-09-25_cloud_soft20k`），
    固定命令 eval 三档的 err_vel_xy 比同代旧代码好 −4%~−15%、步态"右后腿撇"基本消失；
    训练期 `root_height_below_minimum` 偏高（加强 push + 15% 站姿样本所致），
    详见 `DONE_zh.md` 第七节 §5。
  - [x] **① 主线全长 20k 定稿：已完成并通过（2026-09-30，DEF-038）** —— `cloud_soft20k`
    已跑完 20000/20000 并全部拉回本机；与**同代同长**的旧代码 20k 对照（固定命令 eval）：
    静止漂移 **0.1644→0.1067（−35%）**、0.5 档 **−18%**、1.0 档持平，三档摔倒率全降；
    步态"右后腿撇"消失（后腿不对称 **−5.9cm → +0.4cm**，`hl~hr` knee 镜像 RMS **1.114→0.374**）。
    详见 `DONE_zh.md` 第七节 **d)**。
  - [ ] **新观察项（跟随本项，未定论）**：**前轮距收窄** —— 旧代码 0.482/0.515（差 3.3 cm）
    vs 软化版 **0.387/0.468**（差 8.1 cm）；10k 时还是 0.459/0.458（几乎相等）⇒ 10k 之后才收窄，
    且暂无稳定性代价（三档摔倒率都更低）。下次改 EE 区间 / 课程时复核（DEF-038 §6）。
  - [x] **步态归因：已查清（2026-10-02，DEF-040 §3 ④）** —— 补跑第四根轴
    `abl_legacyall_10k`（= `main` 行为 + ⑫，其余三项全退）⇒
    **"右后腿往右前方撇"的根因是 ⑫（`HeightInvariantEECommand.reset()`，DEF-034 §2），
    不是 `joint_mirror` 的符号 bug**：膝差 **1.156 → 0.008 rad**、`hl~hr` 膝镜像 RMS
    **1.257 → 0.334**（且 main 与 LegacyAll 的扰动完全相同 ⇒ 扰动/奖励都控制住了）。
    `params/env.yaml` 全文只差 32 行（行为差异仅"只读的启动期检查 + 删两个 weight=0 死项 + ⑫"）。
    **DEF-027 的归因已降级**（镜像符号修复仍保留：语义正确 + `abl_rewardonly` 的 `fl~hr` 膝 RMS
    0.749 是四格最好，但"修它是为了治撇腿"不成立）。
    顺带修正速度误差的归因：`main + ⑫` 的 `(0,0,0)` 是 **0.0916**（旧代码 0.1543，**−41%**），
    四格最好、摔倒率 0 ⇒ **⑫ 也是静止漂移那笔的最大贡献**，"加强扰动"（0.0968）与它同档。
  - [x] **消融 2×2 已完成**（2026-10-02）：pushonly / rewardonly 都跑完 10k 并拉回（DEF-040）。
    口径提醒：`eval_fixed_command.py` 不关 push 事件 ⇒ **"摔倒率"列不是同口径**
    （⑫-only/LegacyAll 用旧扰动 ±0.5，另三格用加强扰动 ±2/±1/yaw）；速度误差与步态列不受影响。
  - 云端队列现状（2026-09-30 19:35 CST，三台都在跑，**别关机**）：
    ① `cloud_soft20k` **已完成**；② `cloud_cap12_20k`（同实例队列自动接上，验证 P1-1''）；
    ③ `cloud_roughslopes20k` **10528/20000（≈52%）**，中途 checkpoint 已拉回存档
    （本机跑不了生成地形，端到端验收要等能跑地形的机器）；④ `abl_pushonly_10k` **已完成**
    （`model_9999.pt` 已拉回），`abl_rewardonly_10k` 19:33 刚起跑（≈明早 03:40）。
  - 待做：① ③④ 跑完后同样拉回来对比；② 多地形端到端验收（本机做不了，见 DEF-031）；
    ③ **用完记得关机**（DEF-032 §5）。

- [x] **P1-1'' 把 P1-1 的 `max_noise_std=1.2` 落成 cfg 默认值** → **2026-10-02 完成（DEF-039）**
  - 云端 `cloud_cap12_20k`（4096 envs / seed 42 / 20k / cap=1.2）**跑完并拉回**；与同 seed
    同长度的 `cloud_soft20k`（cap=0）对照：`Policy/mean_noise_std` 平台 **1.13（≤1.2）**、
    `Loss/learning_rate` 末段 **2.56e-4**（不再是 1e-5 地板）、`Train/mean_reward` 末 1000
    **35.92 vs 20.23**（基线 23.61）；固定命令 eval 三档 `err_vel_xy` **0.1014 / 0.1301 / 0.1594**
    （软 20k 是 0.1067 / 0.1332 / 0.1900）、**摔倒率 0.000 / 0.002 / 0.000**。
  - 已把 `RslRlPpoActorCriticHistoryCfg.max_noise_std` 默认值 **0.0 → 1.2**（复现旧行为：
    `agent.policy.max_noise_std=0`）；2-iter 冒烟通过 + run 的 `params/agent.yaml` 里
    `max_noise_std: 1.2`（证明默认值生效）。详见 `DONE_zh.md` 第十三节。

- [x] **P1-3 多地形任务的端到端验收** → **2026-10-02 完成（云端，DEF-040）**
  - 本机（Windows + A4000）跑 `terrain_type="generator"` 的任务会在 env 创建期死锁
    （DEF-031，原始代码同样复现）⇒ 全部改到云端做。
  - 云端 `Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0`：2-iter 冒烟 ✅ →
    **全长 20k 跑完**（09-30 00:14 → 10-01 13:07）并拉回本机。训练期结果：
    地形等级中段峰值 **5.80**（满分 9）→ 末段回落到 **3.6**；末段
    `time_out 0.803` / `bad_orientation_2 0.194` / `terrain_out_of_bounds 0.003`、
    `ep_len 901`（上限 1000）。
  - 注意：该任务**关掉了** `root_height_below_minimum`（反斜坡地形有低于 0 m 的部分），
    所以"摔倒"只能看 `bad_orientation_2`；`error_vel_xy 3.58` 是**口径产物**
    （v_x 课程重开到 ±5 m/s），不能与平地 run 的 0.84 直接比。
  - **遗留（新 TODO）**：① 后 1/3 的"地形等级回落"需要处理（把 v_x 课程推迟到地形稳定后放开 /
    拉长到 40k / 用 `-play-v0` 单独量地形能力）；② 云端固定命令验收**这次没拿到**——
    该实例上实测 ≈**1 s/env-step**（跑 3 档 × 1100 步要 1 h+），已改用"2 档 × 400 步 +
    `episode_length_s=6`"的短口径在后台重跑（结果写 `/root/roughslopes_eval_short.json`，
    见 DEF-040 §4）。
  - 同批的 `Rough-*` / `Rough-WO-Stairs-*` 三个老任务仍未跑（要在地形可用的机器上补）。

- [x] **P1-4 把"本机跑不了生成地形任务"钉进回归脚本** → **已完成（2026-09-30，DEF-034 §1）**
  - `scripts/reinforcement_learning/rsl_rl/smoke_regression.py`：逐任务独立进程 + 日志落盘 +
    **"日志 N 秒无增长 ⇒ 判 SKIP 并杀进程树"**（本机 `Rough-*` 稳定 SKIP，不再无限等待）
    + Markdown 汇总表 + 有 FAIL 才非零退出。实测 11 OK / 1 SKIP / 0 FAIL。


- [ ] **探明"探索噪声平台 ~1.5"（原"`noise_std`/`error_vel_xy` 长期退化"已归因，见 DEF-023）**
  - **已归因（2026-09-20，用 `summarize_run.py` 做阶段聚合）**：
    ① `Policy/mean_noise_std` **不是发散而是有界平台**：新 run 0.973/1.009/1.132/**1.473**（s0~s3），
    iter≈5000 起就在 1.43~1.49 抖动；旧 run（无课程）1.236→**1.505**，Δ末 = −0.035。
    机制：`log_std` 无上界 + `loss = surrogate + value_loss − entropy_coef*entropy`（`entropy_coef=0.01`）
    持续给熵正奖励，同时 `schedule="adaptive"`/`desired_kl=0.01` 把 `Loss/learning_rate` 压到地板
    （最低 1e-5）⇒ 高熵 + 低学习率，精度上界被压住（但 reward/ep_len 不降，训练没崩）。
    ② `Metrics/base_velocity/error_vel_xy` 的上升**与命令课程同形**（旧 run 0.15→0.77 同样升）
    ⇒ 是"命令范围放宽到 vx ±5 m/s"后的**口径产物**，不是策略退化：新 run 末 1000 的
    reward **23.6 vs 17.8**、ep_len **917 vs 768**、合计摔倒 **0.122 vs 0.331** 全面更好。
  - **A/B 已收尾（2026-09-22，数字与判定见 DEF-024 §4）**：开关 `max_noise_std` 已实现
    （默认 0 = 不限制）。**统一窗口**（iter 3125–3999，因为基线是 20k iter、阶段均值不可比）实测：
    - `max_noise_std=1.2`（run `2026-09-20_18-54-34_cap_noise_std`）**通过全部口径、建议作为默认**：
      噪声 1.405→**1.052**、reward 23.93→**38.52**、ep_len 858→**905**、s3 合计摔倒 0.194→**0.132**。
    - `entropy_coef=0.002`（run `2026-09-20_22-13-37_ent_coef_low`）也通过、但三项都略逊
      （reward 36.9、ep_len 878、摔倒 0.183）。
    - 意外点 `entropy_coef=0`（run `2026-09-20_19-30-43_ent_coef_low`）**只挂摔倒**（0.222 > 0.194）
      ⇒ **噪声不是越小越好**，存在中间最优区。
  - **剩余（本项未完全关闭）**：① `Loss/learning_rate` 在 s3 仍被 adaptive 调度压到 3e-5~2e-4
    （cap 甚至低于基线）⇒ "LR 地板"机制**未解**，候选：非 adaptive 调度 / 调 `desired_kl`；
    ② cap 的收益只在 4000 iter 上验过 ⇒ **部署前跑一次 20k 全长**（同 seed=42 / 4096 envs /
    `agent.policy.max_noise_std=1.2`）＋固定命令 eval，再改 cfg 默认值。
  - **自动化提醒（DEF-025）**：本环境桌面版 automation **不可用**（唤醒投递缺 `call_id` ⇒ 422
    且把线程写死），两条 automation 已 `PAUSED` ⇒ 巡检/收尾一律手动，命令见 DEF-024 §4 末尾。
  - 当时的修法（**已实现并实测，结论见上**）：① 给 `log_std` 加上界（`max_noise_std`/clamp，
    目标平台 ≤1.2）—— **采用**；② `entropy_coef` 0.01 → 0.005/0.002 —— **通过但未采用**（不如 ①）。落点：
    `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py:HistoryAdaptationPPORunnerCfg`
    （`policy.init_noise_std` / `noise_std_type="log"` / `algorithm.entropy_coef`）
    + `rsl_rl/rsl_rl/modules/actor_critic_history.py`（`log_std` 的取用处）。
    原候选"调 `body_*_rew_s3` 的 `num_steps` / 查 `track_lin_vel_xy_exp` 权重"**已降级**：
    实测 `Curriculum/body_pitch_rew_s3|body_roll_rew_s3` 在 iter≈3125 就到终值 0.8、
    `body_height_rew_s2` 在 s1 就到 0.8，与 `noise_std` 平台**不同期**，不构成解释。
  - 验收（**口径已改**）：(a) `noise_std` 平台 ≤1.2 且 `Loss/learning_rate` 不再长期贴地板（≥1e-4）；
    (b) 固定命令 eval（play/eval 探针，固定 vx/vy/yaw）下的 reward 与速度误差不劣于基线
    `2026-09-20_00-50-31`。跨阶段/跨 run 一律用 `summarize_run.py` 的**阶段均值**，不看单点。
  - 成本：`noise_std` 在 iter≈5000 已饱和 ⇒ A/B 可只跑 ~5k iter（比全量 20k 省 3/4），要更稳再补全量。

- [ ] **s3（完整任务）阶段的臂扰动鲁棒性**
  - 现象（2026-09-20 用阶段均值复核）：`root_height_below_minimum` s0/s1/s2 = 0.0258 / 0.0208 / 0.0275，
    **s3 = 0.1181**（末 1000 0.1151）；同期 `Metrics/ee_pose/orientation_error` 从 0.32 抬到 **0.85**
    （s3 才放开臂的大范围摆动），而 `height_error_bias_steady` 全程只有 1.3~1.8 cm
    ⇒ 剩余摔倒 = **臂摆动时倾覆**，不是高度控制失效。
  - 候选：① 收紧 EE 区间上界（`p_pitch` 上界 +36° ≈ 让臂举高、重心上移）；
    ② 延长 s1/s2 停留步数（现在各 25k）；③ 再评估 `root_height_below_minimum` 0.30 → 0.26。
  - 注意：**单独降阈值没用**（阈值反事实 0.30→0.26 只把 20s 触发率从 25.8% 降到 24.4%）；
    且 `bad_orientation_2` 在 s3 是**下降**的（0.0756→0.0144）⇒ 两个终止项必须看合计（s3 = 0.1325）。

- [ ] **低层 `known_issues` 剩余条目**（原编号）
  - [x] ① `joint_mirror` 用平方差做镜像惩罚（左右关节符号约定可能相反）
    → **2026-09-29 已修**：确认符号确实相反（对角对要求 θ_fl=θ_hr，真实关系是 −θ），
    新增 `joint_mirror_signed` 并扩到 4 对（含左右对），见 `DEFECT_LOG_zh.md` DEF-027；
  - [x] ⑧ `action_mirror`/`action_sync` 用 articulation 关节 id 索引动作向量 + 关节名不存在
    → **2026-09-30 删除**（DEF-034 §3：打开就炸的埋雷，且状态层 `joint_mirror_signed` 已覆盖）
  - [x] ⑩ `arm_rewards.py` 的 `grasp_success`/`ee_approach_object` 依赖不存在的 `object`
    → **2026-09-30 加明确报错**（DEF-034 §4：保留函数，但接错场景会给出可读提示）
  - [x] ⑪ `HeightInvariantEECommand._update_command` 覆盖父类没调 `super()`
    → **2026-09-30 实测澄清**（DEF-034 §2）：父类该方法本来 `pass`，`pose_command_w` 由
    `_update_metrics()` 每步更新（只比 `pose_command_b` 晚一拍），"永不更新"不成立
  - [x] ⑫ reset 后第一帧 `pose_command_b` 全 0（观测/奖励看到零位姿目标）
    → **2026-09-30 实测确认并修复**（DEF-034 §2）：加 `reset()` 覆写，修后与真实 EE 位姿差 0.000e+00
  - [x] ⑬ `UniformThresholdVelocityCommand._resample_command` 的 `> 0.0` 是空操作
    → **2026-09-30 复核后作废**：当前实现用的是 `> 0.1`（不是 `> 0.0`），语义正确 ⇒ 本条**不再存在**；
  - ⑭ EE 目标碰撞检查静默降级 + 硬编码 AABB；
  - ⑮ 硬编码常数（0.513 / 0.09 / 0.135 / action scale）；
  - [x] ⑱ 接触传感器与 articulation body 顺序不同（归因陷阱）
    → **2026-09-30 加启动期打印 + 名字可归因检查**（DEF-034 §5）

- [ ] `[可选]` **执行器刚度课程 / 刚度标定**
  - 依据：`piper_arm` 现在 `DelayedPD(300/20)`，注释里的目标区间是 60~100 / 0~20；
    实测软臂（40/8）臂速 0.95 vs 硬臂（300/20）1.74 rad/s。
  - 做法：刚度做成课程（前段 40/8 → 后期 300/20），或终值降到 60~100 后重新标定。
    现在 `root_height` 已降到 0.12，优先级可往后放。

---

## P2 —— 高层 replay 与工程债（"高层修改先暂放"期间不动）

- [x] **把剩下 3 个高层 action term 收进 `LowLevelPolicyActionBase`**（用户要求：尽量一个基类）
      → **2026-09-30 完成（DEF-035）**：三个类都只留"`_raw_actions` 分配 → `super().__init__`
      → 任务专属状态"，`apply_actions` 全部删掉、改用基类的 `_on_low_level_tick()` /
      `_on_reset(env_ids)` / `_extra_cache_tensors()` 钩子；顺带删除无人注册的
      `PreTrainedPolicyAction`（371 行）；验收 = 回归 11 OK / 1 SKIP / 0 FAIL +
      五个高层任务的 `[ll-replay:*]` 启动打印与改动前**逐字节一致** + 新探针
      `probe_reset_anchor_timing.py` 证明复位钩子读到的是复位后状态。
      （细节见 `DONE_zh.md` 第九节 / `DEFECT_LOG_zh.md` DEF-035。）

- [ ] **高层 `high_level_todo.md` 第 10 条的"待确认"**
  - `HLFlatPickTerminationsCfg_PLAY` 里 `lift_object` 与 `pick_success` 语义/命名重复；
  - [x] `highlevel/mdp/encoder.py` 的 `_frozen_encoders` 以 name 为 key 全局缓存（换 device
    会拿到旧设备上的模型）→ **2026-09-30 已修**（key 改成 `(name, device)`，DEF-034 §5）；
    仍待办：`torch.hub.load("facebookresearch/dinov2", ...)` 需要联网（离线机器上会失败），
    建议改成"本地权重优先，缺失再联网"。

- [ ] **工程性清理**（原 `known_issues.md` 二、1-8）
  - [x] 删掉已被 `policy_report.py` 取代的 4 个老脚本（`gait_test.py` / `torque_test.py` /
    `tracking_test.py` / `test.py`）→ **2026-10-05 已 `git rm`**（DEF-048）：新报告已经覆盖
    它们全部角度（且第 4 轮把"只测一档命令/PCD 稀疏/无臂力矩"等问题都修了），旧命令行语义
    也不再需要对照；历史里可查回。
  - [x] `mdp/__init__.py` 星号导入造成同名遮蔽 → **2026-09-30 核实并收口（DEF-037）**：
    逐名前查（ast 集合交集）后，`randomize_rigid_body_inertia` / `randomize_com_positions`
    **不是**遮蔽（官方没有这两个名字，是本仓库新增）；7 个奖励函数 + highlevel 的
    `undesired_contacts` 是**故意**同名覆盖（行为不变，已写进两个 `mdp/__init__.py` 的注释块）；
    唯一**意外**冲突是地形 cfg 同名 → 本仓库那份改名 `MIXED_TERRAINS_CFG`
    （`TERRAIN_CFGS["mixed"]` 不变，全仓库无其它引用）。验收：`py_compile` + 4 任务冒烟全 OK。
  - [x] `setup.py` 的 `packages` 只列顶层包 → **2026-09-30 改成 `find_packages`**（DEF-034 §5）；
  - [x] `cusrl_cfg_entry_point` 有 9 处指向不存在的 `agents/cusrl_ppo_cfg.py`
    → **2026-09-30 全部删除**（`deeprobotics_m20` 7 处 + `deeprobotics_lite3` 2 处；用户确认
    不用 cusrl 训练）+ 连 `setup.py` 里的 `cusrl[all]` 依赖一起删掉（DEF-034 §7）；
  - 依赖被本地魔改：`IsaacLab-5.1.0/.../task_space_actions.py`（`[IK DEBUG]` 打印 + 私有成员）——
    **注意这不在本仓库里**（是 editable 安装的 IsaacLab 检出），改它属于"动依赖"，建议改成
    给上游提 issue / 打 patch 文件，而不是直接改本地文件；
  - [x] `devices/vr_extented.py` 模块级 print → **2026-09-30 改成 `RL_TRAINING_VR_DEBUG=1` 才打**（DEF-034 §7）；
    剩下的"无超时线程"**评估后降级**（2026-09-30）：该线程是 `daemon=True`
    （不会挡住进程退出），`serve_forever()` / `run_forever()` 是服务器主循环**本来就不该有超时**；
    唯一的"网络调用"是 `_display_info()` 里为拿本机 IP 的 UDP `connect()`（UDP connect 不发包、不阻塞）。
    真正的问题只是"HTTPS/证书起不来时只在 daemon 线程里打 traceback"——但用户实测 VR 能正常连接，
    盲改（本机无法验证）风险大于收益 ⇒ 留着，等真出问题再动；
  - [x] `scripts/utils/mp4-png-composition.py` 4 处裸 `except:` → **2026-09-30 改成
    `except Exception:`**（DEF-034 §5）；
  - `logs/` 每个 run 50 个 `model_*.pt`（~5 MB/个，注意磁盘）；
  - 中文/英文注释混排、大段注释掉的代码。

---

## P3 —— 验证工具与文档

- [x] **回归矩阵脚本化** → **2026-09-30 完成**（DEF-034 §1/§6）：
  `scripts/reinforcement_learning/rsl_rl/smoke_regression.py`，13 任务实测
  **11 OK / 1 SKIP（生成地形，本机，DEF-031）/ 0 FAIL**；带卡死检测、Markdown 汇总、
  FAIL 才非零退出。**透传也补齐（2026-09-30，DEF-036 §5）**：新增 `--env KEY=VALUE`
  （注入环境变量，高层四条的低层 checkpoint 路径 `RL_TRAINING_LOW_LEVEL_POLICY_*` 走这个）
  和 `--hydra OVERRIDE`（透传 hydra 覆盖项），可重复；实测
  `--env RL_TRAINING_LOW_LEVEL_POLICY_TELEOP_HISTORY=does/not/exist_policy.pt`
  时子进程日志里出现该路径并抛 `FileNotFoundError`（证明真的透传到了子进程），
  且脚本按预期 FAIL 非零退出。
  （矩阵本身 2026-09-20 曾手工整跑一遍：8/8 EXIT=0，见 `DONE_zh.md` 第六节）
  - 低层（main 已有）：`History-Adaptation-Deeprobotics-M20-v0`、
    `Flat-Deeprobotics-M20-Piper-WBC-v0`、`Flat-Deeprobotics-M20-Piper-v0`、
    `Flat-Deeprobotics-M20-Piper-Arm-v0` —— 各 `--headless --num_envs 64 --max_iterations 2`。
  - 高层（合并高层链后）：Pick-Flat / Pick-WBC-Flat / Teleop / Nav-Flat-Teacher 同上。
  - 还缺的：把 8 条命令收成一个脚本（顺序跑、逐条记 EXIT、失败即非零退出、日志落到
    `logs/smoke/<日期>_<task>.log`），省得每次手敲；高层四条要能透传低层 checkpoint 路径
    （`RL_TRAINING_LOW_LEVEL_POLICY_*`）。预估 0.5~1 h（跑一次 ~6 min）。

- [x] **训练曲线自动分析脚本**（`scripts/reinforcement_learning/rsl_rl/summarize_run.py`）—— 2026-09-20 完成，见 `DONE_zh.md` 第四节
  - 依据：现在每次手抠 tensorboard（57 MB events，读一次 30~60 s）。
  - 内容：输入 run 目录 → 输出"关键指标 × 迭代"表（终止构成、ep_len、reward、
    `error_vel_xy`、`noise_std`、`height_error_*`、课程权重），支持两 run 对比。
  - 实现：阶段均值表（按课程阶段 s0~s3，边界从 `params/agent.yaml` 的 `num_steps_per_env` 推）
    + 采样网格表 + 两 run 对比（`--derive` 可把终止项求和，如"合计摔倒"）；
    首次解析 70 MB 事件文件 30~50 s，之后走 `<run>/.summary_cache.npz`（<1 s）。
  - 已用它完成 DEF-023（P1-1 归因 + P1-2 新证据）。

- [x] ~~sim2sim(MuJoCo) 落地~~ → **不做**（2026-09-30 用户确认：sim2sim 已经在**另一个
  仓库**里实现了；本仓库只保留 DEF-021 的"接口契约 + 部署态导出 + 探针"，不再自己搭 MuJoCo 脚本）。

- [x] **把"EE 锚点 4 组对照"固化成一键脚本**
      → **2026-09-30 完成（DEF-036）**：新脚本
      `scripts/reinforcement_learning/rsl_rl/sweep_ee_anchor.py`（默认 4 组
      `full/default/low/cfg`，逐组独立 Isaac 子进程 + Markdown 对比表 + 汇总 JSON，
      FAIL 非零退出）；同时修掉旧探针 `--freeze_ee_preset none` 的语义歧义
      （当前 cfg 的默认 EE 区间**已经是 low 锚点**，所以原来的 `none ≡ low`），
      并补 `full`（= 课程 s3 全分布，才是真正的"无课程"）与 `--json_out`。
      实测：`full 7.0% / default 1.8% / low 0.8% / cfg 0.8%`（20k 策略，512 envs × 20 s）⇒
      排序与 DEF-006 一致、`cfg≡low` 得到机器验证。（详见 `DEFECT_LOG_zh.md` DEF-036。）

- [x] **文档收尾**（2026-09-20 完成）
  - [x] 合并高层分支时删掉它们带来的旧 `docs/review/*.md`（`30d5411`、`0772757`）：
    `docs/review/` 现在只剩 TODO/DONE/DEFECT_LOG/NEXT_SESSION_PROMPT + `templates/`。
  - [x] 代码/注释/探针里指向旧文档的 9 处引用改指新文档（`bad_orientation_analysis_zh`
    → `DEFECT_LOG_zh.md` DEF-006、`history_low_level_policy_todo` → DEF-014、
    `known_issues #19` → `TODO_zh.md` P1-1）；`grep` 复核：仓库内（除三份文档自身的
    "旧文档去哪了"表）已无悬空引用。

---

## 旧文档去哪了

| 旧文档 | 现在读什么 | 原文在哪（git） |
|---|---|---|
| `high_level_todo.md` | 本文件 P0/P2 + `DONE_zh.md` 第二节 + `DEFECT_LOG_zh.md` DEF-004~007 | `codex/hl-fix-ee-command @ 0389253` |
| `known_issues.md` | 本文件 P1/P2 + `DONE_zh.md` 第三节 + `DEFECT_LOG_zh.md` | 同上 |
| `history_low_level_policy_todo.md` | `DONE_zh.md` 第四节 + `DEFECT_LOG_zh.md` DEF-014 | 同上 |
| `bad_orientation_analysis_zh.md` | `DONE_zh.md` 第一节 + `DEFECT_LOG_zh.md` DEF-001~003 | `codex/ll-height-stability @ 96e1b66` |
| `progress_summary_zh.md` | `DONE_zh.md` 全文 + 本文件"更新记录" | 同上 |
| `todo_master_zh.md` / `next_session_prompt.md`（旧） | 本文件 + `NEXT_SESSION_PROMPT.md` | `codex/docs-next-session @ bf9c5d3` |
