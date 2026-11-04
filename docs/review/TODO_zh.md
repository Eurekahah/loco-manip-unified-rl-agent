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
| 2026-10-16 | 历史编码器课题：真 env 探针定位"缓存时序"+ "特权向量量级失衡/死维/漏 COM"（DEF-032/033），实现 V4 特权向量 + 探索口径对齐参考实现 + `Latent/*` 监控；新增 P1-5（待云机实测） | `codex/hl-fix-adaptation`（工作区） |
| 2026-10-27 | 最新 20k checkpoint 统一按 **0.55 m 高度上限**复核；归档 23 个不重复末 checkpoint 与参数清单；为每个 run 生成 md/png/npz 性能记录；清理 27 个云端 smoke/短测 run。结论迁入 DONE 第九节 | 云端 753/802/1029；`logs/cloud_20k_checkpoints/logs/rsl_rl/perf_summary.md` |

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

- [ ] **P1-1''' 全长 20k 定稿（需要 autodl 机器；本机 6 s/iter ⇒ 20k 要 33 h）**
  - 要做什么：把 P1-1'（软化后的静止惩罚）+ P1-1''（`max_noise_std=1.2`）合到一起跑全长，
    并按 DONE 第七节 §4 的 `probe_gait_symmetry.py` 复核步态对称性。
  - 顺带要跑：`Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0` 的冒烟 + 短训
    （本机跑不了生成地形，见 DEF-031）。

- [ ] **P1-1'' 把 P1-1 的 `max_noise_std=1.2` 落成 cfg 默认值**
  - 现状：DEF-024 §4 已证明 cap=1.2 在 4000 iter 上全面更好，但 `rsl_rl_ppo_cfg.py` 的
    `RslRlPpoActorCriticHistoryCfg.max_noise_std` 仍是 0（不限制），只能命令行覆盖。
  - 依据（DEF-024 的遗留条件）：先在**全长 20k / 同 seed / 4096 envs** 上再验一次，
    通过后把默认值改成 1.2。
  - 验收：全长 run 的 `Policy/mean_noise_std` 平台 ≤1.2，`Train/mean_reward` 不劣于基线。
  - 注：本次 2000-iter 的静止专项 run 仍是 `max_noise_std=0`，与基线口径一致，便于对比。

- [ ] **P1-3 多地形任务的端到端验收（被 DEF-031 挡住）**
  - 现状：`Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0` 只做到 cfg 级验证；
    本机（Windows + A4000）跑 `terrain_type="generator"` 的任务会在 env 创建期死锁
    （DEF-031，原始代码同样复现）。
  - 要做什么：换到能跑生成地形的机器后 ① `--num_envs 64 --max_iterations 2` 冒烟；
    ② 短训（≥2000 iter）看地形通过率与 `root_height`/`bad_orientation_2` 合计摔倒；
    ③ 把数字回填 DEF-029。
  - 同批要补的还有 `Rough-*` / `Rough-WO-Stairs-*` 三个老任务 —— 它们在本机也从未跑过。

- [ ] **P1-4 把"本机跑不了生成地形任务"钉进回归脚本**
  - 现状：DEF-031。回归矩阵（P3）若不加区分，会在本机对 `Rough-*` 任务无限等待。
  - 做法：给冒烟脚本加"地形类任务跳过 / 超时（如 5 min 无输出即判 SKIP）"的分支，
    并在日志里打印原因；换机器后自动恢复执行。

- [x] **P1-5 V4 实验（特权向量对齐参考实现 + 探索口径）——已启动（2026-10-16 15:0x，BOX4/1029）**
  - 目的：回答"adaptation 通路是不是被**特权向量（量级/死维/漏项）+ 探索口径**坑死的"，
    而不是先做结构消融（LOCAL 已确认两个编码器都塌成常数，见 DONE 第八节）。
  - 任务：`History-Adaptation-V4-Deeprobotics-M20-v0`（= fixV2 的历史窗口/奖励/课程/
    非对称 critic **逐字段不变**，只换特权组 + 两个探索超参）。
  - 改动：① 特权组换成单一 term `privileged_env_params_v4`（89 → **54 维、0 常量维、
    每通道归一到 [-1,1]**、加回 base COM、删掉 27 维惯量 Δ 与 4 个恒 0 的轮 stiffness 死维）；
    ② `entropy_coef` 0.01 → 0.0、`max_noise_std` 0 → 1.2（对齐参考实现口径，见 DEF-032）；
    ③ 新增 `Latent/*` TB 指标（z_priv/z_hist 分散度、输入分散度、**教师 jac ‖∂z/∂x‖**）。
  - **刻意保留** `learning_rate=1e-3 + adaptive`（参考实现是 2e-4 fixed）：本预算 ≈2e9 样本，
    只有 DWL 的 1/4，砍 5 倍 LR 会把"教师是否还活着"这个主问题污染掉；LR 对照单独做。
  - 判据：`Latent/teacher_jac` 在 it≈8000（λ 全开）后是否仍 ≥ ~0.3（初值 1.29），
    `Latent/z_priv_spread` 不塌；然后才是 `eval_fixed_command.py` 的三档误差/摔倒率。
  - 验收：与 `fixV2`（0.0952/10.5%/910/36.5）、`armB`（0.0997/11.3%/911/35.9）同口径比较。
  - **进展（2026-10-16）**：7 个文件已推到 BOX4(`10.60.144.11:1029`) 的
    `/root/autodl-tmp/loco-manip-unified-rl-agent`（NAS 存档 `tools/v4_files.tar.gz.b64`）；
    冒烟通过（`[privileged-v4]` 54 维 / COM 基准跨环境 std=0.00e+00 / `Latent spread 0.0915|0.0265`）；
    20k 已 `setsid nohup` 起跑，日志 `logs/rsl_rl/history_adaptation_v4/train_v4_20k.log`，
    队列已登记 `RUNNING:BOX4`。**下一步：等 it≥8000 看 `Latent/teacher_jac` 是否还 ≥0.3**。
  - ⚠️ 本机沙箱内 socket 被拦、Jupyter contents API 路径相对 `/root`、云机 python 是
    `env_isaaclab`（不是 `env_isaac_lab`）—— 三条都是这次踩过的坑，部署脚本 `logs/xfer/{jput,v4_push}.py`。
  - **2026-10-16 追加：三台实例全用上，做 2×2 归因**（V4 一次改了两件事：特权向量 + 探索口径，
    所以必须拆开才知道是谁救活的）：

    | 臂 | 特权向量 | 探索口径 | 任务 | 机器 | 状态 |
    |---|---|---|---|---|---|
    | fixV2 | 89 维 | entropy 0.01 / 无上限 | `History-Adaptation-V2-*` | NEWBOX | 已有 20k + eval |
    | **Expl0**（新） | 89 维 | entropy 0 / cap 1.2 | `History-Abl-Expl0-*` | BOX2(753) | **跑中** |
    | **PrivV4**（新） | 54 维归一 | entropy 0.01 / 无上限 | `History-Abl-PrivV4-*` | NEWBOX 转 NEWBOX(802) | **跑中** |
    | V4 | 54 维归一 | entropy 0 / cap 1.2 | `History-Adaptation-V4-*` | BOX4(1029) | **跑中** |

    解释规则：`teacher_jac`（it≥8000）与 `z_priv_spread` 在**哪两个臂**活下来 ⇒ 就是那一项起决定作用；
    若只有 V4 活、两条单变量臂都塌 ⇒ 说明两项是**互补**的（缺一不可）。
    **早期信号**（it≈13/19）：54 维臂 `z_priv_spread≈0.34`，89 维臂只有 `≈0.04` ⇒ 归一化确实让
    teacher latent 能"动起来"（与量级失衡的诊断一致）。
  - **2026-10-17 结果：三条 20k 跑完，四条臂全部塌陷**（DEF-034）：
    teacher_jac 末值 fixV2 0.0276 / Expl0 0.0188 / PrivV4 0.0004 / V4 0.0120（初值 1.29）
    ⇒ 特权向量与探索口径都不是主因，损失结构（互相吸引、无锚点）才是；必须上 P1-6。
  - **P1-6 锚点臂（已实现 + 本地验证，2026-10-17）**：`PrivilegedDecoder` + `teacher_recon_coef`
    + `Latent/teacher_recon_r2`；任务 `History-Anchor-Deeprobotics-M20-v0`（= V4 全配置，只多一条重建锚）。
    单元测试：常数教师 R²=−0.04（锚点不可被常数解满足）；训练 300 步 R² −0.04→+0.997。
    端到端冒烟：`[privileged-v4] 54 维` + `Privileged decoder` + `Mean recon loss 0.3005`。
    **verdict（it=8419，λ 已于 7000 全开）：`recon_r2 = 0.690` 稳定（4610→8419），teacher_jac 0.221、
    priv spread 0.022 —— 教师"压缩保信息"，塌陷不再是优解；无锚臂同期 jac 0.004~0.027 且信息为 0。**
    ⇒ P1-6 的机制成立；NoHist 20k 已于 2026-10-17 20:33 UTC 在 NEWBOX(802) 启动。
    **20k 完成（it=19999，recon_r2=0.691）：评测 err_vel_xy = 0.0922/0.1095/0.1392（三档全臂最低）、
    摔倒 6.1%/4.8%/4.8%、回合 953/967/967、回报 38.7/40.0/38.9 ⇒ 既修活通路又提升精度（DEF-034 §6）。**
  - **P1-7 无历史臂（已实现 + 本地验证，2026-10-17）**：`mdp.history_zero_obs` +
    `WBCObservationsV4NoHistCfg`；任务 `History-Abl-NoHist-Deeprobotics-M20-v0`；探针实测 history
    组 830 维 830/830 全零。启动条件：必须等 V4 或 Anchor 确认通路活着。
    **结果（20k，2026-10-18）**：NoHist 0.0924/2.7%/991/40.1（vx=0），与 Anchor（有历史、
    recon_r2=0.69）的 0.0922/6.1%/953/38.7 **几乎相同、摔倒更少** ⇒ 历史信息在当前任务上贡献≈0
    （DEF-034 §7）。**下一步不是继续调结构，而是提高任务对环境的敏感度（更强 DR / 重载搬运 / 变地形）。**
  - **2026-10-17 机器状态**：NEWBOX(802)/BOX2(753) 跑完 20k 后 CUDA 掉卡
    （`torch.cuda.is_available()==False`，NVML Unknown Error），需重启动实例；BOX4(1029) 健康。
  - **2026-10-16 追加：等一台空出来后的队列**（P1-6/P1-7）：
    * **P1-6 锚点臂**（参考实现没有、我们认为必需）：给 `z_priv` 加监督锚（自编码重建归一化特权向量
      或 InfoNCE 跨环境对比）+ 教师独立 optimizer(lr/10)，λ 保留；
    * **P1-7 B 臂**：在**活下来的那套配置**上把历史窗口置零/按时间打乱，回答"历史信息到底有没有用"
      （在 adaptation 通路是死的时候跑没有意义，见 §"为什么先修锚点"）。

- [ ] **P1-8 多地形环境敏感度实验（Anchor vs NoHist）—— 20k 跑中（2026-10-19）**
  - 任务：`Rough-Slopes-History-Anchor-*`（粗糙 0.01~0.05 + 正反斜坡 + 平地 + 地形课程，
    V4 54 维特权 + 教师锚 + entropy 0）vs `Rough-Slopes-History-NoHist-*`（**同宽 700 维全零历史**）。
  - 本机 cfg 级验证通过（`probe_task_cfg.py`，绕开 DEF-031）；云端 64 envs 冒烟 EXIT=0。
  - NEWBOX(802)=Anchor、BOX2(753)=NoHist，4096 envs / seed42 / 20k。
  - 判据：① Anchor 的 `Latent/teacher_recon_r2` 在 it≥8000 仍 ≥0.5；② 评测上 Anchor 是否**显著优于**
    NoHist（平地上两者相同，若地形上出现差距 ⇒ adaptation 的价值被证明）；③ 地形通过率/摔倒构成。
  - 验收：`eval_fixed_command.py` 三档（512 envs/1100 步/seed42，play 任务），JSON 写 NAS `eval/`。
  - **2026-10-27 h055 复核**：地形 history 长度消融（Len5/10/20/50）与平地/地形 h060 checkpoint
    已统一到 `body_pose.height_range=(0.33, 0.55)`。Len5/Len10 最好（地形摔倒 8.8~11.1%），
    Len20 明显退化，Len50 居中；平地摔倒 1.0~2.5%。完整数字见 `DONE_zh.md` 第九节。

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
  - ⑧ `action_mirror`/`action_sync` 用 articulation 关节 id 索引动作向量 + 关节名不存在
    （当前 weight=0，属埋雷）；
  - ⑩ `arm_rewards.py` 的 `grasp_success`/`ee_approach_object` 依赖不存在的 `object`（dead code）；
  - ⑪ `HeightInvariantEECommand._update_command` 覆盖父类没调 `super()`（`pose_command_w` 永不更新）；
  - ⑫ reset 后第一帧 `pose_command_b` 全 0（观测/奖励看到零位姿目标）；
  - ⑬ `UniformThresholdVelocityCommand._resample_command` 的 `> 0.0` 是空操作；
  - ⑭ EE 目标碰撞检查静默降级 + 硬编码 AABB；⑮ 硬编码常数（0.513 / 0.09 / 0.135 / action scale）；
  - ⑱ 接触传感器与 articulation body 顺序不同（归因陷阱，建议加断言）。

- [ ] `[可选]` **执行器刚度课程 / 刚度标定**
  - 依据：`piper_arm` 现在 `DelayedPD(300/20)`，注释里的目标区间是 60~100 / 0~20；
    实测软臂（40/8）臂速 0.95 vs 硬臂（300/20）1.74 rad/s。
  - 做法：刚度做成课程（前段 40/8 → 后期 300/20），或终值降到 60~100 后重新标定。
    现在 `root_height` 已降到 0.12，优先级可往后放。

---

## P2 —— 高层 replay 与工程债（"高层修改先暂放"期间不动）

- [ ] **`pre_trained_policy_action` / `openvla_pick_action` 迁移到 `LowLevelPolicyActionBase`**
  - 现状：只补了 `ll_command`/`ll_command_w` 接口；本体仍是"就地改 cfg + 自己的
    last_action 闭包 + 无布局校验/history"；`openvla_pick_action` 的 `ee_goal` 仍是**世界系**
    （③ 只统一了 3 个 term）。
  - 限制：前者没被任何 task 注册、后者要 OpenVLA 7B 模型 → 只能做 import + 维度级验证。

- [ ] **高层 `high_level_todo.md` 第 10 条的"待确认"**
  - `HLFlatPickTerminationsCfg_PLAY` 里 `lift_object` 与 `pick_success` 语义/命名重复；
  - `highlevel/mdp/encoder.py`：`torch.hub.load` 需要联网；`_frozen_encoders` 以 name 为 key
    全局缓存（换 device 会拿到旧设备上的模型）。

- [ ] **工程性清理**（原 `known_issues.md` 二、1-8）
  - `mdp/__init__.py` 星号导入造成同名遮蔽（`ROUGH_TERRAINS_CFG` /
    `randomize_rigid_body_inertia` / `randomize_com_positions` 覆盖官方实现）；
  - `setup.py` 的 `packages` 只列顶层包、`cusrl_cfg_entry_point` 指向不存在的模块；
  - 依赖被本地魔改：`IsaacLab-5.1.0/.../task_space_actions.py`（`[IK DEBUG]` 打印 + 私有成员）；
  - `devices/vr_extented.py` 模块级 print + 无超时线程；`scripts/utils/mp4-png-composition.py`
    4 处裸 `except:`；
  - `logs/` 每个 run 50 个 `model_*.pt`（~5 MB/个，注意磁盘）；
  - 中文/英文注释混排、大段注释掉的代码。

---

## P3 —— 验证工具与文档

- [ ] **回归矩阵脚本化**（矩阵本身 2026-09-20 已手工整跑一遍：8/8 EXIT=0，见 `DONE_zh.md` 第六节）
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

- [ ] **sim2sim(MuJoCo) 落地**（承接 DEF-021 的部署文档）
  - 现在只有"接口契约 + 探针 + 文档"，**还没有可运行的 MuJoCo 部署脚本**。
  - 下一步：用 `deep_robotics_model/M20_Piper_own/mjcf/M20_Piper_own.xml` 按文档第 8 节的
    七步顺序搭（零动作站立 → 零位移命令 → 开 IK → 小步进 → 速度命令）；
    数值验收用第 8 节的"ONNX vs TorchScript 相对误差 ~1e-6"。
  - 已知待解：MJCF 的 `timestep=0.002` vs Isaac `0.005`；IK 需要自己实现（DLS λ=0.01）。

- [ ] **把"EE 锚点 4 组对照"固化成一键脚本**
  - 现在靠 `probe_root_height_termination.py --freeze_ee_preset {none,default,low}` 手工跑
    （512 envs × 20 s ≈ 2.5 min）。以后改 EE 区间/锚点前先跑一遍。

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
