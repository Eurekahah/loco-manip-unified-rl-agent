# 下个 session 的开工 prompt（可直接整段复制）

> 说明：这是给"下一个 AI session"用的交接 prompt。历史版本在
> `codex/docs-next-session @ bf9c5d3`。写完新版本就把这一节整体替换掉。

```text
仓库：D:\nvidia-isaac-sim\loco-manip-unified-rl-agent（Isaac Lab 5.1 + rsl_rl 的轮腿+Piper 机械臂 RL 训练库）
python：C:\Users\autolab\miniconda3\envs\env_isaac_lab\python.exe（跑 Isaac 脚本前设 $env:PYTHONIOENCODING='utf-8'，
        都必须 --headless；沙箱里 .git 只读，git 命令要 -c safe.directory=D:/nvidia-isaac-sim/loco-manip-unified-rl-agent
        并需要 escalate 才能 checkout/commit/push；`git checkout` 可能被"stat-dirty 的假 M"挡住，
        确认内容一致后用 `git checkout -f`）

【当前状态】部署基线仍是 **main @ 2d49f47**（tag `deploy-baseline-2026-09-20`），
  对应 run `logs/rsl_rl/history_adaptation/2026-09-20_00-50-31` 的 exported_deploy/*。
  **本条主线之外的新工作都在分支 `codex/ll-train-detail-fix`**（基于 main，未合并）：
    6d76de4  训练细节专项：静止伫立/镜像符号/扰动加强/多地形/遥操 history（+ DEF-026~031）
    e37865f  静止伫立惩罚软化（-2.0/-5e-4）+ 步态对称性探针 + 1500-iter A/B 实测回填
  该分支做完的 5 件事（对应本次需求 1~5）：
    ① 静止伫立：新增 stand_still_vel_l2 / stand_still_wheel_vel_l2（只按命令门控）+ rel_standing_envs 0.02→0.15
       + 三条 25k 步爬升课程；实测命令 (0,0,0) 的 err_vel_xy 同代降 22~23%，但同代摔倒率上升（见下）
    ② 镜像符号：joint_mirror → joint_mirror_signed（对角对符号本应为负！扩到 4 对含左右对）—— "右后腿往右前方撇"的根因
    ③ 多地形：ROUGH_SLOPES_FLAT_TERRAINS_CFG（粗糙 0.01~0.05 + 正反斜坡 + 平地）
       → 新任务 Rough-Slopes-History-Adaptation-Deeprobotics-M20-v0(+play)
    ④ 扰动加强：push 间隔 5~10 s、x±2/y±1/yaw±0.52；disturbance_ramp 0.2x→1.0x / 50k 步
    ⑤ 遥操 history：新任务 Isaac-M20-Piper-Teleop-History-v0（低层默认指向 history 部署态策略）

  **该分支后续的几批（都在同一分支上，最新 commit 见 `git log -1`）**：
    DEF-032~033 云端 autodl 接力（环境复制 + 四条 20k 长跑排进队列，见下）
    DEF-034 本机收尾批：回归矩阵脚本化（smoke_regression.py）/ known_issues ⑧⑩⑪⑫⑱ /
            cusrl 全删 / 视觉编码器本地权重优先不联网 / openvla 分支删除 / vr 打印改调试图
    DEF-035 **P2 完成**：PreTrainedPickAction / PreTrainedPickWBCAction / TeleopLLAction
            全部继承 LowLevelPolicyActionBase（减 ~600 行重复机械），删掉无人注册的
            PreTrainedPolicyAction；验收 = 回归 11 OK/1 SKIP/0 FAIL + 5 个高层任务的
            `[ll-replay:*]` 打印逐字节一致 + 新探针 probe_reset_anchor_timing.py
            （结论：`_on_reset` 钩子里读到的 `robot.data` 已是**复位后**状态）
    ⇒ **TODO P2 的 action term 收敛已清空**，P2 只剩 `mdp/__init__.py` 星号导入遮蔽
      + 遗留未使用 import + 调试可视化重复这几条工程债。
    DEF-036 **P3 完成**：EE 锚点对照一键化 —— 新脚本 `sweep_ee_anchor.py`
            （4 组 full/default/low/cfg，逐组独立 Isaac 子进程 + 对比表 + 汇总 JSON）；
            顺带修掉旧探针 `--freeze_ee_preset none` 的语义歧义（当前 cfg 默认 == low 锚点
            ⇒ 原 `none ≡ low`），并补 `full` 组与 `--json_out`。
            实测 4 组：full 7.0% / default 1.8% / low 0.8% / cfg 0.8%（20k 策略，512 envs × 20 s）。
    DEF-037 **P2 工程债之一完成**：星号导入遮蔽核实 —— 7 个奖励函数是"故意同名覆盖"（已注释）、
            `randomize_rigid_body_inertia`/`randomize_com_positions` 不是遮蔽（官方无此名字）、
            唯一意外冲突是地形 cfg 同名 ⇒ 本仓库那份改名 `MIXED_TERRAINS_CFG`。
            P2 只剩"依赖被本地魔改 / vr_extented 无超时线程 / logs 占盘 / 注释与死代码"这几条。

【本 session 的实测结论（都在 DONE_zh.md 第七节，务必读那一节再动手）】
  * 固定命令判据：`scripts/.../eval_fixed_command.py`（`Train/mean_reward` 带命令课程、跨 run 不可比）。
    基线（旧代码 20k）命令 (0,0,0)：err_vel_xy = 0.148 m/s、摔倒 0.133。
  * **同代对照才有效**：基线 run 每 500 iter 存盘，直接取 2026-09-20_00-50-31/model_1500.pt 当"旧代码同代"。
    1500 iter 实测：err_vel_xy 两档难度都降 22~23%（s0 0.1153→0.0886、play 0.1475→0.1157）；
    但摔倒率 s0 0.178→0.708（终止几乎全是 bad_orientation_2 翻倒）。
  * 机理：惩罚量级原本 = 同一步总回报的 122%（-0.483/s vs +0.396/s）⇒ 策略学会"冻住轮子"。
    已软化到 -2.0/-5e-4（合计 ~19%）；**但软化前后的训练曲线几乎重合**
    （bad_orientation_2 逐迭代对比：iter44 峰值都 ~0.55、iter175 都 ~0.33）⇒
    权重不是早期摔倒的主因，更可能是"真的去站"本身在 1500 iter 站不住 + 扰动 4x 加强。
    ⇒ **下一步要跑全长 20k 才能定稿**（TODO_zh.md P1-1'）。
  * 步态：`probe_gait_symmetry.py` 把撇腿量化 —— 旧 20k 基线在 play/命令(1,0,0) 下
    hr 的 knee=-0.375（hl 是 -1.347）、足端往前 0.33 m、后轮距 0.519 > 前 0.479 ⇒ 就是"右后腿往右前方撇"。

【⚠️ 本机（Windows + RTX A4000）跑不了"生成地形"的任务】见 DEF-031：
  Rough-* / Rough-Slopes-* 在 env 创建期死锁（日志停在 simulation_context 的 physics warning、
  带 carb.tasking SharedMutex 断言；GPU 4%、CPU 30s/20min）。**原始代码同样复现**（git stash 对照过），
  所以不是本次改动引入；多地形任务只做到 cfg 级验证，端到端冒烟/训练要换机器（TODO P1-3/P1-4）。

【本机性能】4096 envs ≈ 5.5~6.5 s/iter（1024 envs 也只快 15% ⇒ 绝大部分是固定开销，不是环境数）；
  所以 2000 iter ≈ 3 小时、1000 iter ≈ 1.6 小时。别轻易开长跑，先想清楚验收口径。

【云端（autodl 私有云 TiEV）——本 session 已接手，见 DEF-032】
  * 控制台 https://private.autodl.com/console/instance（租户 TiEV-Tj）；现有 6 个历史实例（各 1×3090），
    **本项目那份环境在 `ffda41bd1f-38f1325f`**（`/root/autodl-tmp/{IsaacLab,loco-manip-unified-rl-agent}` +
    conda env `/root/miniconda3/envs/env_isaaclab` = 20GB），但它所在主机空闲 GPU 是 0/2
    ⇒ 目前用**无卡模式**开机当文件源。
  * **免密/复制套路**（DEF-032 §3）：控制台复制图标拿 `ssh -p <port> root@10.60.144.11` + 密码；
    实例之间 `rsync`（同一物理主机内几分钟传完 20GB 环境）；`git fetch` 在部分实例不通
    ⇒ 用本机 `git bundle` + `scp` 再 `git fetch <bundle> 'branch:refs/heads/branch'`。
  * **驱动 580.x 的主机才干净**（`ffda41bd1f` 580.173.02 / `bbc64d91a6` 580.178.04）；
    570/535 的主机启动会打 Vulkan 报错，但 headless 仍能训（实测 GPU 利用率 80%）。
  * 云端速度实测：History 平地 **3.2 s/iter**、多地形 6.3 s/iter（本机 5.5~6.5）⇒ 20k 约 18~35 h。
  * 已开两条长跑（4096 envs / seed 42 / 20k iter，**结果待回填 DONE 第七节**）：
    `bbc64d91a6-99f1820e`（ssh 端口 1237）跑 History 主线 `--run_name cloud_soft20k`；
    `686346b9c6-b16aa8d9`（端口 291）跑 Rough-Slopes 多地形 `--run_name cloud_roughslopes20k`。
  * **2026-09-30 19:35 CST 的收割结果（DEF-038）**：
    `cloud_soft20k` **已跑完 20000/20000**（00:09→19:34），43 文件/303 MB 已拉回本机
    `logs/rsl_rl/history_adaptation/2026-09-30_00-09-25_cloud_soft20k/`；同实例队列自动接上
    `cloud_cap12_20k`（验 P1-1''）。多地形那条 `10528/20000`（≈52%，ETA ~17 h），
    中途 `model_10000/10500.pt` 已拉回存档（**本机跑不了生成地形 ⇒ 只能存档**）。
    第三台（无 ssh，走 Jupyter）`abl_pushonly_10k` **已跑完**（`model_9999.pt` 已拉回），
    `abl_rewardonly_10k` 19:33 刚起跑。
  * **2026-10-02 00:20 CST 的收割结果（DEF-039/040）：四条 run 全部跑完**
    - `cloud_soft20k`：20000 ✅（DEF-038）｜`cloud_cap12_20k`：**20000 ✅**
      （09-30 19:36→10-01 14:44）⇒ **已把 `max_noise_std` 默认值 0→1.2 定稿**（DEF-039）。
    - `cloud_roughslopes20k`（多地形）：**20000 ✅**（09-30 00:14→10-01 13:07，300 MB 已拉回）。
      训练期：地形等级峰值 5.80 → 末段 **3.6/9**；末段 `time_out 0.803` /
      `bad_orientation_2 0.194` / `terrain_out_of_bounds 0.003`、`ep_len 901`。
      该任务**关掉**了 `root_height_below_minimum`（反斜坡有 <0 m 部分）。
    - `abl_pushonly_10k` / `abl_rewardonly_10k`：都 **10000 ✅** 并已拉回。
    - 三台实例现在都**空着**（GPU 0%）⇒ **记得关机**；`roughslopes_eval_short.json`
      是唯一还挂着的一个后台验收（云端固定命令验收很慢：该实例 ≈1 s/env-step）。
  * **消融 2×2 的结论（DEF-040，10k 同口径）**：`(0,0,0)` 的 `err_vel_xy`
    旧代码 0.1543 / **只加强扰动 0.0968（−37%）** / 只改奖励 0.1361 / 两样都改 0.1074。
    ⇒ **静止漂移的改善主要来自"加强扰动"**。步态那栏有个**反直觉发现**：`abl_pushonly`
    用的是**旧镜像惩罚**（已与 main 的 `params/env.yaml` 逐字段核对），步态却也已经对称
    ⇒ **`joint_mirror` 符号 bug 不是"右后腿撇"的唯一根因**（DEF-027 的归因要降级）；
    未控制的疑似因素：`HeightInvariantEECommand.reset()`（⑫ 修复）。要彻底归因就再加一根
    消融轴 `main + ⑫ only`（10k ≈8 h）。
  * **2026-10-02 22:20 CST：两件新事**
    - **第四根轴已跑完并结案**：`abl_legacyall_10k` = 三项全退 + 只留 ⑫ ⇒
      **"右后腿撇"的根因是 ⑫**（膝差 **1.156 → 0.008 rad**），不是镜像符号（DEF-027 降级）；
      `(0,0,0)` 速度误差 **0.1543 → 0.0916**（−41%，四格最好、摔倒率 0）。
      ⚠️ 口径提醒：`eval_fixed_command.py` 不关 push ⇒ "摔倒率"列跨格不可比。
    - 新长跑在跑：**`cloud_slowvx20k`**（1237，多地形 + v **x 课程台阶 ×2** = 150k/200k/250k/300k，
      cap=1.2）—— 10-02 22:18 时 **10801/20000**、7.04 s/iter、ETA ≈18 h；
      目的：验证"把 v_x 课程推迟一倍能否让地形等级不回退"（DEF-040 §2）。
      新任务 id：`Rough-Slopes-SlowVx-History-Adaptation-Deeprobotics-M20-v0`。
      跑完后照旧：`scp` 回来 → `summarize_run.py` 看 `Curriculum/terrain_levels` 轨迹。
    - 291 现在**空闲**（legacyall 已跑完，GPU 0%）⇒ 可以关机或安排新 run。
  * **从本机免密 ssh/scp 进 autodl 的可用方法**（本机没有 sshpass/plink/paramiko，也没配公钥）：
    用 OpenSSH askpass 把密码从环境变量喂进去（密码不落盘）——
    `$env:CODEX_SSH_PW='<密码>'; $env:SSH_ASKPASS=<一个只 echo %CODEX_SSH_PW% 的 .cmd>; $env:SSH_ASKPASS_REQUIRE='force';`
    之后正常 `ssh -p <port> -o PreferredAuthentications=password -o PubkeyAuthentication=no ...` /
    `scp -P <port> ...`。实测拉 190 MB 只要 3.9 s。**用完把那个 .cmd 删掉。**
  * **读数陷阱**：Jupyter `/api/contents` 的 `last_modified` 是 **UTC**（`ls --time-style=full-iso` 才是本机时区）；
    判"跑没跑完"别用 `tail`（正在长大的日志 tail 可能拿到启动 banner），要用
    `grep -ac "Starting the simulation"`=1 + `grep -ao "Learning iteration [0-9]*" | tail -1`=19999
    + `ls | grep -c "^model_"`=41 + `tr -dc "\0" | wc -c`=0。
  * 纪律：不超过 4 台；**用完关机**；跑完把 run 拿回来（`scp`）或就地分析。

【文档约定】docs/review/ 只保留 TODO_zh.md / DONE_zh.md / DEFECT_LOG_zh.md / NEXT_SESSION_PROMPT.md
  + templates/。每次改动**必须**：更新 TODO/DONE 的"更新记录"（加日期）、给新缺陷/特性在
  DEFECT_LOG 里加一条（骨架见 templates/DEFECT_ENTRY_TEMPLATE_zh.md）。
  新增条目：DEF-026（静止伫立）、DEF-027（镜像符号）、DEF-028（扰动加强）、DEF-029（多地形）、
  DEF-030（遥操 history）、DEF-031（本机跑不了生成地形）、DEF-032/033（云端 autodl 接力）、
  DEF-034（本机收尾批）、DEF-035（P2 action term 收敛）、DEF-036（P3 EE 锚点一键化）、
  DEF-037（星号导入遮蔽核实与收口）、DEF-038（云端收割 + 全长 20k 定稿）、
  DEF-039（`max_noise_std` 默认 1.2 定稿）、DEF-040（多地形 20k + 消融 2×2 归因）。

【命令备忘】
  # 训练（低层主线）
  python scripts/reinforcement_learning/rsl_rl/train.py \
      --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 4096 --seed 42
  # 课程阶段（num_steps_per_env=24）：s0 <1042、s1 <2083、s2 <3125、s3 之后
  # 固定命令 eval（跨 checkpoint 比较唯一的合法口径；--task 换成 -play-v0 就是完整难度）
  python scripts/reinforcement_learning/rsl_rl/eval_fixed_command.py \
      --headless --num_envs 512 --steps 1100 --seed 42 --commands "0,0,0;0.5,0,0;1.0,0,0" \
      --task History-Adaptation-Deeprobotics-M20-v0 \
      --checkpoint <run>/model_1000.pt --out logs/smoke/eval_<tag>.json
  # 步态对称性（每腿关节角均值 / 4 个镜像对 RMS / 足端 body 系坐标 / 左右不对称度）
  python scripts/reinforcement_learning/rsl_rl/probe_gait_symmetry.py \
      --headless --num_envs 256 --steps 600 --warmup 150 --commands "1.0,0,0" \
      --checkpoint <run>/model_1000.pt --label <tag> --out logs/smoke/gait_<tag>.json
  # EE 锚点 4 组对照（full/default/low/cfg；每组一个独立 Isaac 进程，512 envs × 1000 steps ≈ 3~4 min/组）
  python scripts/reinforcement_learning/rsl_rl/sweep_ee_anchor.py \
      --task History-Adaptation-Deeprobotics-M20-v0 \
      --policy <run>/exported_deploy/policy.pt --num_envs 512 --steps 1000
  # 训练曲线分析（不启动 Isaac；走 <run>/.summary_cache.npz，很快）
  python scripts/reinforcement_learning/rsl_rl/summarize_run.py --run <runA> --baseline <runB> \
      --tags Train/mean_reward --tags Episode_Termination/bad_orientation_2
  # ⚠️ summarize_run 对"短 run vs 长 run"会 hold-last 对齐，两个 run 长度差很多时别用它下结论
  # 冒烟回归（每个改动都要跑；退出码用 `cmd *> log; $LASTEXITCODE`）
  python scripts/reinforcement_learning/rsl_rl/train.py --task <task> --headless --num_envs 64 --max_iterations 2
  # 冒烟回归矩阵（一次跑全套 + 卡死判 SKIP；--env / --hydra 可透传给子进程，用来换低层 checkpoint）
  python scripts/reinforcement_learning/rsl_rl/smoke_regression.py --num_envs 64 --max_iterations 2 \
      [--env RL_TRAINING_LOW_LEVEL_POLICY_WBC=<policy.pt>] [--hydra env.actions.x.y=z]
  # 导出部署态策略（默认写 <run>/exported_deploy/：policy.pt + policy.onnx + policy_layout.json）
  python scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py --run <run> --checkpoint model_19999.pt

【踩坑备忘（累计，重点看新的几条）】
  * **回报口径**（本轮最大的坑）：`RewardManager` 返回 `Σ term·weight·dt`，而
    `Episode_Reward/*` 记的是**每秒速率**（episode 积分 / max_episode_length_s），
    `Train/mean_reward` ≈ 20 × Σ Episode_Reward。**定新惩罚的权重前，先用 `Episode_Reward`
    的实测值反解**（rate/weight = 该门控子集上的物理量均值），并把新惩罚的合计与同一步的
    Σ Episode_Reward 比一比 —— 我第一次按"1.75/s"估，低估了一个数量级，直接把策略教会了冻轮子。
  * **门控要看门控子集**：惩罚只对有零速命令的 env 计分（早期 2%、爬升后 15%），
    所以 `Episode_Reward` 是"全 batch 均值"，要乘/除那个比例才是单 env 量级。
  * **同代对照是唯一能说明问题的对照**：本仓库的 run 都是每 500 iter 存盘，
    比"1500 iter 的新代码 vs 20000 iter 的旧代码"毫无意义。
  * **本机跑不了生成地形**（DEF-031）⇒ 多地形的一切都去云端做（DEF-032）；
    云端 `Rough-Slopes-*` 已 2-iter 冒烟通过并开跑，本机的 `Rough-*` 仍未冒烟。
  * 课程的 print 要节流：curriculum 在每次 episode reset 都被调用（4096 envs 时 ~4~8 次/env step），
    逐次打印会在 25k 步里刷出几万行（`ramp_command_param` 已按"变化 ≥1% 才打印"节流）。
  * `episode_length_buf == 0` 在复位后的整步内都为真；判"刚复位"要用"相比上一次 tick 变小"。
  * **hydra 覆盖 `float` 字段要写小数**：`max_noise_std` 默认改成 1.2 之后，`agent.policy.max_noise_std=0`
    会被 `update_class_from_dict` 拒（hydra 把 `0` 解析成 int，报 `Expected: <class 'float'>,
    Received: <class 'int'>`）⇒ 必须写 **`0.0`**（DEF-040 §5）。
  * 想"在复位时做点什么"用基类的 `_on_reset(env_ids)` 钩子（`ActionManager.reset` 会转发）——
    实测那一刻 `robot.data` **已经是复位后状态**（`write_root_pose_to_sim` 会把 body 缓存
    timestamp 置 -1）；比旧写法（`apply_actions` 里看 `episode_length_buf == 0`）早一个 env step。
  * 课程只在 **env reset** 时推进；关掉终止/超时做实验时课程不会动。
  * `Metrics/*` 是"复位那一刻"的均值，单点抖动很大；判趋势用 summarize_run 的阶段均值。
  * **关节顺序有三套**：动作序 12 腿(fl,fr,hl,hr)+4 轮；articulation 原生序（四个 hipx → arm1 →
    四个 hipy → …）；MuJoCo MJCF 序（每腿连续）。一律按**关节名**映射（探针里就是这么写的）。
  * `find_bodies/find_joints` 返回的是**按 articulation 顺序**排的，不是传入顺序 ⇒
    要"每腿一个"就必须逐条解析（否则下标会错位）。
  * 本机 .git 只读；git 需要 `-c safe.directory=...` 且 escalate。
  * 文档是 CRLF（core.autocrlf=true）：用 apply_patch 改多行上下文时容易匹配失败，
    建议一条一条改、或先确认目标行的行尾。
```
