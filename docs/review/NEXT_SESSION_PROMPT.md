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
  * **要看策略表现就别再翻那 4 个老 test 脚本了**：`gait_test`/`torque_test`/`tracking_test`/
    `test` 已被 **`scripts/reinforcement_learning/rsl_rl/policy_report.py`** 取代（DEF-041）——
    一次滚动出 10 个角度（跟踪/步态/关节/力矩+峰值因子/对称/姿态/臂 EE/地形点云/A-B 对比）
    + `report.md` + `summary.json` + `data.npz`：
    `python scripts/reinforcement_learning/rsl_rl/policy_report.py --headless --task <task>
    --checkpoint <run>/model_19999.pt --compare <另一份>/model_19999.pt --commands "0,0,0;0.5,0,0;1.0,0,0"
    --num_envs 32 --steps 200 --out-dir logs/smoke/report_x`
    （地形任务用小环境数 1~16；脚本会临时加一个**只用于诊断**的 height_scanner 才能画地形点云；
    力矩限幅元数据在 IsaacLab 里是 1e9 占位值 ⇒ 报告用"峰值因子"代替。）
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

【2026-10-06 收尾：最新状态 + **未处理清单**（照这个往下做）】

> **2026-10-06 晚更新**：**A 组 4 条已全部做完**（A5/A6/A7/A8 → DEF-050~053，见下面 A 节里
> 每条的 `[x]` 与实测数字）——本机实测、**没有重训**。**B 组 5 条仍未动，
> B1（用修好的扰动课程重跑 20k + `--push-sweep` 验收）最高优先。**

* 最新提交：**`7a52b9c`**（`c50bd86` = DEF-049 的功能修复；`da1c228`/`7a52b9c` 是
  臂表口径修正 + 文档），分支 `codex/ll-train-detail-fix`。
  **7 条云端长跑都已跑完并拉回本机**：`cloud_soft20k`(20k) / `cloud_cap12_20k`(20k) /
  `cloud_roughslopes20k`(20k) / `cloud_slowvx20k`(20k) / `abl_pushonly_10k` /
  `abl_rewardonly_10k` / `abl_legacyall_10k`(10k)。
  云端实例 `bbc64d91a6-99f1820e`（**ssh -p 1237 root@10.60.144.11**，密码问用户/看旧记录）
  **运行中且空闲**（已同步到 `c50bd86`），其它实例在控制台里都是「已关机」。
* 两份**新报告**（第四轮工具 + DEF-049 修复，均已拉回本机）：
  `logs/smoke/report_flatAB_new/`（平地 cap12 vs 旧代码 + push 扫描 + 臂负载）
  `logs/smoke/report_terrain_new/`（多地形 SlowVx，256 envs / `--terrain-grid keep`）。
  旧报告目录已删；**旧 npz 不要再用**（fig03 的 pitch 符号是错的，见下）。

━━━ A. 报告/工具侧（**不需要训练**，本机就能做）━━━

* [x] **A1 fig03 的 pitch 画反**（2026-10-06 已修）：采集时用投影重力反解，`asin(-g_b[0])`
  与实际俯仰差一个负号（实测 corr = **−0.843**）；已统一改成
  `math_utils.euler_xyz_from_quat(root_quat_w)`（与 `body_pitch/roll_tracking` 奖励同口径）。
  重采后 corr = **+0.843**、pitch 稳态误差 cap12 **1.68°** vs 旧代码 5.54°。
  **注意**：老 npz 里存的还是旧符号，**光重画救不回来**（要用就得重新采集）。
* [x] **A2 fig03 右列"空一截"**（同日已修）：前 13 段是纯速度段（没有姿态指令），
  旧 `seg_metrics` 只在"该段有姿态指令"时才算误差 ⇒ NaN。现在一律对着**该段真正生效的
  `body_cmd`** 算（速度段 = 重置时采样的站姿），20 段都有数据。
* [x] **A3 限幅元数据"不可用"**（同日已修）：以前读 `robot.data.joint_effort_limits`（全是
  1e9 占位值）；现在从 **actuator 实例**读 `effort_limit/velocity_limit` +
  `robot.data.joint_pos_limits`（轮的 ±inf 逐关节置 NaN）。实测腿 76.4 / 轮 21.6 / 臂 100 / 夹爪 10 N·m。
* [x] **A4 fig08/报告新增臂诊断**（同日已做）：fig08 右下角 = "饱和时间占比 vs 顶限位时间占比"；
  `report.md` 新增《机械臂负载与限幅》表（逐关节 |tau| 均值/峰值、限幅、饱和%、位置范围/限位、
  顶限位%、|qd| p99、速度限幅、超速%）。
* [x] **A5 臂负载表改全 env 口径**（2026-10-06 完成，DEF-050）：`EpisodeData.arm_pop`
  （`ARM_POP_KEYS`）逐环境累加，`|qd|` p99 走 ≤600 点/env 子采样；`arm_joint_stats` 优先用它、
  老 npz 自动退回旧口径；报告表下加"全部 N env × M 步"脚注。本机 64 envs × 900 步实测：
  joint2 顶限位 **21.3%**、joint4 饱和 46.3%、joint5 饱和 53.2%（顶限位 65.0%）；
  **同一份数据里 env0 单看**是 joint2 顶限位 0%、joint4 饱和 81.2% ⇒ 单条轨迹会漏也会放大。
  产物 `logs/smoke/a5_full/`。
* [x] **A6 `[IK DEBUG]` 刷屏**（2026-10-06 完成，DEF-051）：整块删除
  （`D:\nvidia-isaac-sim\IsaacLab-5.1.0\...\task_space_actions.py`，**依赖侧文件、不在本仓库**），
  同一条报告命令 `grep -c "IK DEBUG"` **15 → 0**；保留上面的 `logger.info`。
* [x] **A7 高速命令误差被"复位"污染**（2026-10-06 完成，DEF-052）：新增 `--reset-grace N`
  （默认 **25**，0 = 关闭），均值类指标剔除复位后前 N 步，`done`/峰值保持原口径；
  报告第 1 节新增 `稳态占比` 列 + 采集时打印剔除比例（60 步档 → 43.3%、100 步档 → 26.0%）。
* [x] **A8 fig04/fig11 只画 A**（2026-10-06 完成，DEF-053）：fig04 每个 label 占一组 4 行、
  fig11 按 (label, 命令) 分组画柱（B 浅色 + 斜纹）；`--from-npz` 合成 2 label npz 渲染通过
  （fig04 855×1710、fig11 1350×675，无 WARN）。

━━━ B. 训练 / 物理侧（要改代码 + 重训）━━━

* [x] **B1 用修好的扰动课程重跑一条 20k —— 2026-10-08 已完成并验收（DEF-059）**：
  `apply_event_scale` 的 `EventManager.active_terms` 判断 bug 让 `disturbance_ramp`
  **从上线起就是空操作**（push 第 0 步就全量 ±2/±1/yaw±0.52）。
  * 已在 `bbc64d91a6-99f1820e`（1237）起：4096 envs / seed 42 / 20k /
    `--run_name cloud_ramp20k`，run 目录 `2026-10-06_15-32-05_cloud_ramp20k`，
    日志 `/root/run_ramp20k.log`，**ETA ≈16 h**（14:32 UTC 起跑）。
  * **开跑即验证了一条**：`[curriculum] 扰动 randomize_push_robot.velocity_range 缩放到 0.20×
    (step=0)` —— 旧三个 run（soft20k/cap12/slowvx）的日志里这类打印**都是 0 次**。
  * **验收结果（两条都拿到了）**：① 前 2000 iter `root_height_below_minimum`
    **0.0386 → 0.0195（−49%）**、`bad_orientation_2` 0.0724 → 0.0521；
    ② **但 3× 外推抗扰变差**：终止/环境/分钟 **5.16 vs 2.70**（2× 也差：0.88 vs 0.29），
    1× 反而更好（0.12 vs 0.00，最大瞬时误差 1.29 vs 1.62）。
    ⇒ **渐进扰动不是无条件更好**（早期吃满扰动的旧 run 学到了更保守的抗扰姿态）。
    候选后续：课程爬到 1.5× 再回落 / 最后 20% 恢复满强度扰动 / 维持现状但把抗扰纳入选型。
  * run 已 `scp` 回本机（`logs/rsl_rl/history_adaptation/2026-10-06_15-32-05_cloud_ramp20k`，42 文件/300 MB），
    云实例**还开着**（空着，随时可安排下一批）。
* [ ] **B2 机械臂 5 条改进**（DEF-049 的结论，按性价比排序）：
  1. [x]（**已实现但默认关**）`_resample_ee_goal*` 加**可达性过滤**——
     现在只查笛卡尔碰撞盒 + 地面高度，会采到关节超程的目标
     —— **2026-10-06 完成（DEF-055）**：新增 `build_reachable_grid()`
     （`pytorch_kinematics` 建 `arm_base_link→gripper_base` 链，20 万次关节采样 FK →
     **1.5 cm 体素占用栅格**，膨胀一格），在 `_resample_ee_goal` 的重采样循环里与碰撞检查
     并联；`urdf_path` 给了才启用、建不起来**直接报错**（不静默降级）。
     实测 joint4 饱和 **47.7%→10.7%**、超速 **93.4%→17.0%**、`|tau|` 均值 71.8→28.4；
     joint1/2/3/6 饱和基本清零。`reach_joint_margin=0.1` 试过、**没帮助**（默认 0）。
     ⚠️ **2026-10-08 补测（DEF-060）：过滤有实测副作用**——EE 位置误差 5.87→**15.20 cm**、
     姿态 35°→**96°**、joint2 贴 0 限位 94%（"位置可达"≠"当前构型够得到"，局部 IK 收敛不到）；
     而且"过滤 + 无位置 clamp"更糟（joint2/3/5 力矩 87 N·m）。⇒ **已改成默认关**
     （`rough_env_cfg.py` 里把 `urdf_path` 注释掉，留一行开关），等"同一目标序列"或
     pick 成功率的受控验收再决定。若将来要开，**必须同时开着位置 clamp**。
  2. [x] IK 输出加**关节限位 clamp**（`DifferentialIKController.compute` 返回的是
     `joint_pos + delta`，下游只按 effort 裁剪）⇒ 让"到不了"表现为停在限位而不是硬顶 100 N·m
     —— **2026-10-06 完成（DEF-054）**：`CommandDrivenIKAction.apply_actions()` 里加
     位置 clamp（默认开）。实测 joint1/2/3/5/6 的饱和与顶限位全面下降
     （joint5 顶限位 65.0%→25.6%、joint6 30.2%→1.0%、joint5 `|tau|` 均值 76.5→55.3）；
     joint4 基本持平（46.3%→47.7%）⇒ 它是"不可达目标"的主犯，留给第 1 条。
  3. [x] **用户 2026-10-08 决定：不做**（"不用把臂跟踪精度作为奖励项"）⇒ 本条关闭；
     "停在限位"这条路已由位置 clamp 落实并做过副作用测试（DEF-054/DEF-060）。
     原方案留档：把 `arm_ee_pos_tracking`/`arm_ee_ori_tracking` 加进 **WBC 奖励表**
     （现在 `WBCRewardsCfg` 里**没有**这两项、`params/env.yaml` 可查），并把姿态 `std`
     从 0.5 rad（≈29°）收紧——否则奖励早饱和、梯度≈0；
  4. [x] 夹爪 `stiffness=4000` 配 ±0.035 rad 行程 / 10 N·m 限幅 ⇒ 误差 >0.0025 rad 就顶满
     （实测 100% 时间在行程端、饱和 95%+）；把刚度降到匹配量级；
     **2026-10-08 完成（DEF-058）**：扫描 5 组后取 **`stiffness=286.0 / damping=5.0`**
     （= 10 N·m ÷ 0.035 rad，"满行程误差刚好顶到限幅"）⇒ 饱和 **88%/84% → 0%/0%**、
     `|qd|` p99 0.52（限幅 1.0）；已改 `assets/deeprobotics.py`，三个 pick/WBC 冒烟 EXIT=0。
     **遗留**：抓取成功率待复验（本机跑不了完整 pick 回合）。
  5. [~] sim2real：臂 `velocity_limit=3.0` 在 `DelayedPDActuator` 里**只参与力矩裁剪、不限速**
     （实测腕关节 |qd| p99 到 5.0 rad/s，超速时间占比 72~97%）⇒ IK 层限速或加进保护逻辑。
     **2026-10-06 做了并做了 A/B/C/D 消融：结论是"只夹 IK 目标"这条路不通**（DEF-054）——
     目标被限速后永远追不上（不可达目标 + 限速 = 常驻跟踪误差），`|tau|` 与 `|qd|` 反而更大
     （joint4 `|tau|` 均值 69.9→88.8 N·m、超速 84%→99.5%）。所以 `max_joint_vel` 已做进
      cfg 但**默认 -1.0（关）**，留作旋钮；真要限速得从力矩/轨迹层做，或**先做第 1 条**。
     **2026-10-08 补**：joint5 的 `|qd|` p99 仍是 5.0 rad/s（默认关着限速）⇒ 仍是遗留项。
  * 现状数字（cap12 / 旧代码，18 s 臂测试）：joint2 顶上限 3.140 rad 占 **26% / 33%**、
    joint5 顶下限占 **25% / 61%**、joint6 顶下限占 **55% / 60%**；
    `|tau|≥99 N·m` 时间占比 joint4 **41%/55%**、joint5 33%/82%、joint6 61%/61%；
    EE 稳态误差：位置可到 18 cm、姿态 55°~99°（且那几段腕关节 100% 在饱和）。
* [x] **B3 坡面短板：已归因（2026-10-08，DEF-061）** —— 先纠正前提：**短板不是"下坡"**，
  而是"**上坡 + 低/零速**"。256 envs 逐地形实测（`logs/smoke/report_slope_analysis/`）：
  上坡 `(0,0,0)` 的速度误差是平地 **4.4×**（0.2257 vs 0.0513）、**高度抖动 24×**（std 0.0496）、
  俯仰 σ 2.41°（平地 0.15°）、力矩 +37%、**终止 0.094**（其它地形全 0）；
  而下坡 `(0,0,0)/(0.8,0,0)` 已经和平地一个量级（0.0629 / 0.1382）。机理：
  **"静止伫立"惩罚按零速命令门控惩罚轮子转动 ⇒ 在坡上等于锁死轮子 ⇒ 下滑—补—滑振荡**
  （高度 std 爆表、平均俯仰几乎不偏），且轮式方案缺"刹车/牵引"通道（最差腿触地 0.176 vs 平地 0.937），
  终止全是 `bad_orientation_2`。
  **待做（修法，按性价比）**：a. 静止惩罚**按坡度门控**（或改成"奖励不滑动"而非"不转轮"）；
  b. 做"**只放上坡**"的地形变体跑 2~10k 验 a 是否对症；
  c. 若 a 不够再单独放宽上坡的 `track_lin_vel_xy_exp` std。
* [x] **B4 把 SlowVx 配方落成默认**（2026-10-08 完成，DEF-056）：四个 v_x 台阶
  （150k/200k/250k/300k 环境步）已写进 `RoughSlopesEnvWBCConfig`；
  `RoughSlopesSlowVxEnvWBCConfig` 变**别名**（不再二次 ×2，老任务名继续可用）、
  `-play-` 变体不变；云端多地形 2-iter 冒烟 EXIT=0。
  复现旧配方（75k/100k/125k/150k）需用 hydra 覆盖 `num_steps`。
* [ ] **B5 其他老账**（都在 TODO_zh.md）：s3 臂摆动鲁棒性、执行器刚度课程、
  低层 known_issues ⑭⑮、`mdp/__init__.py` 星号导入遮蔽（影响面小）、
  `vr_extented` 的"无超时线程"（已评估降级）。

━━━ C. 已定稿、**不要再重跑验证**的结论 ━━━

  ① `max_noise_std` 默认 **1.2**（DEF-039）；
  ② "右后腿撇"的根因是 **⑫**（`HeightInvariantEECommand.reset()`），不是镜像符号（DEF-040 §3④）；
  ③ 静止漂移/步态/抗扰：`cap12_20k` 全面最好；固定命令 eval 三档 0.1014/0.1301/0.1594、摔倒率≈0；
     新报告里九档命令（含 vy/wz）也是**全胜**旧代码，push 外推 3x3 摔倒率 **3.23 vs 7.10** 次/env/min；
  ④ **SlowVx（v_x 课程台阶 ×2 + cap1.2）治住了地形等级回落**：末 1000 `terrain_levels`
     3.706（−0.09/1k）→ **4.74（+0.095/1k）**、`bad_orientation_2` −51%、回报 +33%（DEF-044）；
  ⑤ 多地形分地形（新表按"地形 × 命令档"）：粗糙 0.0916 ＜ 下坡 0.1117 ≈ 平地 0.1128 ＜ 上坡 0.1228；
     平均终止上坡 **0.17** ＞ 平地/粗糙 0.04 ＞ 下坡 0.00；最差腿触地占比 **平地 0.750 / 上坡 0.125 /
     粗糙 0.100 / 下坡 0.043**；
  ⑥ 臂是 **IK 直接驱动**（`ee_ik` 的 `action_dim = 0`、策略动作只有 16 维），
     WBC 奖励表里**没有臂跟踪项** ⇒ 臂的残余误差不是"策略没学好"。

━━━ D. 环境/流程提醒（反复踩）━━━

  * 本机（Windows + A4000）**看/测多地形没问题**（`play.py` 会把地形自动压成 5×5 + 关课程，
    见 `play.py:101-104`；`policy_report.py` 用 `--terrain-grid auto/5x5` 同理）。**会卡死的只有一种组合**：
    训练任务 + 训练级网格（10×20）+ 环境数 ≥48（env 创建阶段挂住）⇒ 只有"多环境地形训练 /
    256 envs 大样本分地形统计"才必须上云（DEF-046 更正了原先过宽的说法）。
  * **本机采集速度只有 ~3~6 步/秒**（8~16 envs）⇒ 全套默认报告（≈8.6k 步）本地要 ~25 min；
    长测试一律放云端 3090（本机只做小规模抽查，`--steps 20 --warmup 10 --schedule none` 这种）。
  * 云端脚本要加 `python -u`（或 `PYTHONUNBUFFERED=1`），否则 `[report]` 的进度行会被块缓冲、
    看起来像"卡住"（本次踩过）。
  * **v_x 命令范围**：训练四个台阶 ±2→±3→±4→±5（75k/100k/125k/150k 环境步 ≈3125/4167/5208/6250 iter）；
    SlowVx 时间表 ×2（终值仍 ±5）；**`-play-v0` 固定 (-1,1)**（DEF-046）。
  * 图已全英文 ⇒ **云端不再需要 `POLICY_REPORT_FONT`**；要"云端采集 + 本机画图"仍可只拉
    `data.npz`+`meta.json` 用 `--from-npz`。
  * 云端 SSH 偶发 `banner exchange` 超时（实例被大环境数地形生成压住）——**不是关机**，
    控制台 `private.autodl.com/console/instance` 看状态是「运行中」就等几分钟重连。

━━━ E. 云端仓库当前状态（2026-10-06 收尾时）━━━

* 实例 `bbc64d91a6-99f1820e`（1237）里仓库在 `c50bd86`；**`git pull` 当时连不上 GitHub
  （`GnuTLS recv error (-110)` / `port 443 timeout`）**，所以 `da1c228` 之后的
  `policy_report.py` 是**直接 scp 覆盖**过去的（md5 `efff749ad841bb2047ed4f1b3ae16432`，
  与本机一致），但 git 状态显示为 `M`（未提交）。⇒ **下次开机先 `git pull` 重试**
  （通了就 `git checkout -- scripts/.../policy_report.py` 再 pull，或 `git stash` 掉这一份），
  没通也能继续用（文件内容是对的）。
* 该实例里还有一个**历史 `stash@{0}`（WIP on wbc）**，是早先 session 留下的，别误删。
* 云端产物目录：`/root/report_flatAB_new`、`/root/report_terrain_new`（都已拉回本机）；
  `/root/report_slowvx_fast` 已删（旧口径）。
```
