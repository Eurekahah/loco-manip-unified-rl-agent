# 机械臂鲁棒 IK 配置说明

本分支把原来的“单步无约束 DLS 直接写 PD 目标”拆成两层：

1. **高频执行层**：带位置/姿态权重、自适应阻尼、关节单步上限和关节限位投影的 DLS。
2. **目标求解层**：在新目标、大跳变或跟踪失败时，从多组种子做完整位姿 IK；完整位姿失败后可退化为位置优先。

实现位于：

- `source/.../locomotion/velocity/mdp/ik.py`：纯 PyTorch 的投影 DLS、姿态误差和多起点求解。
- `source/.../locomotion/velocity/mdp/actions.py`：与仿真 articulation、URDF 链和控制周期对接。

## 默认行为

- 关节限位在 DLS 输出时直接投影，不再先输出超限目标再依赖 PD 硬顶。
- 每控制周期关节目标变化不超过 `max_joint_step`。
- 单步目标相对实测关节角最多超前 `2 * max_joint_step`，避免 target wind-up。
- 新目标首次使用完整 IK；当前/上一解/关节中点/四分之一点/随机种子作为多起点。
- 完整 6D 位姿不可达但位置可达时，自动切换到位置优先，并锁定一个可行姿态。
- 位置和姿态都失败时设置 `tracking_blocked`，保持最近可行关节构型；之后按
  `full_solve_interval_s` 周期重试。
- 连续 `stall_patience` 个控制 tick 没有降低加权误差时，也会进入失败锁存。

## 关键参数

`CommandDrivenIKActionCfg.robust_ik` 控制全部行为。M20 配置已经指向
`M20_Piper_own.urdf`，其他配置也可以显式传入 `urdf_path`；不传时仍可使用投影
DLS，但不会启用多起点完整求解。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `position_tolerance` | `0.01 m` | 位置成功阈值 |
| `orientation_tolerance` | `0.08 rad` | 姿态成功阈值 |
| `step_gain` | `0.45` | 每 tick 使用 DLS 解的比例 |
| `max_joint_step` | `0.05 rad` | 高频层每个控制 tick 的单关节最大步长 |
| `damping_min/max` | `0.002/0.15` | 自适应阻尼范围 |
| `full_solve_iterations` | `20` | 每个多起点种子的最大迭代数 |
| `num_restarts` | `1` | 额外确定性随机种子数 |
| `full_solve_interval_s` | `1.0 s` | 失败后的完整 IK 重试周期 |
| `joint_limit_margin` | `0.01 rad` | 投影时使用的硬限位内缩量 |

## 调试

打开配置项 `robust_ik.debug=true` 后会打印完整 IK、位置退化和不可达计数。运行时也可读取：

```python
action_term.robust_ik_stats
action_term.tracking_blocked
```

需要临时恢复旧行为时，可覆盖：

```text
env.actions.ee_ik.robust_ik.enabled=false
```
