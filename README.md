# TRON2A SFYG 平地训练

基于 mjlab 的基础速度跟踪环境，仅包含 TRON2A 的 SFYG 形态。

| 任务 | Actor / Critic 观测维度 | 动作 |
| --- | --- | --- |
| `Mjlab-Velocity-Flat-TRON2A-SFYG-LBC-CTS` | Actor 历史 5×56 / Critic 68 | 10 个腿关节位置目标 |
| `Mjlab-Velocity-Flat-TRON2A-SFYG-Baseline-CTS` | Actor 历史 5×44 / Critic 56 | 10 个腿关节位置目标 |

实现见 [模型适配](src/tron2_mjlab/robot.py) 和 [环境配置](src/tron2_mjlab/env_cfgs.py)。
机械臂保留 6 个关节自由度，由独立位置执行器控制，不参与策略动作；LBC 观测机械臂状态，Baseline 不观测。
训练时每次重置会采样机械臂姿态，并同步设置关节状态与 PD 目标；回放默认以原 XML 零位为目标。
训练期间，PD 跟踪持续更新的平滑随机机械臂轨迹；回放未接入外部控制时保持零位，并非刚性固定。
夹爪仍固定在零位；机械臂与夹爪的外观、质量和碰撞几何保留。
仅使用平面地形，无高度扫描或地形课程。

两组任务均启用**左脚与右脚之间**的自碰撞，范围仅限两个 `ankle_pitch` 脚部的碰撞几何。
双脚及其他身体部位与地面的碰撞保持不变，腿与腿、脚与小腿、机械臂与躯干等其他自碰撞仍关闭。
足部接触奖励和非法接触终止仍只检测地面，不会把踩到另一只脚算成地面支撑。
另加入 `feet_too_close` 惩罚，在物理接触前引导策略保持足部间距。
此改动不会修改已运行训练进程中的模型；需重启训练或回放才生效，旧模型应在新碰撞条件下重新评估。

两组任务均以膝盖前倾姿态初始化：左髋 yaw 为 `-pi`（−180°），右髋 yaw 为 `+pi`（+180°），
符号遵循原模型左右镜像的关节限位。髋 pitch、膝、踝 pitch 分别保持 `0.3`、`-0.6`、`0.3` rad，
髋 roll 为零。重置、零动作位置目标、相对关节位置观测和姿态奖励均使用这一默认姿态。
执行器位置目标仍按原有机制扣除随机化的编码器偏置。
原始 XML 不改写；此前基于 yaw 零位训练的策略应重新训练。

## 策略观测与机械臂控制

LBC 的单帧 Actor 输入共 56 维，按下表顺序拼接；两组任务均使用 CTS，堆叠 5 帧历史：

| 观测 | 维度 | 含义 |
| --- | --- | --- |
| `base_lin_vel` | 3 | 基座 IMU 局部坐标系线速度 |
| `base_ang_vel` | 3 | 基座 IMU 局部坐标系角速度 |
| `projected_gravity` | 3 | 机身坐标系中的重力方向 |
| `joint_pos` | 16 | 10 个腿关节、6 个臂关节相对默认姿态的角度 |
| `joint_vel` | 16 | 同顺序的关节速度，默认速度为零 |
| `actions` | 10 | 上一步双腿策略动作 |
| `gait_phase` | 2 | 步态周期的 sin/cos，停步时均为零 |
| `command` | 3 | 目标前向、横向、偏航速度 |

臂关节按 `arm1_Joint` 至 `arm6_Joint` 排列，默认角度为零，因此对应观测即实际关节角度。
训练时对前五项添加观测噪声，回放时关闭。Critic 使用无噪声版本，再加双脚高度、
腾空时间、接触状态和接触力共 12 维，总计 68 维。
姿态与关节软限位奖励仅约束双腿，不要求外部控制的机械臂回到默认姿态。

未来 IK 控制器可在 `env.step(leg_actions)` 前下发机械臂关节位置目标：
若使用训练配置，须在创建环境前移除 `cfg.commands["arm_motion"]`，避免随机轨迹覆盖 IK 目标；
默认回放配置不包含该命令。

```python
from tron2_mjlab.robot import ARM_JOINTS

robot = env.scene["robot"]
arm_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)
# q_ik: shape (env.num_envs, 6), radians, on env.device, within joint limits.
robot.set_joint_position_target(q_ik, joint_ids=arm_ids)
obs, reward, terminated, truncated, info = env.step(leg_actions)
```

目标不被双腿策略动作覆盖；训练时由随机轨迹持续更新，训练重置时重新采样姿态，回放重置时回到零。
当前未实现 IK 求解器；随机关节轨迹不保证具有实际任务意义，PD 参数只是仿真起点。
训练时还需加入代表性的机械臂运动，才能评估策略对未来 IK 动作的适应性。
机械臂动力学和观测接口已改变，旧固定臂 checkpoint 不能直接使用，需要重新训练。
新增步态相位后，原无相位的 LBC（54/66 维）和 Baseline（42/54 维）checkpoint
也不能直接使用，应从头训练，不要用 `--resume` 加载旧输入维度的模型。

### 不观测机械臂的 baseline

Baseline 的 Actor 和 Critic 均只读取 10 个腿关节的位置、速度，不包含机械臂关节状态，
也不接收 IK 目标或末端状态。Actor 维度为 `3+3+3+10+10+10+2+3=44`，
Critic 再加 12 维足部状态，共 56 维。任务仍是本体速度跟踪，没有机械臂任务奖励。

两组使用相同的机器人模型、可动机械臂、独立位置控制、奖励、噪声、重置和 PPO 超参数，
区别仅在于是否直接观测机械臂状态。Baseline 仍会通过机身运动感受到机械臂运动带来的扰动，
不是删除机械臂质量或锁死机械臂。外部 IK 接口对两组都可用。
公平对比时应使用相同的机械臂运动、速度指令和随机种子。

两组 checkpoint 的输入维度不同，不能混用。

## 机械臂姿态分布与动力学随机化

训练时，LBC-CTS 和 Baseline-CTS 使用相同分布：

- **每次 episode 重置**，以 80% 概率为 6 个机械臂关节独立采样角度，
  范围为各关节软限位区间居中的 80%；以 20% 概率保留原 XML 零位姿态。
  这里按软限位区间中心采样，不是对零位加对称扰动，因此也覆盖具有单侧限位的关节。
- 将机械臂实际关节位置与 PD 目标设成同一组采样值，初始关节速度归零。
  不会从零位突然跟踪一个远处目标；只重置指定环境，其他环境的姿态和目标不受影响。
- **episode 内持续随机挥臂**：每个环境独立重复采样下一组 6 关节目标，
  仍使用软限位区间居中的 80% 范围和 20% 零位概率，不同步挥臂。
- 相邻目标用五次平滑插值连接，段首/段尾目标速度和加速度为零。
  每段先采样 **2～4 s**，再根据最大关节位移自动延长，使每个关节的**目标角速度不超过 1 rad/s**。
  这约束的是位置目标变化率，不是对实际关节速度的硬限幅。
- 每个策略步更新一次 PD 目标，到达旧目标后再采样下一段，不在 episode 内直接改写关节状态。
  环境重置后从该次初始姿态重新规划，不继承上一 episode 的轨迹；未重置环境继续各自的轨迹。
  该机制用于增加动态挥臂扰动与重心分布，不是 policy 的额外动作，也不向 Baseline 泄漏机械臂目标。
- **环境初始化时**，各环境独立采样 `arm1_Link`～`arm6_Link` 的质量倍率 0.8～1.2，
  每个连杆惯量使用与质量相同的倍率；6 个臂关节阻尼倍率 0.5～1.5，
  Kp/Kd 倍率分别为 0.8～1.2。双腿、躯干和夹爪物理参数不被这些事件修改。
- 质量改变后重算子树质量、参考逆惯量等派生量；动力学参数在重置时不重采样，
  避免逐 episode 频繁修改模型。所有倍率相对标称模型计算，不累计漂移。

回放默认关闭机械臂姿态采样、持续随机轨迹和动力学随机化，使用零位姿态和标称机械臂动力学；原有地面摩擦、
基座质心与编码器偏置设置不变。若要在回放中评估随机手臂，应显式使用训练事件配置，
并加入 `RandomArmMotionCfg` 命令，或通过上述 IK 接口下发目标，不能仅看零位回放来判断泛化效果。
配置位于 [环境配置](src/tron2_mjlab/env_cfgs.py) 的 `arm_pose`、`arm_dynamics` 事件及
`commands["arm_motion"]`；轨迹实现和速度参数见 [RandomArmMotionCfg](src/tron2_mjlab/arm_motion.py)。
使用外部 IK 训练时移除 `arm_motion` 命令，避免两套控制器争用机械臂目标。
臂部自碰撞仍未开启，关节范围内的样本不保证整条机械臂无几何穿插；实机前需加入碰撞约束。
修改需重启训练进程才会生效，观测/动作维度不变。

## Concurrent Teacher–Student（CTS）

项目只提供 `LBC-CTS` 和 `Baseline-CTS`，不再注册不带 `-CTS` 后缀的纯 PPO 任务，
也不再提供关闭 CTS 的环境配置选项。CTS 内部仍使用 PPO 更新教师与共享动作网络。
两种 CTS 使用相同的奖励、动力学、自适应步态和训练预算，机械臂观测区别保持不变。
这是一种 PPO 与潜变量蒸馏分步更新的 CTS 实现，不宣称逐项复现特定论文的优化器。

- **教师编码器**：使用当前任务的无噪声 Critic 观测，包含足部高度、腾空时间、
  接触状态及接触力，输出 64 维潜变量。Baseline 教师也不接收机械臂关节状态。
- **学生编码器**：使用该任务的 5 帧 Actor 历史（含当前帧），由旧到新排列。
  50 Hz 下相邻帧间隔 20 ms，最旧帧距当前 80 ms。Baseline 历史不含机械臂状态。
- 两个编码器均使用 `512/256` 隐藏层和 tanh 潜变量，共享 `512/256/128` 动作网络，
  输入为当前 Actor 观测与所选潜变量，输出仍只有 10 个腿关节动作。
- 默认 75% 并行环境走教师分支，其余走学生分支。角色随 rollout 一起保存，
  不因 PPO minibatch 打乱而改变。训练至少需要 2 个环境，小批量会保留至少一个教师和学生。
- PPO 更新教师、共享动作网络及 Critic，停止学生潜变量的 PPO 梯度；
  每次 PPO 更新后，在同一批 rollout 上以教师潜变量为监督，仅更新学生编码器。
  默认蒸馏系数 1.0、学习率 0.001、5 个辅助更新 epoch，记录 `latent_alignment` 损失。
- CTS 启用观测归一化，历史逐帧复用 Actor 归一化统计，教师使用独立特权观测统计。
  重置时历史用新 episode 的首帧填满，不读取上一个 episode 的状态。
- **回放和导出始终只用学生**，不需要 Critic 观测、教师编码器或角色标记。
  纯 PPO checkpoint 与 CTS checkpoint 不兼容，应从头训练；CTS checkpoint 保存两个编码器、
  PPO 优化器、学生优化器及归一化统计，支持同任务续训。

实验目录分别为 `tron2a_sfyg_velocity_lbc_cts` 和 `tron2a_sfyg_velocity_baseline_cts`，
历史纯 PPO 日志不删除或迁移。当前 CTS 支持单进程/单 GPU，不支持多 GPU 或对称数据增强。
参数定义见 [CTS 实现](src/tron2_mjlab/cts.py)，尚未验证长期收敛。

每次保存检查点会按 mjlab 的速度任务流程导出学生 ONNX；输入名为 `actor_history`，
形状为 `(1, 5, 56)`（LBC）或 `(1, 5, 44)`（Baseline），输出为 `(1, 10)`。
历史由调用者维护，顺序及单帧特征必须与训练一致；每个策略步推进一次，重置时用首帧填满。
归一化和学生编码器包含在导出模型中，不应在外部重复归一化。
原有 `runner.export_policy_to_jit(...)` 接口也导出相同的学生历史输入。

## 交替步态与速度范围

LBC 和 Baseline 使用同一套步态奖励与相位观测：

- `feet_too_close` 权重为 **−1.0**：以 `foot_L`、`foot_R` 参考点的水平 XY 距离 `d` 计算
  `max(0, (0.18 - d) / 0.18)^2`，距离低于 **0.18 m** 才产生惩罚。
  例如 0.09 m 时未乘权重的代价为 0.25，水平重合时为 1；大于等于阈值不奖励继续劈叉。
  站立、移动及摆动阶段都生效，忽略高度差，避免仅抬高一脚来绕过距离约束。
  这里是参考点距离，不是脚部几何边缘的净间隙，也不是严格的防交叉步约束。
  TensorBoard 记录 `Metrics/feet_distance_xy`（m）和 `Metrics/feet_too_close_fraction`（环境比例）。
  观测和动作维度不变；须重启训练，并继续训练或重新训练，旧策略不会自动学会保持间距。
- 每条腿周期根据**目标速度**在 **0.9～0.6 s** 之间调节，左右脚始终相差半周期。
  平移指令模长 ≤0.2 m/s 时目标周期为 0.9 s，≥1.5 m/s 时为 0.6 s，中间线性插值；
  后退与横移使用相同规则，不使用实际速度，避免跟踪不佳时步频随之降低。
- 原地转向单独映射：偏航指令绝对值从 0.05 增至 1.0 rad/s 时，目标周期从 0.9 缩短至 0.6 s。
  平移与转向同时存在时，选取两者要求的较快步频。
- 步频按 **0.2 s 时间常数**做指数平滑，实际周期逐渐接近目标周期。
  每个策略步以 `phase = (phase + dt * frequency) % 1` 连续积分，不使用 `时间 / 当前周期`，
  避免变速时相位跳变；重复读取观测、奖励或刷新调速面板不会推进时钟。
- 每脚支撑占比 **60%**、摆动占比 **40%**；每周期两段双脚支撑，各占周期的 10%，无计划腾空阶段。
- `gait_contact` 权重为 **1.0**，按左右脚实际接触是否符合当前支撑/摆动相位计分，
  两脚均匹配为 1，仅一脚匹配为 0.5，均不匹配为 0。
- 平面速度指令模长大于 **0.05 m/s** 或转向速度绝对值大于 **0.05 rad/s** 时启用交替，
  后退、横移、原地转向均支持。否则要求双脚接触，且相位观测置零，避免停步时奖励踏步。
- 每个环境独立维护时钟，重置后相位归零、步频回到 1/0.9 Hz，不影响其他环境。
  停步期间相位冻结，步频平滑回到慢速值；重新运动时从冻结相位继续。
  自动采样、手动调速、相位奖励和观测共享同一速度指令时钟与阈值。
- 原有离地高度、摆动高度（目标均为 0.1 m）、防滑和落地冲击项保持；
  `air_time` 权重仍为 0，不再额外叠加腾空时长目标。

速度指令范围相对原配置扩大两倍，训练采样、自动回放和手动调速滑块共用：

| 指令 | 原范围 | 当前范围 |
| --- | --- | --- |
| 前后速度（m/s） | [-0.5, 1.0] | **[-1.0, 2.0]** |
| 横向速度（m/s） | [-0.3, 0.3] | **[-0.6, 0.6]** |
| 偏航速度（rad/s） | [-0.5, 0.5] | **[-1.0, 1.0]** |

步态时钟参数定义在 [GaitClockCfg](src/tron2_mjlab/gait.py)，可通过环境的
`commands["twist"].gait` 配置；接触奖励与支撑占比位于 [环境配置](src/tron2_mjlab/env_cfgs.py)。
相较固定周期版本，观测维度不变，但时钟行为已改变，建议重新训练后对比评估。
这些是训练起点，不保证在完整速度范围内已经获得稳定步态；需重新训练并评估速度跟踪、
左右脚接触时序、滑移和跌倒率，不能直接用于实机。

## 安装

需要 Python 3.10-3.13、uv，以及驱动兼容的 NVIDIA CUDA GPU。
仿真依赖固定为 mjlab 1.1.0、MuJoCo/MuJoCo Warp 3.5.0、Warp 1.12.0。

```bash
uv sync --locked
uv run list_envs
```

若 PyPI 下载缓慢，可使用镜像：

```bash
uv sync --default-index https://pypi.tuna.tsinghua.edu.cn/simple
```

## 训练

```bash
uv run train Mjlab-Velocity-Flat-TRON2A-SFYG-LBC-CTS
uv run train Mjlab-Velocity-Flat-TRON2A-SFYG-Baseline-CTS
```

默认 1024 个并行环境，物理步长 2 ms，策略频率 50 Hz，10000 次 PPO 迭代。
训练和回放的 `opt.ccd_iterations` 均设为 100，提高凸碰撞检测的迭代预算。
修改该值需重启进程才能生效；如果仍出现迭代不足警告，应结合具体接触姿态进一步排查，
提高预算可能增加碰撞计算耗时。
使用 TensorBoard 本地日志，无需 W&B 登录。日志和模型位于 `logs/rsl_rl/`。
LBC 实验目录为 `tron2a_sfyg_velocity_lbc_cts`，Baseline 为 `tron2a_sfyg_velocity_baseline_cts`。
旧无 CTS 后缀的任务已移除，不能加载旧单帧 Actor checkpoint。
显存不足时使用 `--env.scene.num-envs 256`。

快速检查训练入口：

```bash
uv run train Mjlab-Velocity-Flat-TRON2A-SFYG-LBC-CTS --env.scene.num-envs 16 --agent.max-iterations 1
```

## 预览

### 手动调速面板

```bash
uv run tron2-play SFYG --checkpoint-file /path/to/lbc_cts/model.pt
uv run tron2-play SFYG --baseline --checkpoint-file /path/to/baseline_cts/model.pt
```

打开终端输出的 Viser 地址，在 `Velocity Command` 面板中调节 `Forward (m/s)`、
`Lateral (m/s)` 和 `Yaw (rad/s)`。默认启用 `Manual control`，初始速度为零；
取消勾选恢复自动随机指令。
`Stop` 将三个目标速度归零并切回手动模式，不是暂停仿真或实机急停。
原 `Simulation / Speed` 调整的是仿真播放速度，与机器人目标速度不同。

没有模型时可用 `uv run tron2-play SFYG --agent zero` 检查面板，但零动作策略不会跟踪速度。
手动回放默认使用 LBC-CTS 学生；检查 Baseline-CTS 时加 `--baseline`，加载其 checkpoint 也必须加此选项。
LBC checkpoint 的单帧观测为 56 维，Baseline 为 44 维，两者不能混用。若报 checkpoint 56 维而当前模型
44 维，应去掉 `--baseline`；反过来则加上该选项。回放在启动仿真前检查任务与历史输入维度，
不使用截断、补零或忽略权重的方式加载不匹配模型。选错任务不需要重新训练，改正回放选项即可。
`--cts` 参数已移除，CTS 始终启用；调速面板只刷新当前帧的指令和相位，不额外推进历史。
必须指定 SFYG 模型完整的 `.pt` 文件路径，不能只指定目录。

### 自动指令回放

```bash
uv run play Mjlab-Velocity-Flat-TRON2A-SFYG-LBC-CTS --agent zero --viewer viser
uv run play Mjlab-Velocity-Flat-TRON2A-SFYG-LBC-CTS --checkpoint-file /path/to/lbc_cts/model.pt --viewer viser
uv run play Mjlab-Velocity-Flat-TRON2A-SFYG-Baseline-CTS --checkpoint-file /path/to/baseline_cts/model.pt --viewer viser
```

零动作仅用于检查模型，不代表已经学会平衡。PD 增益、初始姿态和奖励是训练起点，
尚未验证长期收敛或实机部署，不应直接用于实机控制。

## 测试

```bash
uv run python -m unittest discover -s tests -v
```

覆盖两组 SFYG 任务注册、模型编译、平地配置、机械臂状态观测与独立目标、baseline 观测隔离、
机械臂姿态分布与目标同步、质量/惯量一致性、动力学随机化范围及腿部参数隔离、
episode 内随机轨迹的目标速度上限、重复换目标、轨迹连续性与部分环境重置、
仅双脚自碰撞的过滤、初始姿态无自接触、交叉脚姿态的接触力响应、
手动调速与重置、步态交替/停步/部分重置、自适应周期映射/步频平滑/相位连续性、
两倍速度范围，以及两组任务各自的 GPU 上
100 步接触检测和一次 CTS 更新；另外覆盖两种 CTS 的教师/学生梯度隔离、历史推进与部分重置、
PPO/蒸馏更新、检查点与学生优化器恢复、学生 TorchScript/ONNX 数值一致性；
没有 CUDA 时跳过 GPU 测试。

## 资源

模型来自 [LimX TRON2 robot description](https://github.com/limxdynamics/tron2-robot-description)，
原始提交为 `f547f5bc949f2a4c98e076e61cf6d3ca73d179a0`。
仅保留 SFYG_TRON2A 的 XML 和网格；参考项目、TRON2B、其他形态、
URDF、USD、Xacro 和嵌套 Git 元数据已移除。原 XML 不改写，由加载器完成训练适配。

上游许可与来源声明保留在 [资源目录](src/tron2_mjlab/tron2-robot-description/LICENSE)。
[上游资源清单](src/tron2_mjlab/tron2-robot-description/ASSETS.md) 及上游说明文档为原始仓库快照，不代表本项目包含的形态，
其中仍标注 TRON2A 网格来源确认待完成；公开再分发前应向上游确认。
