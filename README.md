# TRON2A SFYG 平地训练

基于 mjlab 的基础速度跟踪环境，仅包含 TRON2A 的 SFYG 形态。

| 任务 | Actor / Critic 观测维度 | 动作 |
| --- | --- | --- |
| `Mjlab-Velocity-Flat-TRON2A-SFYG-LBC` | 56 / 68，含机械臂状态及步态相位 | 10 个腿关节位置目标 |
| `Mjlab-Velocity-Flat-TRON2A-SFYG-Baseline` | 44 / 56，不含机械臂状态，含步态相位 | 10 个腿关节位置目标 |

实现见 [模型适配](src/tron2_mjlab/robot.py) 和 [环境配置](src/tron2_mjlab/env_cfgs.py)。
机械臂保留 6 个关节自由度，由独立位置执行器控制，不参与策略动作；LBC 观测机械臂状态，Baseline 不观测。
未接入外部控制时，机械臂以原 XML 零位为位置目标，通过 PD 保持，并非刚性固定。
夹爪仍固定在零位；机械臂与夹爪的外观、质量和碰撞几何保留。
仅使用平面地形，无高度扫描或地形课程。

两组任务均以膝盖前倾姿态初始化：左髋 yaw 为 `-pi`（−180°），右髋 yaw 为 `+pi`（+180°），
符号遵循原模型左右镜像的关节限位。髋 pitch、膝、踝 pitch 分别保持 `0.3`、`-0.6`、`0.3` rad，
髋 roll 为零。重置、零动作位置目标、相对关节位置观测和姿态奖励均使用这一默认姿态。
执行器位置目标仍按原有机制扣除随机化的编码器偏置。
原始 XML 不改写；此前基于 yaw 零位训练的策略应重新训练。

## 策略观测与机械臂控制

LBC 的 Actor 输入共 56 维，按下表顺序拼接，无历史帧堆叠：

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

```python
from tron2_mjlab.robot import ARM_JOINTS

robot = env.scene["robot"]
arm_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)
# q_ik: shape (env.num_envs, 6), radians, on env.device, within joint limits.
robot.set_joint_position_target(q_ik, joint_ids=arm_ids)
obs, reward, terminated, truncated, info = env.step(leg_actions)
```

目标会保持到下次更新，不被双腿策略动作覆盖；环境重置时机械臂状态与目标回到零。
当前未实现 IK 求解器或机械臂运动课程，PD 参数只是仿真起点。
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

## 交替步态与速度范围

LBC 和 Baseline 使用同一套步态奖励与相位观测：

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
uv run train Mjlab-Velocity-Flat-TRON2A-SFYG-LBC
uv run train Mjlab-Velocity-Flat-TRON2A-SFYG-Baseline
```

默认 1024 个并行环境，物理步长 2 ms，策略频率 50 Hz，10000 次 PPO 迭代。
使用 TensorBoard 本地日志，无需 W&B 登录。日志和模型位于 `logs/rsl_rl/`。
LBC 实验目录为 `tron2a_sfyg_velocity_lbc`，Baseline 为 `tron2a_sfyg_velocity_baseline`。
原无后缀任务名已改为 `-LBC`；当前版本还加入了相位观测，不能加载旧 54 维 Actor 模型。
历史日志目录不自动迁移。
显存不足时使用 `--env.scene.num-envs 256`。

快速检查训练入口：

```bash
uv run train Mjlab-Velocity-Flat-TRON2A-SFYG-LBC --env.scene.num-envs 16 --agent.max-iterations 1
```

## 预览

### 手动调速面板

```bash
uv run tron2-play SFYG --checkpoint-file /path/to/lbc/model.pt
uv run tron2-play SFYG --baseline --checkpoint-file /path/to/baseline/model.pt
```

打开终端输出的 Viser 地址，在 `Velocity Command` 面板中调节 `Forward (m/s)`、
`Lateral (m/s)` 和 `Yaw (rad/s)`。默认启用 `Manual control`，初始速度为零；
取消勾选恢复自动随机指令。
`Stop` 将三个目标速度归零并切回手动模式，不是暂停仿真或实机急停。
原 `Simulation / Speed` 调整的是仿真播放速度，与机器人目标速度不同。

没有模型时可用 `uv run tron2-play SFYG --agent zero` 检查面板，但零动作策略不会跟踪速度。
手动回放默认使用 LBC；检查 Baseline 时加 `--baseline`，加载 Baseline checkpoint 也必须加此选项。
必须指定 SFYG 模型完整的 `.pt` 文件路径，不能只指定目录。

### 自动指令回放

```bash
uv run play Mjlab-Velocity-Flat-TRON2A-SFYG-LBC --agent zero --viewer viser
uv run play Mjlab-Velocity-Flat-TRON2A-SFYG-LBC --checkpoint-file /path/to/lbc/model.pt --viewer viser
uv run play Mjlab-Velocity-Flat-TRON2A-SFYG-Baseline --checkpoint-file /path/to/baseline/model.pt --viewer viser
```

零动作仅用于检查模型，不代表已经学会平衡。PD 增益、初始姿态和奖励是训练起点，
尚未验证长期收敛或实机部署，不应直接用于实机控制。

## 测试

```bash
uv run python -m unittest discover -s tests -v
```

覆盖两组 SFYG 任务注册、模型编译、平地配置、机械臂状态观测与独立目标、baseline 观测隔离、
手动调速与重置、步态交替/停步/部分重置、自适应周期映射/步频平滑/相位连续性、
两倍速度范围，以及两组任务各自的 GPU 上
100 步接触检测和一次 PPO 更新；
没有 CUDA 时跳过 GPU 测试。

## 资源

模型来自 [LimX TRON2 robot description](https://github.com/limxdynamics/tron2-robot-description)，
原始提交为 `f547f5bc949f2a4c98e076e61cf6d3ca73d179a0`。
仅保留 SFYG_TRON2A 的 XML 和网格；参考项目、TRON2B、其他形态、
URDF、USD、Xacro 和嵌套 Git 元数据已移除。原 XML 不改写，由加载器完成训练适配。

上游许可与来源声明保留在 [资源目录](src/tron2_mjlab/tron2-robot-description/LICENSE)。
[上游资源清单](src/tron2_mjlab/tron2-robot-description/ASSETS.md) 及上游说明文档为原始仓库快照，不代表本项目包含的形态，
其中仍标注 TRON2A 网格来源确认待完成；公开再分发前应向上游确认。
