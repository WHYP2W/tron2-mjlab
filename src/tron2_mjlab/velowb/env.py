"""VeloWB manager terms and isolated experiment configuration."""

from dataclasses import dataclass, field
import math

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_angle_axis, quat_mul

from ..env_cfgs import tron2_flat_env_cfg
from ..robot import LEG_JOINTS, VARIANTS, get_robot_cfg
from .control import (
    DelayBuffer, PoseRegulator, damped_least_squares, kinematic_preview,
    limit_joint_velocity, pose_error, waveform,
)
from .kinematics import ArmKinematics, rotation_matrix


BASELINES = {
    "B0": ("pose", "none", "none"),
    "B1": ("velocity", "none", "none"),
    "B2": ("velocity", "history", "none"),
    "B3": ("velocity", "two_stage", "none"),
    "B4": ("velocity", "cts", "none"),
    "B5": ("velocity", "none", "only"),
    "B6": ("velocity", "none", "observation"),
    "B7": ("velocity", "cts", "residual"),
}


@dataclass
class ServoConfig:
    variant: str = "WFYG"
    baseline: str = "B1"
    num_envs: int = 256
    seed: int = 42
    control_profile: str = "standard"
    actuator_delay_steps: int = 0
    command_frame: str = "world"
    height_tracking: bool = False
    lock_arm_yaw: bool = True
    encoder: str = "default"
    ik: str = "default"
    preview: str = "none"
    history_seconds: float = 0.4
    residual_bound: float = 0.5
    servo_inputs: str = "full"
    command_axes: str = "linear_yaw"
    command_profile: str = "curriculum"
    command_limits: tuple[float, ...] = (0.15, 0.1, 0.1, 0.25, 0.25, 0.3)
    zero_ratio: float = 0.3
    curriculum_steps: int = 120_000
    max_frequency: float = 1.0
    sensor_noise: float = 0.0
    sensor_delay_steps: int = 0
    timestamp_skew_steps: int = 0
    odometry_drift: float = 0.0
    command_delay_steps: int = 0
    action_delay_steps: int = 0
    action_rate_weight: float = 0.05
    episode_seconds: float = 20.0
    privileged_ablation: str = "none"
    payload_mass: tuple[float, float] = (0.0, 0.0)
    payload_com: float = 0.0
    friction: tuple[float, float] = (0.8, 0.8)
    motor_strength: tuple[float, float] = (1.0, 1.0)
    base_mass_offset: tuple[float, float] = (0.0, 0.0)
    base_com_range: tuple[float, float, float] = (0.0, 0.0, 0.0)
    leg_mass_scale: tuple[float, float] = (1.0, 1.0)
    push_velocity: float = 0.0

    @property
    def task(self):
        return BASELINES[self.baseline][0]

    @property
    def encoder_mode(self):
        return BASELINES[self.baseline][1] if self.encoder == "default" else self.encoder

    @property
    def ik_mode(self):
        return BASELINES[self.baseline][2] if self.ik == "default" else self.ik

    def validate(self):
        if self.variant not in VARIANTS or self.baseline not in BASELINES:
            raise ValueError("Unknown robot variant or baseline")
        if self.control_profile not in ("standard", "wbc"):
            raise ValueError("Unknown control profile")
        if self.command_frame not in ("world", "heading"):
            raise ValueError("Command frame must be world or heading")
        if self.task == "pose" and (self.height_tracking or self.command_frame != "world"):
            raise ValueError("Pose baselines use world pose targets, not heading/height servo modes")
        if self.height_tracking and self.control_profile != "wbc":
            raise ValueError("Height feedback is an explicit WBC engineering ablation only")
        if self.control_profile == "wbc":
            if self.variant != "WFYG" or not self.lock_arm_yaw:
                raise ValueError("WBC requires WFYG with locked arm1 and arm6")
            if self.ik_mode != "none" or self.preview != "none":
                raise ValueError("WBC position actions do not yet support IK/preview integration")
        elif self.actuator_delay_steps:
            raise ValueError("Physics-step actuator delay is only supported by WBC")
        if self.encoder_mode not in ("none", "history", "two_stage", "cts"):
            raise ValueError("Unknown encoder mode")
        if self.ik_mode not in ("none", "observation", "residual", "only"):
            raise ValueError("Unknown IK integration")
        if self.preview not in ("none", "joint", "joint_ee", "full"):
            raise ValueError("Unknown preview mode")
        if self.command_axes not in ("linear_yaw", "full", "curriculum"):
            raise ValueError("Unknown command axes")
        if self.servo_inputs not in ("full", "error", "cmd_meas"):
            raise ValueError("Unknown servo input ablation")
        if self.command_profile not in ("curriculum", "constant", "ramp", "trapezoid",
                                        "sine", "bandlimited", "minimum_jerk"):
            raise ValueError("Unknown training command profile")
        if not 0 <= self.zero_ratio <= 1 or self.num_envs <= 0:
            raise ValueError("Invalid zero ratio or environment count")
        if (min(self.actuator_delay_steps, self.sensor_delay_steps, self.timestamp_skew_steps, self.command_delay_steps,
            self.action_delay_steps, self.sensor_noise, self.odometry_drift) < 0
                or self.history_seconds < 0 or self.residual_bound < 0
                or self.episode_seconds <= 0 or self.curriculum_steps <= 0):
            raise ValueError("Invalid history, delay, noise, bound or duration")
        if len(self.command_limits) != 6 or min(self.command_limits) <= 0:
            raise ValueError("Six positive twist limits are required")
        for name in ("payload_mass", "friction", "motor_strength", "leg_mass_scale"):
            lower, upper = getattr(self, name)
            if lower < 0 or upper < lower or (name != "payload_mass" and lower == 0):
                raise ValueError(f"Invalid {name} range")
        if self.payload_com < 0 or self.push_velocity < 0 or self.max_frequency < 0.1:
            raise ValueError("Invalid COM, disturbance or frequency range")
        if self.base_mass_offset[0] <= -12.57 or self.base_mass_offset[1] < self.base_mass_offset[0]:
            raise ValueError("Base mass offsets must preserve a positive robot base mass")
        if len(self.base_com_range) != 3 or min(self.base_com_range) < 0:
            raise ValueError("Base COM ranges must have three nonnegative values")
        if self.privileged_ablation not in ("none", "payload", "contact", "com"):
            raise ValueError("Unknown privileged ablation")


@dataclass(kw_only=True)
class ServoCommandCfg(CommandTermCfg):
    settings: ServoConfig = field(default_factory=ServoConfig)
    evaluation: bool = False

    def build(self, env):
        return ServoCommand(self, env)


class ServoCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.settings = cfg.settings
        self.robot = env.scene["robot"]
        self.chain = ArmKinematics(env.sim.mj_model, prefix="robot/", device=self.device)
        self.arm_ids, _ = self.robot.find_joints(
            [name.split("/")[-1] for name in self.chain.joint_names], preserve_order=True)
        self.leg_ids, _ = self.robot.find_joints(LEG_JOINTS)
        self.wheel_ids = (self.robot.find_joints("wheel_.*")[0]
                          if self.settings.variant == "WFYG" else [])
        if self.wheel_ids:
            model = env.sim.mj_model
            nominal = mujoco.MjData(model)
            nominal.qpos[self.robot.indexing.joint_q_adr.cpu().numpy()] = self.robot.data.default_joint_pos[0].cpu().numpy()
            mujoco.mj_forward(model, nominal)
            left, right = model.body("robot/wheel_L_Link").id, model.body("robot/wheel_R_Link").id
            self.track_width = abs(float(nominal.xpos[left, 1] - nominal.xpos[right, 1]))
            wheel_geoms = [geom for geom in range(model.ngeom) if model.geom_bodyid[geom] == left
                           and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER]
            self.wheel_radius = float(model.geom_size[wheel_geoms[0], 0])
        self.ee_id = self.robot.find_sites("ee")[0][0]
        self.action_dim = len(self.leg_ids) + len(self.wheel_ids) + len(self.arm_ids)
        self.limits = torch.tensor(self.settings.command_limits, device=self.device)
        self._command = torch.zeros((self.num_envs, 6), device=self.device)
        self.requested_command = torch.zeros_like(self._command)
        self.command_buffer = DelayBuffer(self.settings.command_delay_steps, self._command)
        self.command_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.height_target = torch.zeros(self.num_envs, device=self.device)
        self.height_origin = torch.zeros_like(self.height_target)
        self.height_tick = torch.full_like(self.command_delay, -1)
        self.height_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.pending_command_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.delivery_step = -1
        self.target = torch.zeros_like(self._command)
        self.start = torch.zeros_like(self._command)
        self.target_position = self._command[:, :3].clone()
        self.target_rotation = torch.eye(3, device=self.device).repeat(self.num_envs, 1, 1)
        self.elapsed = torch.zeros((self.num_envs, 1), device=self.device)
        self.duration = torch.ones_like(self.elapsed)
        self.frequency = torch.ones_like(self.elapsed) * 0.1
        self.phase = torch.zeros_like(self._command)
        self.zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.profile = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.sensor_delay = torch.zeros_like(self.profile)
        self.odom_delay = torch.zeros_like(self.profile)
        self.odom_bias = torch.zeros((self.num_envs, 3), device=self.device)
        self.action_delay = torch.zeros_like(self.profile)
        self.pending_reset = torch.ones_like(self.zero)
        self.pending_goal = torch.ones_like(self.zero)
        self.teacher_mask = torch.zeros((self.num_envs, 1), device=self.device)
        if self.settings.encoder_mode == "cts" and not cfg.evaluation:
            ratio = 0.75 if self.settings.control_profile == "wbc" else 0.5
            teacher_count = max(1, min(self.num_envs - 1, int(self.num_envs * ratio)))
            self.teacher_mask[:teacher_count] = 1
        elif self.settings.encoder_mode == "two_stage" and not cfg.evaluation:
            self.teacher_mask[:] = 1
        self.external = False
        self.external_pose = False
        self.sensor_buffer = None
        self.sensor_step = -1
        self.prior_cache = None
        self.preview_cache = None
        self.regulator = PoseRegulator(self.num_envs, self.device, env.step_dt,
                                       tuple(self.settings.command_limits))
        self.metrics["twist_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return self._command

    def reset(self, env_ids):
        self.pending_reset[env_ids] = True
        self.pending_command_reset[env_ids] = True
        self.delivery_step = -1
        self.odom_bias[env_ids] = 0
        self.height_pending[env_ids] = True
        self.height_tick[env_ids] = -1
        self.sensor_step = -1
        self.sensor_delay[env_ids] = torch.randint(
            self.settings.sensor_delay_steps + 1, (len(env_ids),), device=self.device)
        self.odom_delay[env_ids] = self.sensor_delay[env_ids] + torch.randint(
            self.settings.timestamp_skew_steps + 1, (len(env_ids),), device=self.device)
        self.command_delay[env_ids] = torch.randint(
            self.settings.command_delay_steps + 1, (len(env_ids),), device=self.device)
        self.action_delay[env_ids] = torch.randint(
            self.settings.action_delay_steps + 1, (len(env_ids),), device=self.device)
        self.regulator.reset(env_ids)
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        if self.external:
            return
        self.start[env_ids] = self._command[env_ids]
        self.target[env_ids] = (2 * torch.rand((len(env_ids), 6), device=self.device) - 1) * self.limits
        self.zero[env_ids] = torch.rand(len(env_ids), device=self.device) < self.settings.zero_ratio
        self.elapsed[env_ids] = 0
        self.duration[env_ids] = self.time_left[env_ids, None].clamp_min(self._env.step_dt)
        progress = min(1.0, self._env.common_step_counter / self.settings.curriculum_steps)
        upper = 0.1 + progress * (self.settings.max_frequency - 0.1)
        self.frequency[env_ids] = 0.1 + torch.rand((len(env_ids), 1), device=self.device) * (upper - 0.1)
        self.phase[env_ids] = 2 * math.pi * torch.rand((len(env_ids), 6), device=self.device)
        profile_count = 2 if progress < 0.25 else (4 if progress < 0.6 else 6)
        self.profile[env_ids] = torch.randint(profile_count, (len(env_ids),), device=self.device)
        self.pending_goal[env_ids] = True

    def _update_metrics(self):
        self.metrics["twist_error"] += torch.linalg.vector_norm(
            self._command - self.transform_twist(self.robot.data.site_vel_w[:, self.ee_id],
                                                  estimated=False), dim=-1) / self._env.max_episode_length

    def _update_command(self):
        self.sensor_step = -1
        self.sense()
        if not self.external:
            profiles = ("constant", "ramp", "sine", "bandlimited", "trapezoid", "minimum_jerk")
            if self.settings.command_profile == "curriculum":
                for index, profile in enumerate(profiles):
                    value = waveform(profile, self.elapsed, self.duration, self.start,
                                     self.target, self.frequency, self.phase)
                    self._command[:] = torch.where((self.profile == index)[:, None], value, self._command)
            else:
                self._command[:] = waveform(self.settings.command_profile, self.elapsed,
                                             self.duration, self.start, self.target,
                                             self.frequency, self.phase)
            self._command[self.zero] = 0
            if (self.settings.command_axes == "linear_yaw" or
                    (self.settings.command_axes == "curriculum" and
                     self._env.common_step_counter < self.settings.curriculum_steps // 3)):
                self._command[:, 3:5] = 0
            self.elapsed += self._env.step_dt
            self._deliver_command(self._command.clone())
        self.prior_cache = self.preview_cache = None

    def _deliver_command(self, requested):
        self.requested_command[:] = requested
        reset_ids = self.pending_command_reset.nonzero().flatten()
        if len(reset_ids):
            self.command_buffer.reset(reset_ids, torch.zeros_like(requested))
            self.pending_command_reset[reset_ids] = False
        if self.delivery_step != self._env.common_step_counter:
            self._command[:] = self.command_buffer.push(requested, self.command_delay)
            self.delivery_step = self._env.common_step_counter
        else:
            self.command_buffer.buffer[self.command_buffer.cursor] = requested
            self._command[:] = self.command_buffer.read(self.command_delay)
        if self.settings.height_tracking:
            self._update_height_target()

    def transform_twist(self, value, *, to_world=False, estimated=True):
        if self.settings.command_frame == "world":
            return value
        if estimated:
            self.sense()
            rotation = self.base_rotation
        else:
            rotation = rotation_matrix(self.robot.data.root_link_quat_w)
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        axis = torch.zeros((self.num_envs, 3), device=self.device)
        axis[:, 2] = 1
        heading = rotation_matrix(quat_from_angle_axis(yaw, axis))
        if not to_world:
            heading = heading.transpose(-1, -2)
        return torch.cat(((heading @ value[:, :3, None]).squeeze(-1),
                          (heading @ value[:, 3:, None]).squeeze(-1)), dim=-1)

    def _update_height_target(self):
        self.sense()
        fresh = self.height_tick != self._env.common_step_counter
        self.height_origin[fresh] = self.height_target[fresh]
        self.height_tick[:] = self._env.common_step_counter
        target = (self.height_origin + self._command[:, 2] * self._env.step_dt).clamp(0.05, 1.6)
        actual = self.estimate.position[:, 2]
        self.height_target[:] = target.clamp(actual - 0.15, actual + 0.15)

    def set_twist(self, command: torch.Tensor):
        self.external, self.external_pose = True, False
        self._deliver_command(command.to(self.device).clamp(-self.limits, self.limits))
        self.prior_cache = self.preview_cache = None

    def set_pose(self, position: torch.Tensor, rotation: torch.Tensor,
                 feedforward: torch.Tensor | None = None):
        """Advance the outer loop once; call before each policy evaluation."""
        self.external, self.external_pose = True, True
        self.target_position[:] = position
        self.target_rotation[:] = rotation
        self.pending_goal[:] = False
        self.feedforward = torch.zeros_like(self._command) if feedforward is None else feedforward
        self.sense()
        if self.settings.task != "pose":
            world_command = self.regulator.step(position, rotation, self.estimate.position,
                                                 self.estimate.rotation, self.estimate.twist,
                                                 self.feedforward)
            self._deliver_command(self.transform_twist(world_command).clamp(-self.limits, self.limits))
        else:
            self._command[:] = self.feedforward
            self.requested_command[:] = self.feedforward
        self.prior_cache = self.preview_cache = None

    def sense(self):
        if self.sensor_step == self._env.common_step_counter:
            return
        self.sensor_step = self._env.common_step_counter
        data = self.robot.data
        noise = self.settings.sensor_noise
        joint_position = data.joint_pos_biased.clone()
        joint_velocity = data.joint_vel.clone()
        base_pose = data.root_link_pose_w.clone()
        base_velocity = data.root_link_vel_w.clone()
        if noise:
            joint_position += noise * 0.002 * torch.randn_like(joint_position)
            joint_velocity += noise * 0.02 * torch.randn_like(joint_velocity)
            base_pose[:, :3] += noise * 0.002 * torch.randn_like(base_pose[:, :3])
            rotvec = noise * 0.005 * torch.randn_like(base_pose[:, :3])
            angle = rotvec.norm(dim=-1)
            perturbation = quat_from_angle_axis(angle, rotvec / angle[:, None].clamp_min(1e-9))
            base_pose[:, 3:] = quat_mul(perturbation, base_pose[:, 3:])
            base_velocity += noise * 0.015 * torch.randn_like(base_velocity)
        if self.settings.odometry_drift:
            self.odom_bias += self.settings.odometry_drift * math.sqrt(self._env.step_dt) * torch.randn_like(self.odom_bias)
            base_pose[:, :3] += self.odom_bias
        sensor = torch.cat((joint_position, joint_velocity, base_pose, base_velocity), dim=-1)
        if self.sensor_buffer is None:
            self.sensor_buffer = DelayBuffer(self.settings.sensor_delay_steps + self.settings.timestamp_skew_steps, sensor)
        reset_ids = self.pending_reset.nonzero().flatten()
        if len(reset_ids):
            self.sensor_buffer.reset(reset_ids, sensor)
            self.pending_reset[reset_ids] = False
        sensor = self.sensor_buffer.push(sensor, self.sensor_delay)
        joint_count = joint_position.shape[1]
        odometry = self.sensor_buffer.read(self.odom_delay)
        self.q, self.qd = sensor[:, :joint_count], sensor[:, joint_count:2 * joint_count]
        self.base_pose = odometry[:, 2 * joint_count:2 * joint_count + 7]
        self.base_velocity = odometry[:, -6:]
        self.base_rotation = rotation_matrix(self.base_pose[:, 3:])
        self.estimate = self.chain.estimate(
            self.q[:, self.arm_ids], self.qd[:, self.arm_ids],
            self.base_pose[:, :3], self.base_pose[:, 3:],
            self.base_velocity[:, :3], self.base_velocity[:, 3:])
        if self.settings.height_tracking:
            self.height_target[self.height_pending] = self.estimate.position[self.height_pending, 2]
            self.height_origin[self.height_pending] = self.height_target[self.height_pending]
            self.height_pending[:] = False
        goal_ids = self.pending_goal.nonzero().flatten()
        if len(goal_ids):
            goal_q = self.q[goal_ids][:, self.arm_ids] + 0.5 * (
                torch.rand((len(goal_ids), len(self.arm_ids)), device=self.device) - 0.5)
            goal_q = goal_q.clamp(self.chain.limits[:, 0], self.chain.limits[:, 1])
            goal = self.chain.forward(goal_q)
            rotation = self.base_rotation[goal_ids]
            self.target_position[goal_ids] = self.base_pose[goal_ids, :3] + (
                rotation @ goal.position.unsqueeze(-1)).squeeze(-1)
            self.target_rotation[goal_ids] = rotation @ goal.rotation
            self.pending_goal[goal_ids] = False

    def prior(self):
        self.sense()
        if self.prior_cache is not None:
            return self.prior_cache
        jacobian = self.estimate.jacobian
        wheel_count = len(self.wheel_ids)
        if wheel_count:
            heading = self.base_rotation[:, :, 0].clone()
            heading[:, 2] = 0
            heading = torch.nn.functional.normalize(heading, dim=-1)
            angular = torch.zeros_like(heading)
            angular[:, 2] = 1
            offset = self.estimate.position - self.base_pose[:, :3]
            yaw_linear = torch.linalg.cross(angular, offset, dim=-1)
            half_radius = self.wheel_radius / 2
            yaw_gain = self.wheel_radius / self.track_width
            left = torch.cat((half_radius * heading - yaw_gain * yaw_linear,
                              -yaw_gain * angular), dim=-1)
            right = torch.cat((half_radius * heading + yaw_gain * yaw_linear,
                               yaw_gain * angular), dim=-1)
            jacobian = torch.cat((torch.stack((left, right), dim=-1), jacobian), dim=-1)
        speed_limits = torch.tensor([10.0] * wheel_count + [1.5] * len(self.arm_ids), device=self.device)
        world_command = self.transform_twist(self._command, to_world=True)
        velocity = damped_least_squares(jacobian / self.limits[None, :, None],
                           world_command / self.limits,
                                       velocity_limit=speed_limits)
        velocity[:, wheel_count:] = limit_joint_velocity(
            self.q[:, self.arm_ids], velocity[:, wheel_count:], self.chain.limits, self._env.step_dt)
        self.arm_prior_velocity = velocity[:, wheel_count:]
        self.wheel_prior_twist = ((jacobian[:, :, :wheel_count] @ velocity[:, :wheel_count, None]).squeeze(-1)
                                  if wheel_count else torch.zeros_like(self._command))
        self.prior_cache = torch.cat((torch.zeros((self.num_envs, len(self.leg_ids)), device=self.device),
                                      velocity / speed_limits), dim=-1)
        return self.prior_cache

    def preview(self):
        self.sense()
        if self.preview_cache is None:
            self.prior()
            inverse = self.base_rotation.transpose(-1, -2)
            arm_command = self.transform_twist(self._command, to_world=True) - self.wheel_prior_twist
            command_base = torch.cat(((inverse @ arm_command[:, :3, None]).squeeze(-1),
                                      (inverse @ arm_command[:, 3:, None]).squeeze(-1)), dim=-1)
            self.preview_cache = kinematic_preview(self.chain, self.q[:, self.arm_ids],
                                                     command_base, self.settings.preview,
                                                     initial_velocity=self.arm_prior_velocity)
        return self.preview_cache


@dataclass(kw_only=True)
class WholeBodyActionCfg(ActionTermCfg):
    def build(self, env):
        return WholeBodyAction(self, env)


class WholeBodyAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.servo = env.command_manager.get_term("ee_twist")
        self._raw = torch.zeros((self.num_envs, self.servo.action_dim), device=self.device)
        self.processed = torch.zeros_like(self._raw)
        self.previous = torch.zeros_like(self._raw)
        self.previous_previous = torch.zeros_like(self._raw)
        self.arm_target = self._entity.data.default_joint_pos[:, self.servo.arm_ids].clone()
        self.delay_buffer = DelayBuffer(self.servo.settings.action_delay_steps, self._raw)
        self.position_ids = self.servo.leg_ids + self.servo.arm_ids
        self.is_wbc = self.servo.settings.control_profile == "wbc"
        self.physics_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.physics_buffer = DelayBuffer(self.servo.settings.actuator_delay_steps, self._raw)
        self.applied = torch.zeros_like(self._raw)
        if self.is_wbc:
            names = [self._entity.joint_names[index] for index in self.position_ids]
            self.position_scale = torch.tensor([
                1.0 if name in ("arm2_Joint", "arm3_Joint") else
                0.5 if name.startswith("arm") else
                0.15 if name.startswith("proximal_yaw") else 0.2192
                for name in names
            ], device=self.device)
            joint_names = self._entity.joint_names
            self.torque_weights = torch.tensor([
                0.1 if name.startswith("arm") else 30.0 if name.startswith("wheel") else 8.0
                for name in joint_names
            ], device=self.device)
            self.power_weights = torch.tensor([
                0.05 if name.startswith("arm") else 15.0 if name.startswith("wheel") else 6.0
                for name in joint_names
            ], device=self.device)
            self.velocity_limits = torch.tensor([
                5.0 if name.startswith("arm") else 41.89 if name.startswith("wheel") else
                14.66 if name.startswith("proximal_yaw") else 12.57 for name in joint_names
            ], device=self.device)
            self.wheel_body_ids = self._entity.find_bodies("wheel_[LR]_Link")[0]
            self.symmetry_pairs = {
                kind: self._entity.find_joints(f"{kind}_[LR]_Joint", preserve_order=True)[0]
                for kind in ("knee", "proximal_pitch", "proximal_yaw")
            }
            self.previous_torque = torch.zeros_like(self._entity.data.joint_pos)
            self.previous_base_velocity = torch.zeros((self.num_envs, 3), device=self.device)

    @property
    def action_dim(self):
        return self.servo.action_dim

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self._raw[:] = actions
        if self.is_wbc:
            self.previous_torque[:] = self._env.sim.data.qfrc_actuator[:, self._entity.indexing.joint_v_adr]
            self.previous_base_velocity[:] = self._entity.data.root_link_lin_vel_w
        mode = self.servo.settings.ik_mode
        if mode == "only":
            normalized = self.servo.prior()
        elif mode == "residual":
            normalized = self.servo.prior() + self.servo.settings.residual_bound * actions.clamp(-1, 1)
        else:
            normalized = actions
        self.previous_previous[:] = self.previous
        self.previous[:] = self.processed
        if self.is_wbc:
            position_count = len(self.position_ids)
            limits = self._entity.data.joint_pos_limits[:, self.position_ids]
            default = self._entity.data.default_joint_pos[:, self.position_ids]
            bounded = torch.cat((
                normalized[:, :position_count].clamp((limits[:, :, 0] - default) / self.position_scale,
                                                       (limits[:, :, 1] - default) / self.position_scale),
                normalized[:, position_count:].clamp(-41.89 / 7.0, 41.89 / 7.0),
            ), dim=-1)
        else:
            bounded = normalized.clamp(-1, 1)
        self.processed[:] = self.delay_buffer.push(bounded, self.servo.action_delay)
        if self.is_wbc:
            return
        arm_count = len(self.servo.arm_ids)
        self.arm_target += self._env.step_dt * 1.5 * self.processed[:, -arm_count:]
        self.arm_target.clamp_(self.servo.chain.limits[:, 0], self.servo.chain.limits[:, 1])

    def apply_actions(self):
        servo = self.servo
        if self.is_wbc:
            self.applied[:] = self.physics_buffer.push(self.processed, self.physics_delay)
            position_count = len(self.position_ids)
            target = (self._entity.data.default_joint_pos[:, self.position_ids]
                      + self.position_scale * self.applied[:, :position_count])
            limits = self._entity.data.joint_pos_limits[:, self.position_ids]
            target = target.clamp(limits[:, :, 0], limits[:, :, 1])
            self.arm_target[:] = target[:, -len(servo.arm_ids):]
            self._entity.set_joint_position_target(target, joint_ids=self.position_ids)
            self._entity.set_joint_velocity_target(7.0 * self.applied[:, position_count:],
                                                   joint_ids=servo.wheel_ids)
            return
        self.applied[:] = self.processed
        leg_count = len(servo.leg_ids)
        self._entity.set_joint_position_target(
            self._entity.data.default_joint_pos[:, servo.leg_ids] + 0.25 * self.processed[:, :leg_count],
            joint_ids=servo.leg_ids)
        if servo.wheel_ids:
            self._entity.set_joint_velocity_target(10 * self.processed[:, leg_count:leg_count + 2],
                                                   joint_ids=servo.wheel_ids)
        self._entity.set_joint_position_target(self.arm_target, joint_ids=servo.arm_ids)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._raw[env_ids] = self.processed[env_ids] = self.previous[env_ids] = 0
        self.previous_previous[env_ids] = self.applied[env_ids] = 0
        self.arm_target[env_ids] = self._entity.data.default_joint_pos[env_ids][:, self.servo.arm_ids]
        self.delay_buffer.reset(env_ids, self._raw)
        if self.is_wbc:
            self.physics_delay[env_ids] = torch.randint(self.servo.settings.actuator_delay_steps + 1,
                                                       (len(env_ids),), device=self.device)
            self.previous_torque[env_ids] = 0
            self.previous_base_velocity[env_ids] = 0
        self.physics_buffer.reset(env_ids, self._raw)


def servo_observation(env):
    servo = env.command_manager.get_term("ee_twist")
    servo.sense()
    data = servo.robot.data
    joint_position = servo.q - data.default_joint_pos
    joint_position[:, servo.wheel_ids] = 0
    gravity = quat_apply_inverse(servo.base_pose[:, 3:], data.gravity_vec_w)
    action = env.action_manager.get_term("whole_body")
    previous = action.processed
    proprioception = torch.cat((servo.base_velocity, gravity,
                               servo.base_rotation[:, :, :2].flatten(1), joint_position,
                               0.1 * servo.qd, previous,
                               action.arm_target - servo.q[:, servo.arm_ids],
                               servo.sensor_delay[:, None] * env.step_dt,
                               servo.odom_delay[:, None] * env.step_dt), dim=-1)
    if servo.settings.task == "pose":
        task = torch.cat((servo.target_position - servo.base_pose[:, :3],
                          servo.target_rotation[:, :, :2].flatten(1),
                          pose_error(servo.target_position, servo.target_rotation,
                                     servo.estimate.position, servo.estimate.rotation),
                          servo.estimate.twist / servo.limits), dim=-1)
    else:
        command = servo.command / servo.limits
        measured = servo.transform_twist(servo.estimate.twist) / servo.limits
        mode = servo.settings.servo_inputs
        task = (command - measured if mode == "error" else
                torch.cat((command, measured), dim=-1) if mode == "cmd_meas" else
                torch.cat((command, measured, command - measured), dim=-1))
    if servo.settings.control_profile == "wbc":
        torque = env.sim.data.qfrc_actuator[:, servo.robot.indexing.joint_v_adr]
        proprioception = torch.cat((proprioception, torque / 100.0), dim=-1)
    if servo.settings.height_tracking:
        task = torch.cat((task, (servo.height_target - servo.estimate.position[:, 2])[:, None]), dim=-1)
    return torch.cat((proprioception, task), dim=-1)


def prior_observation(env):
    return env.command_manager.get_term("ee_twist").prior()


def preview_observation(env):
    return env.command_manager.get_term("ee_twist").preview()


def role_observation(env):
    return env.command_manager.get_term("ee_twist").teacher_mask


def privileged_observation(env):
    servo = env.command_manager.get_term("ee_twist")
    data = servo.robot.data
    contact = env.scene["feet_ground_contact"].data.force.flatten(1) / 100.0
    com = env.sim.data.subtree_com[:, servo.chain.base_id] - data.root_link_pos_w
    masses = env.sim.model.body_mass[:, servo.robot.indexing.body_ids]
    com_velocity = (data.body_com_vel_w[:, :, :3] * masses[:, :, None]).sum(1) / masses.sum(1, keepdim=True)
    if servo.settings.privileged_ablation == "contact":
        contact = torch.zeros_like(contact)
    if servo.settings.privileged_ablation == "com":
        com = torch.zeros_like(com)
        com_velocity = torch.zeros_like(com_velocity)
    model = env.sim.model
    payload_id = env.sim.mj_model.body("robot/payload").id
    payload = torch.cat((model.body_mass[:, payload_id, None] - 0.0001,
                         model.body_ipos[:, payload_id]), dim=-1)
    if servo.settings.privileged_ablation == "payload":
        payload = torch.zeros_like(payload)
    defaults = env.sim.get_default_field("actuator_gainprm")
    actuator_ids = servo.robot.indexing.ctrl_ids
    strength = model.actuator_gainprm[:, actuator_ids, 0] / defaults[actuator_ids, 0]
    friction = model.geom_friction[:, 0, :1]
    privileged = torch.cat((data.site_vel_w[:, servo.ee_id] / servo.limits,
                             data.root_link_vel_w, com, com_velocity, contact, payload, friction, strength,
                             servo.action_delay[:, None] * env.step_dt), dim=-1)
    if servo.settings.control_profile == "wbc":
        nominal_mass = env.sim.get_default_field("body_mass")[servo.robot.indexing.body_ids]
        mass_ratio = masses / nominal_mass.clamp_min(1e-6)
        base_com = model.body_ipos[:, servo.chain.base_id] - env.sim.get_default_field("body_ipos")[servo.chain.base_id]
        if servo.settings.privileged_ablation == "payload":
            payload_local = list(servo.robot.body_names).index("payload")
            mass_ratio = mass_ratio.clone()
            mass_ratio[:, payload_local] = 0
        if servo.settings.privileged_ablation == "com":
            base_com = torch.zeros_like(base_com)
        delay = env.action_manager.get_term("whole_body").physics_delay[:, None] * env.physics_dt
        privileged = torch.cat((privileged, data.joint_acc.clamp(-5000, 5000) / 5000,
                                 mass_ratio, base_com, delay), dim=-1)
    return privileged


@requires_model_fields("body_mass", "body_inertia", "body_ipos", "geom_friction",
                       "actuator_gainprm", "actuator_biasprm")
def randomize_dynamics(env, env_ids, settings: ServoConfig):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    model = env.sim.model
    robot = env.scene["robot"]
    count = len(env_ids)
    payload_id = env.sim.mj_model.body("robot/payload").id
    lower, upper = settings.payload_mass
    mass = lower + (upper - lower) * torch.rand(count, device=env.device) + 0.0001
    model.body_mass[env_ids, payload_id] = mass
    model.body_inertia[env_ids, payload_id] = (0.4 * 0.025**2 * mass)[:, None]
    model.body_ipos[env_ids, payload_id] = settings.payload_com * (
        2 * torch.rand((count, 3), device=env.device) - 1)
    lower, upper = settings.friction
    friction = lower + (upper - lower) * torch.rand(count, device=env.device)
    model.geom_friction[env_ids, :, 0] = friction[:, None]
    actuator_ids = robot.indexing.ctrl_ids
    lower, upper = settings.motor_strength
    strength = lower + (upper - lower) * torch.rand((count, len(actuator_ids)), device=env.device)
    for name in ("actuator_gainprm", "actuator_biasprm"):
        nominal = env.sim.get_default_field(name)[actuator_ids]
        getattr(model, name)[env_ids[:, None], actuator_ids[None, :]] = nominal[None] * strength[:, :, None]
    if tuple(settings.base_mass_offset) != (0.0, 0.0):
        base_id = robot.indexing.root_body_id
        lower, upper = settings.base_mass_offset
        nominal_mass = env.sim.get_default_field("body_mass")[base_id]
        mass = nominal_mass + lower + (upper - lower) * torch.rand(count, device=env.device)
        model.body_mass[env_ids, base_id] = mass
        model.body_inertia[env_ids, base_id] = (
            env.sim.get_default_field("body_inertia")[base_id] * (mass / nominal_mass)[:, None])
    if any(settings.base_com_range):
        base_id = robot.indexing.root_body_id
        extent = torch.tensor(settings.base_com_range, device=env.device)
        model.body_ipos[env_ids, base_id] = env.sim.get_default_field("body_ipos")[base_id] + (
            2 * torch.rand((count, 3), device=env.device) - 1) * extent
    if tuple(settings.leg_mass_scale) != (1.0, 1.0):
        body_ids = robot.find_bodies(("proximal_(pitch|roll|yaw)_[LR]_Link", "knee_[LR]_Link", "wheel_[LR]_Link"))[0]
        global_ids = robot.indexing.body_ids[body_ids]
        lower, upper = settings.leg_mass_scale
        scale = lower + (upper - lower) * torch.rand((count, len(body_ids)), device=env.device)
        for name in ("body_mass", "body_inertia"):
            nominal = env.sim.get_default_field(name)[global_ids]
            multiplier = scale if name == "body_mass" else scale[:, :, None]
            getattr(model, name)[env_ids[:, None], global_ids[None, :]] = nominal[None] * multiplier


def tracking_reward(env):
    servo = env.command_manager.get_term("ee_twist")
    if servo.settings.task == "pose":
        pose = servo.robot.data.site_pose_w[:, servo.ee_id]
        error = pose_error(servo.target_position, servo.target_rotation,
                           pose[:, :3], rotation_matrix(pose[:, 3:]))
        scale = torch.tensor([0.1] * 3 + [0.3] * 3, device=env.device)
    else:
        error = servo.command - servo.transform_twist(servo.robot.data.site_vel_w[:, servo.ee_id], estimated=False)
        scale = torch.tensor([0.15] * 3 + [0.3] * 3, device=env.device)
    return torch.exp(-((error / scale)**2).mean(dim=-1))


def balance_penalty(env):
    data = env.scene["robot"].data
    gravity = quat_apply_inverse(data.root_link_quat_w, data.gravity_vec_w)
    return (gravity[:, :2]**2).sum(-1) + 0.05 * (data.root_link_ang_vel_w[:, :2]**2).sum(-1)


def posture_penalty(env):
    servo = env.command_manager.get_term("ee_twist")
    data = servo.robot.data
    return ((data.joint_pos[:, servo.leg_ids] - data.default_joint_pos[:, servo.leg_ids])**2).sum(-1)


def action_rate(env):
    action = env.action_manager.get_term("whole_body")
    return ((action.processed - action.previous)**2).sum(-1)


def energy(env):
    robot = env.scene["robot"]
    torque = env.sim.data.qfrc_actuator[:, robot.indexing.joint_v_adr]
    return (torque * robot.data.joint_vel).abs().sum(-1)


def wbc_tracking(env, axis: int):
    servo = env.command_manager.get_term("ee_twist")
    if axis == 2 and servo.settings.height_tracking:
        error = servo.height_target - servo.robot.data.site_pose_w[:, servo.ee_id, 2]
        return torch.exp(-(error / 0.08)**2)
    actual = servo.transform_twist(servo.robot.data.site_vel_w[:, servo.ee_id], estimated=False)
    variance = 0.1 if axis in (3, 4) else 0.25
    return torch.exp(-(servo.command[:, axis] - actual[:, axis])**2 / variance)


def wbc_alive(env):
    return (~env.termination_manager.terminated).float()


def wbc_penalty(env, kind: str):
    action = env.action_manager.get_term("whole_body")
    servo = action.servo
    data = servo.robot.data
    if kind in ("roll_velocity", "pitch_velocity"):
        angular = quat_apply_inverse(data.root_link_quat_w, data.root_link_ang_vel_w)
        return angular[:, 0 if kind == "roll_velocity" else 1].square()
    if kind in ("base_acceleration", "arm_first"):
        acceleration = (data.root_link_lin_vel_w - action.previous_base_velocity) / env.step_dt
        if kind == "base_acceleration":
            return acceleration.square().sum(-1)
        limits = data.soft_joint_pos_limits[:, servo.arm_ids[:2]]
        normalized = (2 * data.joint_pos[:, servo.arm_ids[:2]] - limits.sum(-1)) / (
            limits[:, :, 1] - limits[:, :, 0]).clamp_min(1e-6)
        gate = 1 - ((normalized.abs().amax(-1) - 0.6) / 0.4).clamp(0, 1)
        return acceleration[:, 2].square() * gate
    if kind == "action_jerk":
        return (action.processed - 2 * action.previous + action.previous_previous).square().sum(-1)
    if kind in ("torque_rate", "weighted_torque", "weighted_power"):
        torque = env.sim.data.qfrc_actuator[:, servo.robot.indexing.joint_v_adr]
        if kind == "torque_rate":
            return (torque - action.previous_torque).square().sum(-1)
        if kind == "weighted_torque":
            return (torque.square() * action.torque_weights).sum(-1)
        return ((torque * data.joint_vel).abs() * action.power_weights).sum(-1)
    if kind == "leg_velocity":
        return data.joint_vel[:, servo.leg_ids].square().sum(-1)
    if kind == "wheel_idle":
        horizontal = servo.command[:, [0, 1, 5]].square().sum(-1)
        return data.joint_vel[:, servo.wheel_ids].square().sum(-1) * torch.exp(-horizontal / 0.15**2)
    if kind == "joint_acceleration":
        return data.joint_acc.clamp(-5000, 5000).square().sum(-1)
    if kind == "velocity_limits":
        return (data.joint_vel.abs() - 0.92 * action.velocity_limits).clamp(0, 1).sum(-1)
    if kind == "torque_limits":
        ctrl_ids = servo.robot.indexing.ctrl_ids
        model = env.sim.model
        computed = (model.actuator_gainprm[:, ctrl_ids, 0] * env.sim.data.ctrl[:, ctrl_ids]
                    + model.actuator_biasprm[:, ctrl_ids, 1] * env.sim.data.actuator_length[:, ctrl_ids]
                    + model.actuator_biasprm[:, ctrl_ids, 2] * env.sim.data.actuator_velocity[:, ctrl_ids])
        return (computed - data.actuator_force).abs().sum(-1)
    if kind in ("knee_symmetry", "pitch_symmetry", "yaw_sum", "yaw_deviation"):
        group = "knee" if kind == "knee_symmetry" else "proximal_pitch" if kind == "pitch_symmetry" else "proximal_yaw"
        position = data.joint_pos[:, action.symmetry_pairs[group]]
        if kind == "yaw_deviation":
            return position.abs().sum(-1)
        if kind == "yaw_sum":
            return position.sum(-1).square()
        return (position[:, 0] - position[:, 1]).square()
    if kind in ("wheel_slip", "wheel_contact"):
        force = env.scene["feet_ground_contact"].data.force.reshape(env.num_envs, -1, 3)
        contact = force.norm(dim=-1) > 5.0
        if kind == "wheel_contact":
            return contact.float().sum(-1)
        velocity = data.body_link_vel_w[:, action.wheel_body_ids]
        contact_offset = torch.zeros_like(velocity[:, :, :3])
        contact_offset[:, :, 2] = -servo.wheel_radius
        slip = velocity[:, :, :3] + torch.linalg.cross(velocity[:, :, 3:], contact_offset, dim=-1)
        return (slip[:, :, :2].square().sum(-1) * contact).sum(-1)
    if kind in ("leg_contact", "torso_contact"):
        force = env.scene[kind].data.force.reshape(env.num_envs, -1, 3)
        threshold = 10 if kind == "leg_contact" else 1
        return (force.norm(dim=-1) > threshold).float().sum(-1)
    raise ValueError(f"Unknown WBC penalty: {kind}")


def wbc_fallen(env):
    return env.scene["robot"].data.root_link_pos_w[:, 2] < 0.25


def reset_arm_to_default(env, env_ids):
    servo = env.command_manager.get_term("ee_twist")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    joint_ids = torch.tensor(servo.arm_ids, device=env.device)
    data = servo.robot.data
    servo.robot.write_joint_state_to_sim(
        data.default_joint_pos[env_ids][:, joint_ids], data.default_joint_vel[env_ids][:, joint_ids],
        joint_ids=joint_ids, env_ids=env_ids,
    )


def make_env_cfg(settings: ServoConfig, *, evaluation: bool = False) -> ManagerBasedRlEnvCfg:
    settings.validate()
    if settings.encoder_mode == "cts" and not evaluation and settings.num_envs < 2:
        raise ValueError("Concurrent CTS requires at least two training environments")
    cfg = tron2_flat_env_cfg(settings.variant)
    cfg.scene.entities["robot"] = get_robot_cfg(settings.variant, active_arm=True,
                                                lock_arm_yaw=settings.lock_arm_yaw,
                                                control_profile=settings.control_profile)
    cfg.scene.num_envs, cfg.seed = settings.num_envs, settings.seed
    cfg.episode_length_s = settings.episode_seconds
    cfg.commands = {"ee_twist": ServoCommandCfg(settings=settings, evaluation=evaluation,
                                               resampling_time_range=(5.0, 5.0) if settings.control_profile == "wbc" else (2.0, 5.0))}
    cfg.actions = {"whole_body": WholeBodyActionCfg(entity_name="robot")}
    actor_terms = {"servo": ObservationTermCfg(func=servo_observation)}
    if settings.ik_mode != "none":
        actor_terms["ik"] = ObservationTermCfg(func=prior_observation)
    if settings.preview != "none":
        actor_terms["preview"] = ObservationTermCfg(func=preview_observation)
    cfg.observations = {
        "actor": ObservationGroupCfg(terms=actor_terms, nan_policy="error"),
        "privileged": ObservationGroupCfg(terms={"dynamics": ObservationTermCfg(func=privileged_observation)},
                                           nan_policy="error"),
        "critic": ObservationGroupCfg(terms={**actor_terms, "dynamics": ObservationTermCfg(func=privileged_observation)},
                                       nan_policy="error"),
        "role": ObservationGroupCfg(terms={"teacher": ObservationTermCfg(func=role_observation)}),
    }
    if settings.encoder_mode != "none":
        cfg.observations["history"] = ObservationGroupCfg(
            terms={"deployable": ObservationTermCfg(func=servo_observation)},
            history_length=max(1, round(settings.history_seconds / 0.02)),
            flatten_history_dim=True, nan_policy="error")
    cfg.events = {name: term for name, term in cfg.events.items()
                  if name in ("reset_base", "reset_robot_joints")}
    if settings.control_profile == "standard":
        cfg.events["reset_arm_default"] = EventTermCfg(func=reset_arm_to_default, mode="reset")
    cfg.events["reset_base"].params["pose_range"]["yaw"] = (-math.pi, math.pi)
    cfg.events["dynamics"] = EventTermCfg(func=randomize_dynamics, mode="reset",
                                         params={"settings": settings})
    if settings.push_velocity:
        cfg.events["push"] = EventTermCfg(
            func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(3.0, 6.0),
            params={"velocity_range": {"x": (-settings.push_velocity, settings.push_velocity),
                                       "y": (-settings.push_velocity, settings.push_velocity)}})
    cfg.rewards = {
        "tracking": RewardTermCfg(func=tracking_reward, weight=3.0),
        "balance": RewardTermCfg(func=balance_penalty, weight=-1.0),
        "posture": RewardTermCfg(func=posture_penalty, weight=-0.1),
        "action_rate": RewardTermCfg(func=action_rate, weight=-settings.action_rate_weight),
        "energy": RewardTermCfg(func=energy, weight=-1e-5),
        "joint_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    }
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(55)
    cfg.sim.nconmax, cfg.sim.njmax = 192, 768
    if settings.control_profile == "wbc":
        cfg.sim.mujoco.timestep, cfg.decimation = 0.005, 4
        cfg.sim.nconmax, cfg.sim.njmax = 512, 2048
        cfg.sim.contact_sensor_maxmatch = 512
        cfg.scene.sensors += tuple(
            ContactSensorCfg(
                name=name, primary=ContactMatch(mode="body", pattern=patterns, entity="robot"),
                fields=("found", "force"), reduce="netforce", num_slots=1,
            )
            for name, patterns in (
                ("leg_contact", ("proximal_pitch_[LR]_Link", "proximal_roll_[LR]_Link")),
                ("torso_contact", ("transition_Link", "camera_mount_Link", "antenna_[LR]_Link")),
            )
        )
        cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)
        cfg.events["reset_base"].params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.5, 0.5),
        }
        cfg.events["reset_robot_joints"].params.update(
            position_range=(-0.2, 0.2), velocity_range=(-0.5, 0.5))
        cfg.terminations["base_height"] = TerminationTermCfg(func=wbc_fallen)
        if settings.task == "velocity":
            cfg.rewards = {
                f"track_{name}": RewardTermCfg(func=wbc_tracking, weight=weight, params={"axis": axis})
                for axis, (name, weight) in enumerate(zip(
                    ("vx", "vy", "z_height" if settings.height_tracking else "vz", "wx", "wy", "wz"),
                    (3.0, 2.0, 3.5, 2.0, 2.0, 2.0)))
            }
        else:
            cfg.rewards = {"tracking": cfg.rewards["tracking"]}
        cfg.rewards.update({
            "alive": RewardTermCfg(func=wbc_alive, weight=1.0),
            "action_rate": RewardTermCfg(func=action_rate, weight=-settings.action_rate_weight),
            "joint_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-2.0),
        })
        for kind, weight in (
            ("roll_velocity", -0.3), ("pitch_velocity", -0.1),
            ("base_acceleration", -0.002), ("arm_first", -0.02),
            ("action_jerk", -0.05), ("torque_rate", -1.5e-7),
            ("weighted_torque", -4e-6), ("weighted_power", -2.5e-6),
            ("wheel_idle", -0.0005), ("leg_velocity", -0.001),
            ("joint_acceleration", -2.5e-6), ("velocity_limits", -0.025),
            ("torque_limits", -1e-2), ("yaw_sum", -0.8), ("yaw_deviation", -0.5),
            ("knee_symmetry", -1.0), ("pitch_symmetry", -1.0),
            ("wheel_slip", -0.5), ("wheel_contact", 0.05),
            ("leg_contact", -0.2), ("torso_contact", -1.0),
        ):
            cfg.rewards[kind] = RewardTermCfg(func=wbc_penalty, weight=weight, params={"kind": kind})
    return cfg