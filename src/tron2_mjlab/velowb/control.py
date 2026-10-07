"""Command excitation, constrained outer regulation and geometric priors."""

import math

import torch
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_from_matrix

from .kinematics import ArmKinematics


def pose_error(target_position: torch.Tensor, target_rotation: torch.Tensor,
               position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    angular = axis_angle_from_quat(quat_from_matrix(
        target_rotation @ rotation.transpose(-1, -2)))
    return torch.cat((target_position - position, angular), dim=-1)


def waveform(kind: str, time: torch.Tensor, duration: torch.Tensor | float,
             start: torch.Tensor, target: torch.Tensor,
             frequency: torch.Tensor | float = 0.2,
             phase: torch.Tensor | float = 0.0) -> torch.Tensor:
    fraction = (time / duration).clamp(0.0, 1.0)
    if kind == "zero":
        return torch.zeros_like(target + time)
    if kind == "constant":
        return target + torch.zeros_like(time)
    if kind == "step":
        return torch.where(fraction < 0.25, start, target)
    if kind == "ramp":
        return start + fraction * (target - start)
    if kind == "trapezoid":
        envelope = torch.minimum(4 * fraction, 4 * (1 - fraction)).clamp(0, 1)
        return start + envelope * (target - start)
    if kind == "minimum_jerk":
        envelope = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
        return start + envelope * (target - start)
    angle = 2 * math.pi * frequency * time
    if kind == "sine":
        return target * torch.sin(angle + phase)
    if kind == "bandlimited":
        return target * (torch.sin(angle + phase)
                         + 0.5 * torch.sin(0.67 * angle + 1.7 * phase)
                         + 0.25 * torch.sin(0.37 * angle + 2.3 * phase)) / 1.75
    raise ValueError(f"Unknown command profile: {kind}")


def damped_least_squares(jacobian: torch.Tensor, command: torch.Tensor,
                        damping: float = 0.05,
                        joint_cost: torch.Tensor | None = None,
                        velocity_limit: float | torch.Tensor = 1.5) -> torch.Tensor:
    if damping <= 0:
        raise ValueError("IK damping must be positive")
    if joint_cost is not None and bool((joint_cost <= 0).any()):
        raise ValueError("IK joint costs must be positive")
    inverse_cost = 1.0 if joint_cost is None else joint_cost.reciprocal()
    weighted_transpose = jacobian.transpose(-1, -2) * (
        inverse_cost if joint_cost is None else inverse_cost[..., None])
    identity = torch.eye(6, device=jacobian.device, dtype=jacobian.dtype)
    solution = torch.linalg.solve(jacobian @ weighted_transpose + damping**2 * identity,
                                  command.unsqueeze(-1))
    return (weighted_transpose @ solution).squeeze(-1).clamp(
        -velocity_limit, velocity_limit)


def limit_joint_velocity(position: torch.Tensor, velocity: torch.Tensor,
                         limits: torch.Tensor, dt: float) -> torch.Tensor:
    return torch.maximum(torch.minimum(velocity, (limits[:, 1] - position) / dt),
                         (limits[:, 0] - position) / dt)


def kinematic_preview(chain: ArmKinematics, position: torch.Tensor,
                      command_base: torch.Tensor, mode: str = "full",
                      horizons: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
                      initial_velocity: torch.Tensor | None = None) -> torch.Tensor:
    if mode not in ("none", "joint", "joint_ee", "full"):
        raise ValueError(f"Unknown preview mode: {mode}")
    if mode == "none":
        return position[:, :0]
    if not horizons or any(end <= start for start, end in zip((0.0,) + horizons, horizons)):
        raise ValueError("Preview horizons must be positive and strictly increasing")
    predicted = position.clone()
    previous = 0.0
    features = []
    for horizon in horizons:
        dt = horizon - previous
        state = chain.forward(predicted)
        velocity = (initial_velocity if previous == 0.0 and initial_velocity is not None
                else damped_least_squares(state.jacobian, command_base))
        velocity = limit_joint_velocity(predicted, velocity, chain.limits, dt)
        predicted = predicted + dt * velocity
        state = chain.forward(predicted)
        features.append(predicted)
        if mode in ("joint_ee", "full"):
            features.extend((state.position, state.rotation[:, :, :2].flatten(1),
                             (state.jacobian @ velocity.unsqueeze(-1)).squeeze(-1)))
        if mode == "full":
            features.extend((state.arm_com, state.jacobian.flatten(1)))
        previous = horizon
    return torch.cat(features, dim=-1)


class PoseRegulator:
    def __init__(self, num_envs: int, device: str = "cpu", dt: float = 0.02,
                 limits: tuple[float, ...] = (0.15, 0.1, 0.1, 0.25, 0.25, 0.3),
                 kp: float = 2.0, kd: float = 0.15, cutoff_hz: float = 3.0,
                 slew: tuple[float, ...] = (0.5, 0.5, 0.5, 1.0, 1.0, 1.0)):
        if dt <= 0 or cutoff_hz <= 0 or len(limits) != 6 or min(limits) <= 0:
            raise ValueError("PD requires positive dt, bandwidth and six command limits")
        if len(slew) != 6 or min(slew) <= 0 or kp < 0 or kd < 0:
            raise ValueError("PD gains must be nonnegative and slew limits positive")
        self.dt, self.kp, self.kd = dt, kp, kd
        self.limits = torch.tensor(limits, device=device)
        self.slew = torch.tensor(slew, device=device)
        self.alpha = 1 - math.exp(-2 * math.pi * cutoff_hz * dt)
        self.command = torch.zeros((num_envs, 6), device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.command.zero_()
        else:
            self.command[env_ids] = 0

    def step(self, target_position: torch.Tensor, target_rotation: torch.Tensor,
             position: torch.Tensor, rotation: torch.Tensor, twist: torch.Tensor,
             feedforward: torch.Tensor | None = None) -> torch.Tensor:
        if feedforward is None:
            feedforward = torch.zeros_like(twist)
        error = pose_error(target_position, target_rotation, position, rotation)
        target = self.kp * error + self.kd * (feedforward - twist) + feedforward
        target = target.clamp(-self.limits, self.limits)
        delta = (self.alpha * (target - self.command)).clamp(
            -self.dt * self.slew, self.dt * self.slew)
        self.command += delta
        return self.command.clone()


class DelayBuffer:
    """Per-environment integer sensor/action delay, with explicit reset isolation."""

    def __init__(self, max_delay: int, initial: torch.Tensor):
        if max_delay < 0:
            raise ValueError("Delay must be nonnegative")
        self.buffer = initial.unsqueeze(0).repeat(max_delay + 1, 1, 1)
        self.cursor = 0
        self.env_ids = torch.arange(initial.shape[0], device=initial.device)

    def reset(self, env_ids: torch.Tensor, initial: torch.Tensor):
        self.buffer[:, env_ids] = initial[env_ids]

    def push(self, value: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
        self.cursor = (self.cursor + 1) % len(self.buffer)
        self.buffer[self.cursor] = value
        return self.read(delay)

    def read(self, delay: torch.Tensor) -> torch.Tensor:
        if bool(((delay < 0) | (delay >= len(self.buffer))).any()):
            raise ValueError("Delay exceeds allocated buffer")
        indices = (self.cursor - delay) % len(self.buffer)
        return self.buffer[indices, self.env_ids].clone()