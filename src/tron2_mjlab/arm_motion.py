"""Smooth random arm targets independent of the locomotion policy."""

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .domain_randomization import sample_arm_pose
from .robot import ARM_JOINTS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class RandomArmMotionCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (2.0, 4.0)
    range_fraction: float = 0.8
    nominal_probability: float = 0.2
    max_target_speed: float = 1.0

    def __post_init__(self):
        low, high = self.resampling_time_range
        if not all(isfinite(value) for value in (low, high)) or not 0 < low <= high:
            raise ValueError("Arm motion durations must be finite, positive and ordered")
        if not isfinite(self.max_target_speed) or self.max_target_speed <= 0:
            raise ValueError("Arm target speed must be finite and positive")
        if not isfinite(self.range_fraction) or not 0 < self.range_fraction <= 1:
            raise ValueError("Arm motion range_fraction must be in (0, 1]")
        if not isfinite(self.nominal_probability) or not 0 <= self.nominal_probability <= 1:
            raise ValueError("Arm motion nominal_probability must be in [0, 1]")

    def build(self, env: "ManagerBasedRlEnv") -> "RandomArmMotion":
        return RandomArmMotion(self, env)


class RandomArmMotion(CommandTerm):
    cfg: RandomArmMotionCfg

    def __init__(self, cfg: RandomArmMotionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        joint_ids, names = self.robot.find_joints(ARM_JOINTS, preserve_order=True)
        if tuple(names) != ARM_JOINTS:
            raise ValueError("Random arm motion requires all six ordered arm joints")
        self.joint_ids = torch.tensor(joint_ids, device=self.device)
        self.start = self.robot.data.joint_pos_target[:, self.joint_ids].clone()
        self.goal = self.start.clone()
        self._command = self.start.clone()
        self.duration = torch.ones(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.start[env_ids] = self.robot.data.joint_pos_target[env_ids][:, self.joint_ids]
        self._command[env_ids] = self.start[env_ids]
        self.goal[env_ids] = sample_arm_pose(
            self.robot.data.soft_joint_pos_limits[env_ids][:, self.joint_ids],
            self.robot.data.default_joint_pos[env_ids][:, self.joint_ids],
            self.cfg.range_fraction, self.cfg.nominal_probability,
        )
        # Quintic smoothstep has a peak derivative of 1.875.
        minimum_duration = (
            1.875 * (self.goal[env_ids] - self.start[env_ids]).abs().amax(dim=1)
            / self.cfg.max_target_speed
        )
        self.duration[env_ids] = torch.maximum(self.time_left[env_ids], minimum_duration)
        self.time_left[env_ids] = self.duration[env_ids]

    def _update_command(self) -> None:
        phase = (1 - self.time_left / self.duration).clamp(0, 1)
        blend = (phase.pow(3) * (10 + phase * (-15 + 6 * phase))).clamp(0, 1)
        self._command.copy_(self.start + blend[:, None] * (self.goal - self.start))
        self.robot.set_joint_position_target(self._command, joint_ids=self.joint_ids)

    def compute(self, dt: float) -> None:
        if not isfinite(dt) or dt <= 0:
            raise ValueError("Arm motion time step must be finite and positive")
        active = self._env.episode_length_buf > 0
        self.time_left -= dt * active
        # Reach the old endpoint before sampling the next segment to avoid target jumps.
        self._update_command()
        finished = (active & (self.time_left <= 0)).nonzero().flatten()
        self._resample(finished)
