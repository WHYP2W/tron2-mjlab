"""Arm posture sampling and dynamics randomization for locomotion training."""

from math import isfinite
from typing import TYPE_CHECKING

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def randomize_arm_pose(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    range_fraction: float = 0.8,
    nominal_probability: float = 0.2,
) -> None:
    """Sample episode arm poses and matching independent position targets."""
    if not isfinite(range_fraction) or not 0 < range_fraction <= 1:
        raise ValueError("Arm pose range_fraction must be in (0, 1]")
    if not isfinite(nominal_probability) or not 0 <= nominal_probability <= 1:
        raise ValueError("Arm nominal_probability must be in [0, 1]")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene[asset_cfg.name]
    joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=env.device)
    limits = robot.data.soft_joint_pos_limits[env_ids][:, joint_ids]
    midpoint = limits.mean(dim=-1)
    half_range = (limits[..., 1] - limits[..., 0]) * (0.5 * range_fraction)
    position = midpoint + (2 * torch.rand_like(midpoint) - 1) * half_range
    nominal = torch.rand((len(env_ids), 1), device=env.device) < nominal_probability
    position = torch.where(nominal, robot.data.default_joint_pos[env_ids][:, joint_ids], position)
    robot.write_joint_state_to_sim(
        position, torch.zeros_like(position), joint_ids=joint_ids, env_ids=env_ids,
    )
    robot.set_joint_position_target(position, joint_ids=joint_ids, env_ids=env_ids[:, None])


@requires_model_fields(
    "body_mass", "body_inertia", "dof_damping", "actuator_gainprm", "actuator_biasprm",
    "body_subtreemass", "body_invweight0", "dof_invweight0", "actuator_acc0",
    "tendon_length0", "tendon_invweight0", "cam_pos0", "cam_poscom0", "cam_mat0",
    "light_pos0", "light_poscom0", "light_dir0",
)
def randomize_arm_dynamics(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float],
    damping_range: tuple[float, float],
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
) -> None:
    for name, bounds in (
        ("mass", mass_range), ("damping", damping_range), ("kp", kp_range), ("kd", kd_range),
    ):
        if not all(isfinite(value) for value in bounds) or not 0 < bounds[0] <= bounds[1]:
            raise ValueError(f"Arm {name} scale range must be finite, positive and ordered")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene[asset_cfg.name]
    body_ids = robot.indexing.body_ids[asset_cfg.body_ids]
    dof_ids = robot.indexing.joint_v_adr[asset_cfg.joint_ids]
    ctrl_ids = robot.indexing.ctrl_ids[asset_cfg.actuator_ids]
    model = env.sim.model

    def sample(bounds, count):
        return torch.empty((len(env_ids), count), device=env.device).uniform_(*bounds)

    mass_scale = sample(mass_range, len(body_ids))
    body_index = (env_ids[:, None], body_ids)
    model.body_mass[body_index] = env.sim.get_default_field("body_mass")[body_ids] * mass_scale
    model.body_inertia[body_index] = (
        env.sim.get_default_field("body_inertia")[body_ids] * mass_scale[..., None]
    )
    model.dof_damping[env_ids[:, None], dof_ids] = (
        env.sim.get_default_field("dof_damping")[dof_ids] * sample(damping_range, len(dof_ids))
    )
    kp_scale, kd_scale = sample(kp_range, len(ctrl_ids)), sample(kd_range, len(ctrl_ids))
    gain = env.sim.get_default_field("actuator_gainprm")
    bias = env.sim.get_default_field("actuator_biasprm")
    model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = gain[ctrl_ids, 0] * kp_scale
    model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = bias[ctrl_ids, 1] * kp_scale
    model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = bias[ctrl_ids, 2] * kd_scale

    # set_const also writes meaninertia; its default shared storage would race across worlds.
    wp_model = env.sim.wp_model
    expanded = wp_model.stat.meaninertia.shape[0] != env.num_envs
    if expanded:
        wp_model.stat.meaninertia = wp.full(
            env.num_envs, env.sim.mj_model.stat.meaninertia,
            dtype=wp.float32, device=wp_model.body_mass.device,
        )
    mjwarp.set_const(wp_model, env.sim.wp_data)
    if expanded:
        env.sim.create_graph()
    env.sim.forward()
