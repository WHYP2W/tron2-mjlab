"""Flat velocity tracking for TRON2A SFYG."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from .robot import LEG_JOINTS, VARIANTS, get_robot_cfg


def tron2_flat_env_cfg(variant: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown TRON2A variant: {variant}")
    cfg = make_velocity_env_cfg()
    foot_bodies = tuple(f"ankle_pitch_{side}_Link" for side in ("L", "R"))
    foot_geoms = (r"ankle_pitch_[LR]_Link_collision_.*",)
    sites = ("foot_L", "foot_R")
    cfg.scene.entities = {"robot": get_robot_cfg(variant)}
    cfg.scene.num_envs = 1 if play else 1024
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.clear()
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="body", pattern=foot_bodies, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="nonfoot_ground_touch",
            primary=ContactMatch(
                mode="geom", pattern=".*_collision_.*", exclude=foot_geoms,
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",), reduce="none", num_slots=1,
        ),
    )
    cfg.actions = {"joint_pos": JointPositionActionCfg(
        entity_name="robot", actuator_names=LEG_JOINTS, scale=0.25,
        use_default_offset=True,
    )}
    for group in cfg.observations.values():
        group.terms.pop("height_scan", None)
        group.terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINTS,
        )
    cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = sites
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geoms
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_Link",)
    cfg.events.pop("push_robot")
    cfg.rewards.pop("angular_momentum")
    cfg.rewards["pose"].params["asset_cfg"].joint_names = LEG_JOINTS
    for regime, std in (("standing", 0.1), ("walking", 0.3), ("running", 0.5)):
        cfg.rewards["pose"].params[f"std_{regime}"] = {".*": std}
    for name in ("upright", "body_ang_vel"):
        cfg.rewards[name].params["asset_cfg"].body_names = ("base_Link",)
    for name in ("foot_clearance", "foot_swing_height", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = sites
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact, params={"sensor_name": "nonfoot_ground_touch"},
    )
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (-0.5, 1.0)
    command.ranges.lin_vel_y = (-0.3, 0.3)
    command.viz.z_offset = 0.5
    cfg.viewer.body_name = "base_Link"
    cfg.viewer.distance = 2.5
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 512
    cfg.sim.contact_sensor_maxmatch = 128
    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.episode_length_s = int(1e9)
    return cfg