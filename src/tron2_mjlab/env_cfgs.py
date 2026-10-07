"""Flat velocity tracking for TRON2A SFYG with an independently controlled arm.

Actor observations (56): base linear/angular velocity (3+3), gravity (3),
joint positions relative to defaults (10 legs + 6 arm), joint velocities
(10 legs + 6 arm), previous leg actions (10), gait phase (2), and velocity command (3).
Arm values follow arm1_Joint through arm6_Joint. Critic observations add
12 foot-state values (68 total). Training adds joint observation noise;
playback does not. The policy controls only the 10 leg position targets.
With observe_arm=False, both networks omit arm positions and velocities:
Actor/Critic dimensions become 44/56, with otherwise identical dynamics.

The arm uses independent position targets, initially zero; no IK solver or
arm-motion curriculum is provided here. Train with representative external
arm trajectories before expecting robustness to IK motion. The observation
and dynamics changes require retraining old fixed-arm policies.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from .robot import ARM_JOINTS, LEG_JOINTS, VARIANTS, get_robot_cfg
from .gait import GaitVelocityCommandCfg, alternating_contacts, gait_phase


def tron2_flat_env_cfg(
    variant: str, play: bool = False, *, observe_arm: bool = True,
) -> ManagerBasedRlEnvCfg:
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
    gait_params = {"command_name": "twist"}
    for group in cfg.observations.values():
        group.terms.pop("height_scan", None)
        for term in ("joint_pos", "joint_vel"):
            group.terms[term].params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=LEG_JOINTS + ARM_JOINTS if observe_arm else LEG_JOINTS,
                preserve_order=True,
            )
        terms = {}
        for name, term in group.terms.items():
            if name == "command":
                terms["gait_phase"] = ObservationTermCfg(func=gait_phase, params=gait_params.copy())
            terms[name] = term
        group.terms = terms
    cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = sites
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geoms
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_Link",)
    cfg.events["reset_robot_joints"].params["asset_cfg"].joint_names = LEG_JOINTS
    cfg.events.pop("push_robot")
    cfg.rewards.pop("angular_momentum")
    cfg.rewards["pose"].params["asset_cfg"].joint_names = LEG_JOINTS
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=LEG_JOINTS,
    )
    for regime, std in (("standing", 0.1), ("walking", 0.3), ("running", 0.5)):
        cfg.rewards["pose"].params[f"std_{regime}"] = {".*": std}
    for name in ("upright", "body_ang_vel"):
        cfg.rewards[name].params["asset_cfg"].body_names = ("base_Link",)
    for name in ("foot_clearance", "foot_swing_height", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = sites
    cfg.rewards["gait_contact"] = RewardTermCfg(
        func=alternating_contacts, weight=1.0,
        params={**gait_params, "sensor_name": "feet_ground_contact", "stance_ratio": 0.6},
    )
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact, params={"sensor_name": "nonfoot_ground_touch"},
    )
    command = GaitVelocityCommandCfg(**vars(cfg.commands["twist"]))
    cfg.commands["twist"] = command
    command.ranges.lin_vel_x = (-1.0, 2.0)
    command.ranges.lin_vel_y = (-0.6, 0.6)
    command.ranges.ang_vel_z = (-1.0, 1.0)
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