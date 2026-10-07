"""Model preparation for TRON2A SFYG."""

import xml.etree.ElementTree as ET
from functools import partial
from math import pi
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

VARIANTS = ("SFYG",)
ASSET_ROOT = Path(__file__).parent / "tron2-robot-description" / "tron2a"
LEG_JOINTS = (r"(?:proximal_.*|knee|ankle_pitch)_[LR]_Joint",)
ARM_JOINTS = tuple(f"arm{index}_Joint" for index in range(1, 7))


def get_spec(variant: str) -> mujoco.MjSpec:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown TRON2A variant: {variant}")
    model_dir = ASSET_ROOT / f"{variant}_TRON2A"
    root = ET.parse(model_dir / "xml" / "robot.xml").getroot()
    defaults = root.find("default")
    assert defaults is not None
    ET.SubElement(defaults, "general", ctrlrange="-30 30")
    for tag in ("actuator", "sensor", "keyframe", "option", "size"):
        for element in root.findall(tag):
            root.remove(element)
    world = root.find("worldbody")
    assert world is not None
    for element in list(world):
        if element.tag != "body":
            world.remove(element)
    for body in world.iter("body"):
        for index, geom in enumerate(body.findall("geom")):
            kind = "visual" if geom.get("class") == "visual" else "collision"
            geom.set("name", f"{body.get('name')}_{kind}_{index}")
        for joint in body.findall("joint"):
            name = joint.get("name", "")
            if name.startswith("gripper"):
                body.remove(joint)
        if body.get("name", "").startswith("ankle_pitch_"):
            side = body.get("name").split("_")[-2]
            ET.SubElement(body, "site", name=f"foot_{side}", pos="-0.014 0 -0.074")
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str(model_dir / "meshes"))
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "velocimeter", name="imu_lin_vel", site="base_imu")
    ET.SubElement(sensors, "gyro", name="imu_ang_vel", site="base_imu")
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def get_robot_cfg(variant: str) -> EntityCfg:
    """Use leg policy targets and independent arm position targets.

    The six arm joints start at zero and are held by position actuators.
    An external IK controller can use robot.find_joints(ARM_JOINTS,
    preserve_order=True) and robot.set_joint_position_target(q_ik,
    joint_ids=arm_ids) before env.step(). Targets are in radians and must
    respect joint limits. They persist until replaced or the environment
    resets to zero; the leg policy does not overwrite them.
    """
    actuators = [
        BuiltinPositionActuatorCfg(
            target_names_expr=(r"(?:proximal_(?:pitch|roll)|knee)_[LR]_Joint",),
            stiffness=200.0, damping=10.0, effort_limit=150.0,
            armature=0.0685894,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(r"(?:proximal_yaw|ankle_pitch)_[LR]_Joint",),
            stiffness=80.0, damping=4.0, effort_limit=60.0,
            armature=0.0169834,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=ARM_JOINTS,
            stiffness=60.0, damping=3.0, effort_limit=30.0,
            armature=0.0110718,
        ),
    ]
    joint_pos = {"proximal_pitch_.*": 0.3, "knee_.*": -0.6,
                 "proximal_roll_.*": 0.0, "ankle_pitch_.*": 0.3,
                 "proximal_yaw_L_Joint": -pi, "proximal_yaw_R_Joint": pi}
    joint_pos.update({name: 0.0 for name in ARM_JOINTS})
    return EntityCfg(
        spec_fn=partial(get_spec, variant),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.72),
            joint_pos=joint_pos, joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9,
        ),
        collisions=(CollisionCfg(
            geom_names_expr=(".*_collision_.*",),
            contype=0, conaffinity=1, condim=3, friction=(0.8, 0.005, 0.0001),
        ),),
    )