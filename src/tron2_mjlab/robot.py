"""Shared model preparation for the two TRON2A YG variants."""

import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

VARIANTS = ("SFYG", "WFYG")
ASSET_ROOT = Path(__file__).parent / "tron2-robot-description" / "tron2a"
LEG_JOINTS = (r"(?:proximal_.*|knee|ankle_pitch)_[LR]_Joint",)


def get_spec(
    variant: str, *, active_arm: bool = False, lock_arm_yaw: bool = True,
    control_profile: str = "standard",
) -> mujoco.MjSpec:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown TRON2A variant: {variant}")
    if control_profile not in ("standard", "wbc"):
        raise ValueError(f"Unknown control profile: {control_profile}")
    if control_profile == "wbc" and (variant != "WFYG" or not active_arm or not lock_arm_yaw):
        raise ValueError("WBC profile requires WFYG with an active arm and locked arm1")
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
            if (name.startswith("gripper")
                    or (name.startswith("arm") and not active_arm)
                    or (name == "arm1_Joint" and active_arm and lock_arm_yaw)
                    or (name == "arm6_Joint" and control_profile == "wbc")):
                body.remove(joint)
            elif name.startswith("wheel_"):
                joint.set("limited", "false")
        if active_arm and body.get("name") == "gripper_pick":
            ET.SubElement(body, "site", name="ee", pos="0 0 0")
            payload = ET.SubElement(body, "body", name="payload")
            ET.SubElement(payload, "inertial", pos="0 0 0", mass="0.0001",
                          diaginertia="0.000000025 0.000000025 0.000000025")
        if body.get("name", "").startswith(("ankle_pitch_", "wheel_")):
            side = body.get("name").split("_")[-2]
            position = "0 0 0" if variant == "WFYG" else "-0.014 0 -0.074"
            ET.SubElement(body, "site", name=f"foot_{side}", pos=position)
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str(model_dir / "meshes"))
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "velocimeter", name="imu_lin_vel", site="base_imu")
    ET.SubElement(sensors, "gyro", name="imu_ang_vel", site="base_imu")
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def get_robot_cfg(
    variant: str, *, active_arm: bool = False, lock_arm_yaw: bool = True,
    control_profile: str = "standard",
) -> EntityCfg:
    if control_profile not in ("standard", "wbc"):
        raise ValueError(f"Unknown control profile: {control_profile}")
    reference = control_profile == "wbc"
    if reference and (variant != "WFYG" or not active_arm or not lock_arm_yaw):
        raise ValueError("WBC profile requires WFYG with an active arm and locked arm1")
    wheeled = variant == "WFYG"
    actuators = [
        BuiltinPositionActuatorCfg(
            target_names_expr=(r"(?:proximal_(?:pitch|roll)|knee)_[LR]_Joint",),
            stiffness=200.0, damping=10.0, effort_limit=150.0,
            armature=0.0685894,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(r"(?:proximal_yaw|ankle_pitch)_[LR]_Joint",)
            if not wheeled else (r"proximal_yaw_[LR]_Joint",),
            stiffness=80.0, damping=4.0, effort_limit=60.0,
            armature=0.0169834,
        ),
    ]
    joint_pos = {"proximal_pitch_.*": 0.3, "knee_.*": -0.6,
                 "proximal_(roll|yaw)_.*": 0.0}
    if wheeled:
        actuators.append(BuiltinVelocityActuatorCfg(
            target_names_expr=("wheel_.*",), damping=2.0, effort_limit=22.0,
            armature=0.0110718,
        ))
        joint_pos["wheel_.*"] = 0.0
    else:
        joint_pos["ankle_pitch_.*"] = 0.3
    if active_arm:
        arm_pattern = r"arm[2-6]_Joint" if lock_arm_yaw else r"arm[1-6]_Joint"
        actuators.append(BuiltinPositionActuatorCfg(
            target_names_expr=(arm_pattern,), stiffness=60.0, damping=3.0,
            effort_limit=30.0, armature=0.0110718,
        ))
        joint_pos[arm_pattern] = 0.0
    if reference:
        actuators = [
            BuiltinPositionActuatorCfg(
                target_names_expr=(r"(?:proximal_(?:pitch|roll)|knee)_[LR]_Joint",),
                stiffness=139.41, damping=17.75, effort_limit=130.0,
                armature=2.26e-4 * 25.0**2,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(r"proximal_yaw_[LR]_Joint",),
                stiffness=182.14, damping=23.19, effort_limit=60.0,
                armature=1.424e-4 * 36.0**2,
            ),
            BuiltinVelocityActuatorCfg(
                target_names_expr=(r"wheel_[LR]_Joint",), damping=1.32,
                effort_limit=22.0, armature=2.15e-4 * 7.0**2, frictionloss=0.01,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(r"arm[23]_Joint",), stiffness=50.0, damping=2.0,
                effort_limit=100.0, armature=0.0,
            ),
            BuiltinPositionActuatorCfg(
                target_names_expr=(r"arm[45]_Joint",), stiffness=30.0, damping=3.0,
                effort_limit=100.0, armature=0.0,
            ),
        ]
        joint_pos = {".*": 0.0}
    return EntityCfg(
        spec_fn=partial(get_spec, variant, active_arm=active_arm,
                        lock_arm_yaw=lock_arm_yaw, control_profile=control_profile),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.85 if reference else 0.75 if wheeled else 0.72),
            joint_pos=joint_pos, joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(actuators), soft_joint_pos_limit_factor=0.95 if reference else 0.9,
        ),
        collisions=(CollisionCfg(
            geom_names_expr=(".*_collision_.*",),
            contype=1 if reference else 0, conaffinity=1, condim=3, friction=(0.8, 0.005, 0.0001),
        ),),
    )