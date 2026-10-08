"""Foot-foot and foot-body collision filtering and physical response."""

import unittest
from itertools import combinations

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from tron2_mjlab.robot import get_robot_cfg


class FootCollisionTests(unittest.TestCase):
    def test_feet_collide_with_each_other_and_other_robot_parts(self):
        for variant in ("LBC", "Baseline"):
            cfg = load_env_cfg(f"Mjlab-Velocity-Flat-TRON2A-SFYG-{variant}-CTS")
            model = Entity(cfg.scene.entities["robot"]).spec.compile()
            left, right, other = set(), set(), set()
            for geom_id in range(model.ngeom):
                name = model.geom(geom_id).name
                if "_collision_" not in name:
                    self.assertEqual(model.geom_contype[geom_id], 0)
                    self.assertEqual(model.geom_conaffinity[geom_id], 0)
                    continue
                if name.startswith("ankle_pitch_L_Link_collision_"):
                    left.add(geom_id)
                elif name.startswith("ankle_pitch_R_Link_collision_"):
                    right.add(geom_id)
                else:
                    other.add(geom_id)
                # Preserve matching against ordinary terrain (type=1, affinity=1).
                self.assertTrue(model.geom_conaffinity[geom_id] & 1)
            self.assertTrue(left)
            self.assertTrue(right)
            self.assertTrue(other)
            feet = left | right
            for first, second in combinations(range(model.ngeom), 2):
                matches = bool(
                    (model.geom_contype[first] & model.geom_conaffinity[second])
                    | (model.geom_contype[second] & model.geom_conaffinity[first])
                )
                expected = (first in left and second in right) or (first in right and second in left)
                expected |= (
                    (first in feet and second in other)
                    or (second in feet and first in other)
                )
                self.assertEqual(matches, expected, (model.geom(first).name, model.geom(second).name))
            for sensor in cfg.scene.sensors:
                self.assertEqual(sensor.secondary.pattern, "terrain")

    def test_initial_clearance_and_crossed_feet_response(self):
        model = Entity(get_robot_cfg("SFYG")).spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        self.assertEqual(data.ncon, 0)

        for side, angle in (("L", -0.2), ("R", 0.2)):
            joint = model.joint(f"proximal_roll_{side}_Joint")
            self.assertTrue(joint.range[0] <= angle <= joint.range[1])
            data.qpos[joint.qposadr[0]] = angle
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 0)
        force = np.zeros(6)
        normal_forces = []
        for index, contact in enumerate(data.contact):
            names = {model.body(model.geom_bodyid[geom]).name for geom in contact.geom}
            self.assertEqual(names, {"ankle_pitch_L_Link", "ankle_pitch_R_Link"})
            mujoco.mj_contactForce(model, data, index, force)
            normal_forces.append(force[0])
        self.assertGreater(max(normal_forces), 0)
        for _ in range(20):
            mujoco.mj_step(model, data)
            self.assertTrue(np.isfinite(data.qpos).all())
            self.assertTrue(np.isfinite(data.qvel).all())

        # Reproduce the old filter: identical crossed pose no longer makes contact.
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[model.joint("proximal_roll_L_Joint").qposadr[0]] = -0.2
        data.qpos[model.joint("proximal_roll_R_Joint").qposadr[0]] = 0.2
        model.geom_contype[:] = 0
        mujoco.mj_forward(model, data)
        self.assertEqual(data.ncon, 0)

    def test_both_feet_can_contact_the_torso(self):
        for side, sign in (("L", 1), ("R", -1)):
            with self.subTest(side=side):
                model = Entity(get_robot_cfg("SFYG")).spec.compile()
                data = mujoco.MjData(model)
                mujoco.mj_resetDataKeyframe(model, data, 0)
                pose = {
                    "proximal_pitch": -2.5722178, "proximal_roll": sign * 0.4534898,
                    "proximal_yaw": sign * -1.4886508, "knee": -2.1107489,
                    "ankle_pitch": -1.1859066,
                }
                for name, angle in pose.items():
                    joint = model.joint(f"{name}_{side}_Joint")
                    self.assertTrue(joint.range[0] <= angle <= joint.range[1])
                    data.qpos[joint.qposadr[0]] = angle
                mujoco.mj_forward(model, data)
                forces = []
                for index, contact in enumerate(data.contact):
                    bodies = {model.body(model.geom_bodyid[geom]).name for geom in contact.geom}
                    if bodies == {f"ankle_pitch_{side}_Link", "base_Link"}:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, index, force)
                        forces.append(force[0])
                self.assertTrue(forces)
                self.assertGreater(max(forces), 0)
                for geom_id in range(model.ngeom):
                    if model.geom(geom_id).name.startswith("ankle_pitch_"):
                        model.geom_conaffinity[geom_id] &= ~8
                mujoco.mj_forward(model, data)
                for contact in data.contact:
                    bodies = {model.body(model.geom_bodyid[geom]).name for geom in contact.geom}
                    self.assertNotEqual(bodies, {f"ankle_pitch_{side}_Link", "base_Link"})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for foot-body contacts")
    def test_gpu_foot_body_contacts_are_not_ground_support(self):
        cfg = load_env_cfg("Mjlab-Velocity-Flat-TRON2A-SFYG-Baseline-CTS", play=True)
        cfg.scene.num_envs = 2
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
        try:
            env.reset()
            robot = env.scene["robot"]
            for index, side, sign in ((0, "L", 1), (1, "R", -1)):
                names = tuple(f"{name}_{side}_Joint" for name in (
                    "proximal_pitch", "proximal_roll", "proximal_yaw", "knee", "ankle_pitch",
                ))
                ids, _ = robot.find_joints(names, preserve_order=True)
                robot.write_joint_position_to_sim(
                    torch.tensor([[-2.5722178, sign * 0.4534898, sign * -1.4886508,
                                   -2.1107489, -1.1859066]], device=env.device),
                    joint_ids=torch.tensor(ids, device=env.device),
                    env_ids=torch.tensor([index], device=env.device),
                )
            env.sim.forward()
            env.sim.sense()
            count = int(wp.to_torch(env.sim.wp_data.nacon).item())
            geoms = wp.to_torch(env.sim.wp_data.contact.geom)[:count].cpu().tolist()
            worlds = wp.to_torch(env.sim.wp_data.contact.worldid)[:count].cpu().tolist()
            contacting = set()
            for pair, world in zip(geoms, worlds):
                bodies = {env.sim.mj_model.body(env.sim.mj_model.geom_bodyid[geom]).name for geom in pair}
                if "robot/base_Link" in bodies and any("ankle_pitch_" in name for name in bodies):
                    contacting.add(world)
            self.assertEqual(contacting, {0, 1})
            for name in ("feet_ground_contact", "nonfoot_ground_touch"):
                sensor = env.scene[name]
                sensor.update(0.0)
                self.assertFalse(sensor.data.found.any())
        finally:
            env.close()
