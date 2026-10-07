"""Selective opposite-foot collision filtering and physical contact response."""

import unittest
from itertools import combinations

import mujoco
import numpy as np
from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg

from tron2_mjlab.robot import get_robot_cfg


class FootCollisionTests(unittest.TestCase):
    def test_only_opposite_feet_pass_self_collision_filter(self):
        for variant in ("LBC", "Baseline"):
            cfg = load_env_cfg(f"Mjlab-Velocity-Flat-TRON2A-SFYG-{variant}-CTS")
            model = Entity(cfg.scene.entities["robot"]).spec.compile()
            left, right = set(), set()
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
                # Preserve matching against ordinary terrain (type=1, affinity=1).
                self.assertTrue(model.geom_conaffinity[geom_id] & 1)
            self.assertTrue(left)
            self.assertTrue(right)
            for first, second in combinations(range(model.ngeom), 2):
                matches = bool(
                    (model.geom_contype[first] & model.geom_conaffinity[second])
                    | (model.geom_contype[second] & model.geom_conaffinity[first])
                )
                expected = (first in left and second in right) or (first in right and second in left)
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
