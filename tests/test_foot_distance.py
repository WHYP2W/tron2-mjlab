"""Foot-separation penalty boundaries, invariance and physical model checks."""

import unittest
from math import pi
from types import SimpleNamespace

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg

from tron2_mjlab.gait import feet_too_close
from tron2_mjlab.robot import get_robot_cfg


class FootDistanceTests(unittest.TestCase):
    def setUp(self):
        self.feet = torch.zeros((5, 2, 3), dtype=torch.float64)
        self.feet[:, 1, 0] = torch.tensor([0.0, 0.09, 0.18, 0.25, 0.09], dtype=torch.float64)
        self.feet[4, 1, 2] = 1.0
        self.env = SimpleNamespace(
            num_envs=5, extras={"log": {}},
            scene={"robot": SimpleNamespace(data=SimpleNamespace(site_pos_w=self.feet))},
        )
        self.asset = SceneEntityCfg("robot", site_ids=[0, 1])

    def test_distance_boundaries_and_lifted_foot(self):
        cost = feet_too_close(self.env, self.asset, min_distance=0.18)
        torch.testing.assert_close(cost, torch.tensor([1, 0.25, 0, 0, 0.25], dtype=torch.float64))
        self.assertAlmostEqual(self.env.extras["log"]["Metrics/feet_distance_xy"].item(), 0.122)
        self.assertAlmostEqual(self.env.extras["log"]["Metrics/feet_too_close_fraction"].item(), 0.6)

    def test_translation_rotation_and_foot_order_invariance(self):
        before = feet_too_close(self.env, self.asset, 0.18)
        angle = torch.tensor(pi / 3, dtype=torch.float64)
        rotation = torch.stack((
            torch.stack((angle.cos(), -angle.sin())),
            torch.stack((angle.sin(), angle.cos())),
        ))
        self.feet[:, :, :2] = self.feet[:, :, :2] @ rotation
        self.feet += torch.tensor([12.0, -5.0, 2.0], dtype=torch.float64)
        torch.testing.assert_close(feet_too_close(self.env, self.asset, 0.18), before)
        torch.testing.assert_close(
            feet_too_close(self.env, SceneEntityCfg("robot", site_ids=[1, 0]), 0.18), before,
        )

    def test_invalid_configuration(self):
        for threshold in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                feet_too_close(self.env, self.asset, threshold)
        with self.assertRaises(ValueError):
            feet_too_close(self.env, SceneEntityCfg("robot", site_ids=[0]), 0.18)

    def test_both_tasks_and_playback_share_penalty(self):
        for variant in ("LBC", "Baseline"):
            for play in (False, True):
                cfg = load_env_cfg(f"Mjlab-Velocity-Flat-TRON2A-SFYG-{variant}-CTS", play=play)
                term = cfg.rewards["feet_too_close"]
                self.assertIs(term.func, feet_too_close)
                self.assertEqual(term.weight, -1.0)
                self.assertEqual(term.params["min_distance"], 0.18)
                self.assertEqual(term.params["asset_cfg"].site_names, ("foot_L", "foot_R"))

    def test_default_stance_is_free_and_crossed_feet_are_penalized(self):
        model = Entity(get_robot_cfg("SFYG")).spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        ids = [model.site(name).id for name in ("foot_L", "foot_R")]
        positions = torch.from_numpy(data.site_xpos[ids].copy()).unsqueeze(0)
        env = SimpleNamespace(
            num_envs=1, extras={"log": {}},
            scene={"robot": SimpleNamespace(data=SimpleNamespace(site_pos_w=positions))},
        )
        self.assertEqual(feet_too_close(env, self.asset, 0.18).item(), 0.0)
        for side, angle in (("L", -0.2), ("R", 0.2)):
            data.qpos[model.joint(f"proximal_roll_{side}_Joint").qposadr[0]] = angle
        mujoco.mj_forward(model, data)
        positions[:] = torch.from_numpy(data.site_xpos[ids].copy())
        self.assertGreater(feet_too_close(env, self.asset, 0.18).item(), 0.0)
