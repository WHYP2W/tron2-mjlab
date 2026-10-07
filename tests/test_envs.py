"""Run with: uv run python -m unittest discover -s tests -v."""

import tempfile
import unittest
import re
from dataclasses import asdict

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_rl_cfg, load_runner_cls

from tron2_mjlab.env_cfgs import tron2_flat_env_cfg
from tron2_mjlab.play import ManualVelocityCommandCfg, VelocityViewer
from tron2_mjlab.robot import VARIANTS, get_robot_cfg


class Tron2Tests(unittest.TestCase):
    def test_arm_articulation(self):
        for variant in VARIANTS:
            for locked, arm_dofs in ((True, 5), (False, 6)):
                with self.subTest(variant=variant, locked=locked):
                    cfg = get_robot_cfg(
                        variant, active_arm=True, lock_arm_yaw=locked,
                    )
                    model = Entity(cfg).spec.compile()
                    self.assertEqual((model.nq, model.nv, model.nu),
                                     (17 + arm_dofs, 16 + arm_dofs, 10 + arm_dofs))
                    self.assertGreaterEqual(model.site("ee").id, 0)
                    self.assertEqual(
                        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                         "arm1_Joint") == -1, locked,
                    )
                    for joint_id in range(model.njnt):
                        name = model.joint(joint_id).name or ""
                        if name.startswith("arm"):
                            values = [value for pattern, value in cfg.init_state.joint_pos.items()
                                      if re.fullmatch(pattern, name)]
                            self.assertEqual(values, [0.0])
                    self.assertEqual(cfg.init_state.joint_vel, {".*": 0.0})

    def test_models_and_configs(self):
        self.assertEqual(
            [task for task in list_tasks() if "TRON2" in task],
            [f"Mjlab-Velocity-Flat-TRON2A-{variant}" for variant in VARIANTS],
        )
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = Entity(get_robot_cfg(variant)).spec.compile()
                self.assertEqual((model.nq, model.nv, model.nu), (17, 16, 10))
                self.assertFalse((model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).any())
                cfg = tron2_flat_env_cfg(variant)
                self.assertEqual(cfg.scene.terrain.terrain_type, "plane")
                self.assertIsNone(cfg.scene.terrain.terrain_generator)
                self.assertFalse(cfg.curriculum)
                for group in cfg.observations.values():
                    self.assertNotIn("height_scan", group.terms)
                play = tron2_flat_env_cfg(variant, play=True)
                self.assertFalse(play.observations["actor"].enable_corruption)
                self.assertTrue(cfg.observations["actor"].enable_corruption)
                if variant == "WFYG":
                    self.assertIn("wheel_vel", cfg.actions)
                    self.assertNotIn("foot_slip", cfg.rewards)
        with self.assertRaises(ValueError):
            tron2_flat_env_cfg("SFYG_TRON2B")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU smoke test")
    def test_manual_command(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = tron2_flat_env_cfg(variant, play=True)
                cfg.commands["twist"] = ManualVelocityCommandCfg(**vars(cfg.commands["twist"]))
                env = ManagerBasedRlEnv(cfg, device="cuda:0")
                try:
                    env.reset()
                    command = env.command_manager.get_term("twist")
                    target = (0.5, 0.2 if variant == "SFYG" else 0.0, -0.3)
                    command.manual_target = target
                    command.time_left[:] = 0.0
                    env.step(torch.zeros((1, 10), device=env.device))
                    torch.testing.assert_close(command.command[0], torch.tensor(target, device=env.device))
                    env.reset()
                    torch.testing.assert_close(command.command[0], torch.tensor(target, device=env.device))
                    command.manual_target = (0.0, 0.0, 0.0)
                    command._update_command()
                    self.assertFalse(command.command.any())
                    command.manual_target = None
                    command._resample_command(torch.arange(env.num_envs, device=env.device))
                    self.assertTrue(torch.isfinite(command.command).all())
                    wrapped = RslRlVecEnvWrapper(env)
                    viewer = VelocityViewer(wrapped, lambda obs: torch.zeros((1, 10), device=env.device))
                    viewer.setup()
                    try:
                        viewer.sliders[0].value = 0.5
                        viewer.sliders[1].value = 0.2
                        viewer.sliders[2].value = -0.3
                        viewer.sync_viewer_to_env()
                        torch.testing.assert_close(wrapped.get_observations()["actor"][0, -3:],
                                                   torch.tensor(target, device=env.device))
                        self.assertEqual(viewer.sliders[1].disabled, variant == "WFYG")
                        viewer.manual.value = False
                        viewer.sync_viewer_to_env()
                        self.assertIsNone(command.manual_target)
                        viewer._stop(None)
                        viewer.sync_viewer_to_env()
                        self.assertFalse(command.command.any())
                    finally:
                        viewer.close()
                finally:
                    env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU smoke test")
    def test_step_and_ppo(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as log_dir:
                cfg = tron2_flat_env_cfg(variant)
                cfg.scene.num_envs = 4
                cfg.seed = 42
                env = ManagerBasedRlEnv(cfg, device="cuda:0")
                try:
                    env.reset()
                    actions = torch.zeros((4, 10), device=env.device)
                    touched_ground = False
                    for step in range(100):
                        obs, reward, _, _, _ = env.step(actions)
                        self.assertTrue(torch.isfinite(reward).all())
                        self.assertTrue(all(torch.isfinite(value).all() for value in obs.values()))
                        touched_ground |= bool(env.scene["feet_ground_contact"].data.found.any())
                    self.assertTrue(touched_ground)
                    task = f"Mjlab-Velocity-Flat-TRON2A-{variant}"
                    runner_cfg = load_rl_cfg(task)
                    runner_cfg.num_steps_per_env = 8
                    runner_cfg.algorithm.num_learning_epochs = 1
                    runner_cfg.algorithm.num_mini_batches = 1
                    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
                    runner = load_runner_cls(task)(wrapped, asdict(runner_cfg), log_dir, env.device)
                    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
                    self.assertTrue(all(torch.isfinite(param).all() for param in runner.alg.get_policy().parameters()))
                finally:
                    env.close()