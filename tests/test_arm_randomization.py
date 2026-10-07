"""Arm dynamics bounds, physical consistency and isolation from leg parameters."""

import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg

from tron2_mjlab.domain_randomization import randomize_arm_dynamics, randomize_arm_pose
from tron2_mjlab.robot import ARM_JOINTS


class ArmRandomizationTests(unittest.TestCase):
    def test_configs_and_invalid_ranges(self):
        events = []
        for name in ("LBC", "Baseline"):
            task = f"Mjlab-Velocity-Flat-TRON2A-SFYG-{name}-CTS"
            event = load_env_cfg(task).events["arm_dynamics"]
            events.append(asdict(event))
            self.assertEqual(event.mode, "startup")
            self.assertEqual(event.params["asset_cfg"].joint_names, ARM_JOINTS)
            self.assertNotIn("arm_dynamics", load_env_cfg(task, play=True).events)
            self.assertNotIn("arm_pose", load_env_cfg(task, play=True).events)
            pose = load_env_cfg(task).events["arm_pose"]
            self.assertEqual(pose.mode, "reset")
            self.assertEqual(pose.params["range_fraction"], 0.8)
            self.assertEqual(pose.params["nominal_probability"], 0.2)
        self.assertEqual(events[0], events[1])
        params = dict(events[0]["params"])
        params["asset_cfg"] = SceneEntityCfg("robot")
        for field, bounds in (
            ("mass_range", (-0.1, 1.0)), ("damping_range", (1.0, 0.5)),
            ("kp_range", (0.8, float("nan"))), ("kd_range", (0, 1)),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                randomize_arm_dynamics(SimpleNamespace(), None, **{**params, field: bounds})

    def test_pose_distribution_and_target_consistency(self):
        robot = Mock()
        robot.data = SimpleNamespace(
            soft_joint_pos_limits=torch.tensor([-2.0, 2.0]).repeat(1024, 16, 1),
            default_joint_pos=torch.zeros((1024, 16)),
        )
        env = SimpleNamespace(num_envs=1024, device="cpu", scene={"robot": robot})
        asset = SceneEntityCfg("robot", joint_ids=list(range(10, 16)))
        torch.manual_seed(42)
        randomize_arm_pose(env, None, asset)
        position, velocity = robot.write_joint_state_to_sim.call_args.args
        self.assertEqual(position.shape, (1024, 6))
        self.assertTrue((position.abs() <= 1.6).all())
        nominal = (position == 0).all(dim=1)
        self.assertGreater(nominal.float().mean().item(), 0.15)
        self.assertLess(nominal.float().mean().item(), 0.25)
        self.assertTrue((position[~nominal].std(dim=0) > 0.7).all())
        torch.testing.assert_close(velocity, torch.zeros_like(velocity))
        torch.testing.assert_close(robot.set_joint_position_target.call_args.args[0], position)
        torch.testing.assert_close(
            robot.set_joint_position_target.call_args.kwargs["joint_ids"], torch.arange(10, 16),
        )
        ids = torch.tensor([2, 7])
        randomize_arm_pose(env, ids, asset, nominal_probability=1.0)
        self.assertEqual(robot.set_joint_position_target.call_args.args[0].shape, (2, 6))
        torch.testing.assert_close(robot.set_joint_position_target.call_args.args[0], torch.zeros((2, 6)))
        torch.testing.assert_close(robot.set_joint_position_target.call_args.kwargs["env_ids"], ids[:, None])
        for kwargs in ({"range_fraction": 0}, {"range_fraction": 1.1}, {"nominal_probability": -1}):
            with self.assertRaises(ValueError):
                randomize_arm_pose(env, ids, asset, **kwargs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for arm randomization")
    def test_randomized_parameters_and_reset_isolation(self):
        cfg = load_env_cfg("Mjlab-Velocity-Flat-TRON2A-SFYG-LBC-CTS")
        cfg.scene.num_envs = 8
        cfg.seed = 42
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
        try:
            env.reset()
            robot = env.scene["robot"]
            motion = env.command_manager.get_term("arm_motion")
            asset = SceneEntityCfg(
                "robot", body_names=tuple(f"arm{index}_Link" for index in range(1, 7)),
                joint_names=ARM_JOINTS, actuator_names=ARM_JOINTS, preserve_order=True,
            )
            asset.resolve(env.scene)
            bodies = robot.indexing.body_ids[asset.body_ids]
            dofs = robot.indexing.joint_v_adr[asset.joint_ids]
            ctrls = robot.indexing.ctrl_ids[asset.actuator_ids]
            arm_ids = asset.joint_ids
            arm_position = robot.data.joint_pos[:, arm_ids].clone()
            torch.testing.assert_close(motion.command, arm_position)
            torch.testing.assert_close(robot.data.joint_pos_target[:, arm_ids], arm_position)
            self.assertGreater(arm_position.std(dim=0).max().item(), 0.1)
            torch.testing.assert_close(robot.data.joint_vel[:, arm_ids], torch.zeros_like(arm_position))
            env.reset(env_ids=torch.tensor([0, 3], device=env.device))
            torch.testing.assert_close(
                robot.data.joint_pos_target[:, arm_ids], robot.data.joint_pos[:, arm_ids],
            )
            remaining = torch.tensor([1, 2, 4, 5, 6, 7], device=env.device)
            torch.testing.assert_close(robot.data.joint_pos[remaining][:, arm_ids], arm_position[remaining])
            torch.testing.assert_close(
                motion.command, robot.data.joint_pos_target[:, arm_ids],
            )
            model = env.sim.model
            default_mass = env.sim.get_default_field("body_mass")
            default_inertia = env.sim.get_default_field("body_inertia")
            mass_scale = model.body_mass[:, bodies] / default_mass[bodies]
            self.assertTrue(((mass_scale >= 0.8) & (mass_scale <= 1.2)).all())
            self.assertGreater(mass_scale.std().item(), 0.02)
            torch.testing.assert_close(
                model.body_inertia[:, bodies], default_inertia[bodies] * mass_scale[..., None],
            )
            checks = (
                ("dof_damping", dofs, None, 0.5, 1.5),
                ("actuator_gainprm", ctrls, 0, 0.8, 1.2),
                ("actuator_biasprm", ctrls, 2, 0.8, 1.2),
            )
            for field, ids, axis, low, high in checks:
                current = getattr(model, field)
                default = env.sim.get_default_field(field)
                ratio = current[:, ids] / default[ids] if axis is None else current[:, ids, axis] / default[ids, axis]
                self.assertTrue(((ratio >= low) & (ratio <= high)).all(), field)
                self.assertGreater(ratio.std().item(), 0.01)
            torch.testing.assert_close(
                model.actuator_gainprm[:, ctrls, 0], -model.actuator_biasprm[:, ctrls, 1],
            )
            for field, ids in (
                ("body_mass", bodies), ("body_inertia", bodies), ("dof_damping", dofs),
                ("actuator_gainprm", ctrls), ("actuator_biasprm", ctrls),
            ):
                current = getattr(model, field)
                mask = torch.ones(current.shape[1], dtype=torch.bool, device=env.device)
                mask[ids] = False
                expected = env.sim.get_default_field(field)[mask]
                torch.testing.assert_close(current[:, mask], expected.expand_as(current[:, mask]))
            subtree = model.body_mass.clone()
            for body in range(env.sim.mj_model.nbody - 1, 0, -1):
                subtree[:, env.sim.mj_model.body_parentid[body]] += subtree[:, body]
            torch.testing.assert_close(model.body_subtreemass[:], subtree)
            for field in ("body_invweight0", "dof_invweight0", "actuator_acc0"):
                values = getattr(model, field)[:]
                self.assertEqual(values.shape[0], env.num_envs)
                self.assertTrue(torch.isfinite(values).all())
            self.assertEqual(env.sim.wp_model.stat.meaninertia.shape[0], env.num_envs)
            original_mass = model.body_mass.clone()
            original_gain = model.actuator_gainprm.clone()
            env.reset(env_ids=torch.tensor([0, 3], device=env.device))
            torch.testing.assert_close(model.body_mass[:], original_mass)
            torch.testing.assert_close(model.actuator_gainprm[:], original_gain)
            params = dict(cfg.events["arm_dynamics"].params)
            params["asset_cfg"] = asset
            ids = torch.tensor([0, 3], device=env.device)
            for _ in range(2):
                torch.manual_seed(123)
                randomize_arm_dynamics(env, ids, **params)
                if _ == 0:
                    repeated_mass = model.body_mass.clone()
                    repeated_gain = model.actuator_gainprm.clone()
                else:
                    torch.testing.assert_close(model.body_mass[:], repeated_mass)
                    torch.testing.assert_close(model.actuator_gainprm[:], repeated_gain)
            unchanged = torch.tensor([1, 2, 4, 5, 6, 7], device=env.device)
            torch.testing.assert_close(model.body_mass[unchanged], original_mass[unchanged])
            torch.testing.assert_close(model.actuator_gainprm[unchanged], original_gain[unchanged])
            moved_target = False
            for _ in range(60):
                previous_target = motion.command.clone()
                obs, reward, _, _, _ = env.step(torch.zeros((8, 10), device=env.device))
                self.assertTrue(torch.isfinite(reward).all())
                self.assertTrue(all(torch.isfinite(value).all() for value in obs.values()))
                continuing = env.episode_length_buf > 0
                if continuing.any():
                    difference = (motion.command[continuing] - previous_target[continuing]).abs()
                    self.assertLessEqual(difference.max().item(), 0.02001)
                    moved_target |= difference.max().item() > 1e-4
                torch.testing.assert_close(robot.data.joint_pos_target[:, arm_ids], motion.command)
            self.assertTrue(moved_target)
        finally:
            env.close()
