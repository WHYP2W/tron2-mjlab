"""CTS routing, gradient isolation, history, checkpoint and export contracts."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from tron2_mjlab.cts import CtsActor, CtsActorCfg, CtsPpoCfg, HISTORY_LENGTH, teacher_role
from tron2_mjlab.play import ManualVelocityCommandCfg, VelocityViewer
from tron2_mjlab.robot import ARM_JOINTS, LEG_JOINTS


def observations(width=44):
    return TensorDict({
        "actor": torch.randn(4, HISTORY_LENGTH, width),
        "critic": torch.randn(4, width + 12),
        "cts_role": torch.tensor([[1.0], [1.0], [1.0], [0.0]]),
    }, batch_size=[4])


def actor_for(obs):
    return CtsActor(
        obs, {"actor": ("actor",)}, "actor", 10, latent_dim=8,
        encoder_hidden=(32, 16), hidden_dims=(32, 16),
        stochastic=True, obs_normalization=True,
    )


class CtsTests(unittest.TestCase):
    def test_configs_and_role_split(self):
        for baseline in (False, True):
            name = "Baseline" if baseline else "LBC"
            task = f"Mjlab-Velocity-Flat-TRON2A-SFYG-{name}-CTS"
            rl = load_rl_cfg(task)
            self.assertIsInstance(rl.actor, CtsActorCfg)
            self.assertIsInstance(rl.algorithm, CtsPpoCfg)
            self.assertTrue(rl.experiment_name.endswith(f"_{name.lower()}_cts"))
            for play in (False, True):
                cfg = load_env_cfg(task, play=play)
                self.assertEqual(cfg.observations["actor"].history_length, 5)
                self.assertFalse(cfg.observations["actor"].flatten_history_dim)
                self.assertIsNone(cfg.observations["critic"].history_length)
                joints = LEG_JOINTS if baseline else LEG_JOINTS + ARM_JOINTS
                for group in ("actor", "critic"):
                    for term in ("joint_pos", "joint_vel"):
                        self.assertEqual(
                            cfg.observations[group].terms[term].params["asset_cfg"].joint_names, joints,
                        )
                        if group == "critic":
                            self.assertEqual(cfg.observations[group].terms[term].history_length, 0)
        env = SimpleNamespace(num_envs=4, device="cpu")
        torch.testing.assert_close(teacher_role(env), torch.tensor([[1.0], [1.0], [1.0], [0.0]]))
        torch.testing.assert_close(teacher_role(env, play=True), torch.zeros((4, 1)))
        env.num_envs = 1
        with self.assertRaises(ValueError):
            teacher_role(env)
        torch.testing.assert_close(teacher_role(env, play=True), torch.zeros((1, 1)))

    def test_training_gradient_isolation(self):
        obs = observations()
        actor = actor_for(obs)
        actor.train()
        actor(obs).square().sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in actor.teacher_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in actor.student_encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in actor.mlp.parameters()))
        actor.zero_grad(set_to_none=True)
        actor.alignment_loss(obs).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in actor.student_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in actor.teacher_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in actor.mlp.parameters()))

    def test_student_ignores_privileged_state_and_role(self):
        for width in (44, 56):
            obs = observations(width)
            actor = actor_for(obs)
            actor.eval()
            expected = actor(obs)
            obs["critic"] = torch.randn_like(obs["critic"]) * 1000
            obs["cts_role"] = 1 - obs["cts_role"]
            torch.testing.assert_close(actor(obs), expected)
            deployable = TensorDict({"actor": obs["actor"]}, batch_size=[4])
            torch.testing.assert_close(actor(deployable), expected)
            scripted = torch.jit.script(actor.as_jit().eval())
            torch.testing.assert_close(scripted(obs["actor"]), expected)
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / "student.pt")
                scripted.save(path)
                torch.testing.assert_close(torch.jit.load(path)(obs["actor"]), expected)

    def test_training_routes_by_saved_environment_role(self):
        obs = observations()
        actor = actor_for(obs)
        actor.train()
        before = actor(obs)
        obs["critic"] += 100
        after = actor(obs)
        torch.testing.assert_close(after[3], before[3])
        self.assertGreater((after[:3] - before[:3]).abs().max().item(), 1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for CTS integration")
    def test_history_training_resume_and_student_export(self):
        for baseline in (False, True):
            with self.subTest(baseline=baseline), tempfile.TemporaryDirectory() as directory:
                name = "Baseline" if baseline else "LBC"
                task = f"Mjlab-Velocity-Flat-TRON2A-SFYG-{name}-CTS"
                width = 44 if baseline else 56
                cfg = load_env_cfg(task)
                cfg.scene.num_envs = 4
                cfg.observations["actor"].enable_corruption = False
                cfg.commands["twist"] = ManualVelocityCommandCfg(**vars(cfg.commands["twist"]))
                env = ManagerBasedRlEnv(cfg, device="cuda:0")
                try:
                    wrapped = RslRlVecEnvWrapper(env)
                    obs = wrapped.get_observations()
                    self.assertEqual(obs["actor"].shape, (4, 5, width))
                    self.assertEqual(obs["critic"].shape, (4, width + 12))
                    torch.testing.assert_close(obs["actor"], obs["actor"][:, -1:].expand(-1, 5, -1))
                    for _ in range(3):
                        previous = obs["actor"].clone()
                        obs, _, dones, _ = wrapped.step(torch.zeros((4, 10), device=env.device))
                        continuing = ~dones.bool()
                        torch.testing.assert_close(
                            obs["actor"][continuing, :-1], previous[continuing, 1:],
                        )
                    env.reset(env_ids=torch.tensor([1], device=env.device))
                    obs = wrapped.get_observations()
                    torch.testing.assert_close(obs["actor"][1], obs["actor"][1, -1:].expand(5, -1))
                    viewer = VelocityViewer(wrapped, lambda obs: torch.zeros((4, 10), device=env.device))
                    viewer.setup()
                    try:
                        previous = wrapped.get_observations()["actor"].clone()
                        viewer.sliders[0].value = 0.8
                        viewer.sync_viewer_to_env()
                        refreshed = wrapped.get_observations()["actor"]
                        torch.testing.assert_close(refreshed[:, :-1], previous[:, :-1])
                        torch.testing.assert_close(refreshed[:, -1, -3], torch.full((4,), 0.8, device=env.device))
                    finally:
                        viewer.close()
                    rl = load_rl_cfg(task)
                    rl.actor.hidden_dims = (32, 16)
                    rl.actor.encoder_hidden = (32, 16)
                    rl.critic.hidden_dims = (32, 16)
                    rl.num_steps_per_env = 4
                    rl.algorithm.num_learning_epochs = 1
                    rl.algorithm.num_mini_batches = 1
                    rl.algorithm.alignment_epochs = 1
                    runner = load_runner_cls(task)(wrapped, asdict(rl), directory, env.device)
                    actor = runner.alg.actor
                    before_teacher = [p.detach().clone() for p in actor.teacher_encoder.parameters()]
                    before_student = [p.detach().clone() for p in actor.student_encoder.parameters()]
                    original_update = runner.alg.update
                    recorded = []

                    def record_update():
                        losses = original_update()
                        recorded.append(losses)
                        return losses

                    with patch.object(runner.alg, "update", side_effect=record_update):
                        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
                    self.assertTrue(np.isfinite(recorded[0]["latent_alignment"]))
                    for before, parameters in (
                        (before_teacher, actor.teacher_encoder.parameters()),
                        (before_student, actor.student_encoder.parameters()),
                    ):
                        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, parameters)))
                    checkpoint = str(Path(directory) / "model_resume.pt")
                    runner.save(checkpoint)
                    saved = torch.load(checkpoint, weights_only=False)
                    self.assertTrue(saved["cts_student_optimizer"]["state"])
                    expected_state = {key: value.clone() for key, value in actor.state_dict().items()}
                    resumed = load_runner_cls(task)(wrapped, asdict(rl), device=env.device)
                    resumed.load(checkpoint)
                    for key, value in resumed.alg.actor.state_dict().items():
                        torch.testing.assert_close(value, expected_state[key])
                    for key, state in saved["cts_student_optimizer"]["state"].items():
                        for field, value in state.items():
                            torch.testing.assert_close(
                                resumed.alg.student_optimizer.state_dict()["state"][key][field], value,
                            )
                    obs = wrapped.get_observations()
                    policy = resumed.get_inference_policy()
                    deployable = TensorDict({"actor": obs["actor"]}, batch_size=[4])
                    with torch.no_grad():
                        expected = policy(deployable)
                    self.assertEqual(expected.shape, (4, 10))
                    exported_path = Path(directory) / "student.onnx"
                    resumed.export_policy_to_onnx(directory, "student.onnx")
                    model = onnx.load(exported_path)
                    onnx.checker.check_model(model)
                    self.assertEqual([item.name for item in model.graph.input], ["actor_history"])
                    result = ReferenceEvaluator(model).run(
                        None, {"actor_history": obs["actor"][:1].cpu().numpy()},
                    )[0]
                    np.testing.assert_allclose(result, expected[:1].cpu().numpy(), rtol=1e-4, atol=1e-5)
                finally:
                    env.close()
