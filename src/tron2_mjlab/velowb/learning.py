"""History/CTS extensions of rsl_rl, without duplicating its PPO implementation."""

import copy
from dataclasses import asdict

import torch
from torch import nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, MLP
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .env import ServoConfig


class ServoActor(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, *, encoder_mode="none",
                 latent_dim=32, encoder_hidden=(128, 64), detach_student_ppo=False, **kwargs):
        self.encoder_mode = encoder_mode
        self.detach_student_ppo = detach_student_ppo
        self.context_dim = 0 if encoder_mode == "none" else latent_dim
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.student_encoder = (
            nn.Sequential(MLP(obs["history"].shape[-1], latent_dim, encoder_hidden, "elu"), nn.Tanh())
            if self.context_dim else nn.Identity())
        self.teacher_encoder = (
            nn.Sequential(MLP(obs["privileged"].shape[-1], latent_dim, encoder_hidden, "elu"), nn.Tanh())
            if encoder_mode in ("cts", "two_stage") else nn.Identity())
        self.teacher_normalizer = (
            EmpiricalNormalization(obs["privileged"].shape[-1])
            if self.obs_normalization and encoder_mode in ("cts", "two_stage") else nn.Identity())
        if self.context_dim and self.obs_normalization and obs["history"].shape[-1] % self.obs_dim:
            raise ValueError("Normalized history must contain complete actor observation frames")

    def student_context(self, history):
        if self.obs_normalization:
            history = self.obs_normalizer(history.reshape(-1, self.obs_dim)).reshape_as(history)
        return self.student_encoder(history)

    def update_normalization(self, obs):
        super().update_normalization(obs)
        if self.obs_normalization and self.encoder_mode in ("cts", "two_stage"):
            self.teacher_normalizer.update(obs["privileged"])

    def _get_latent_dim(self):
        return self.obs_dim + self.context_dim

    def get_latent(self, obs, masks=None, hidden_state=None):
        current = super().get_latent(obs, masks, hidden_state)
        if not self.context_dim:
            return current
        context = self.student_context(obs["history"])
        if self.training and self.encoder_mode in ("cts", "two_stage"):
            teacher = self.teacher_encoder(self.teacher_normalizer(obs["privileged"]))
            if self.detach_student_ppo:
                context = context.detach()
            context = torch.where(obs["role"] > 0.5, teacher, context)
        return torch.cat((current, context), dim=-1)

    def alignment_loss(self, obs, *, distill_actions=False):
        student = self.student_context(obs["history"])
        with torch.no_grad():
            teacher = self.teacher_encoder(self.teacher_normalizer(obs["privileged"]))
        loss = torch.nn.functional.mse_loss(student, teacher)
        if distill_actions:
            current = super().get_latent(obs)
            student_action = self.mlp(torch.cat((current, student), dim=-1))
            with torch.no_grad():
                teacher_action = self.mlp(torch.cat((current, teacher), dim=-1))
            loss = loss + torch.nn.functional.mse_loss(student_action, teacher_action)
        return loss

    def as_jit(self):
        return StudentPolicy(self)


class StudentPolicy(nn.Module):
    """Export boundary: two deployable tensors, no teacher, role or privileged state."""

    __constants__ = ["has_history", "normalize_history", "input_dim"]

    def __init__(self, actor: ServoActor):
        super().__init__()
        self.has_history = bool(actor.context_dim)
        self.normalize_history = actor.obs_normalization
        self.input_dim = actor.obs_dim
        self.normalizer = copy.deepcopy(actor.obs_normalizer)
        self.student_encoder = copy.deepcopy(actor.student_encoder)
        self.policy = copy.deepcopy(actor.mlp)

    def forward(self, observation: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        features = self.normalizer(observation)
        if self.has_history:
            if self.normalize_history:
                history = self.normalizer(history.reshape(-1, self.input_dim)).reshape_as(history)
            features = torch.cat((features, self.student_encoder(history)), dim=-1)
        return self.policy(features)


class AdaptationPPO(PPO):
    """PPO then student-only auxiliary updates on the same rollout (operator split)."""

    def __init__(self, *args, alignment_coef=1.0, alignment_lr=3e-4, clip_rewards=None,
                 alignment_epochs=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.alignment_coef = alignment_coef
        self.clip_rewards = clip_rewards
        self.alignment_epochs = alignment_epochs
        mode = self.actor.encoder_mode
        self.phase = "teacher" if mode == "two_stage" else "concurrent"
        self.adaptation_optimizer = (
            torch.optim.Adam(self.actor.student_encoder.parameters(), lr=alignment_lr)
            if mode in ("cts", "two_stage") else None)

    def process_env_step(self, obs, rewards, dones, extras):
        if self.clip_rewards is not None:
            if not torch.isfinite(rewards).all():
                raise ValueError("Non-finite rewards must be diagnosed, not hidden by clipping")
            rewards = rewards.clamp(-self.clip_rewards, self.clip_rewards)
        super().process_env_step(obs, rewards, dones, extras)

    def set_distillation_phase(self):
        if self.actor.encoder_mode != "two_stage":
            raise ValueError("Distillation phase is only valid for B3/two_stage")
        self.phase = "distill"
        self.actor.requires_grad_(False)
        self.critic.requires_grad_(False)
        self.actor.student_encoder.requires_grad_(True)
        self.train_mode()

    def train_mode(self):
        super().train_mode()
        if self.phase == "distill":
            self.actor.obs_normalizer.eval()
            self.actor.teacher_normalizer.eval()
            self.critic.obs_normalizer.eval()

    def update(self):
        if self.adaptation_optimizer is None or self.phase == "teacher":
            return super().update()
        observations = self.storage.observations.flatten(0, 1)
        distilling = self.phase == "distill"
        losses = ({"value": 0.0, "surrogate": 0.0, "entropy": 0.0}
                  if distilling else super().update())
        count = observations.batch_size[0]
        total, updates = 0.0, 0
        for _ in range(self.alignment_epochs):
            indices = torch.randperm(count, device=self.device)
            for batch_ids in indices.split(max(1, count // self.num_mini_batches)):
                loss = self.alignment_coef * self.actor.alignment_loss(
                    observations[batch_ids], distill_actions=distilling)
                self.adaptation_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.student_encoder.parameters(), self.max_grad_norm)
                self.adaptation_optimizer.step()
                total += loss.item()
                updates += 1
        self.storage.clear()
        losses["latent_alignment"] = total / updates
        return losses

    def save(self):
        saved = super().save()
        saved["adaptation_phase"] = self.phase
        if self.adaptation_optimizer is not None:
            saved["adaptation_optimizer_state_dict"] = self.adaptation_optimizer.state_dict()
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        result = super().load(loaded_dict, load_cfg, strict)
        self.phase = loaded_dict.get("adaptation_phase", self.phase)
        if self.phase == "distill":
            self.set_distillation_phase()
        if (self.adaptation_optimizer is not None
                and (load_cfg is None or load_cfg.get("optimizer", False))):
            self.adaptation_optimizer.load_state_dict(loaded_dict["adaptation_optimizer_state_dict"])
        return result


def runner_config(settings: ServoConfig, *, iterations: int = 10_000,
                  steps_per_env: int = 24) -> dict:
    if settings.ik_mode == "only":
        raise ValueError("IK-only has no learned parameters; run evaluation instead")
    config = asdict(RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(256, 128), stochastic=True, init_noise_std=0.3,
                            noise_std_type="log", class_name="tron2_mjlab.velowb.learning:ServoActor"),
        critic=RslRlModelCfg(hidden_dims=(256, 128)),
        algorithm=RslRlPpoAlgorithmCfg(class_name="tron2_mjlab.velowb.learning:AdaptationPPO",
                                        entropy_coef=0.005, learning_rate=3e-4),
        num_steps_per_env=steps_per_env, max_iterations=iterations, logger="tensorboard",
        seed=settings.seed, experiment_name=f"velowb_{settings.variant.lower()}_{settings.baseline.lower()}",
    ))
    for model_name in ("actor", "critic"):
        config[model_name] = {key: value for key, value in config[model_name].items()
                              if value is not None}
    config["actor"]["encoder_mode"] = settings.encoder_mode
    config["actor"]["latent_dim"] = 32
    config["actor"]["encoder_hidden"] = (128, 64)
    config["algorithm"]["alignment_coef"] = 1.0
    config["algorithm"]["alignment_lr"] = 3e-4
    if settings.control_profile == "wbc":
        for model_name in ("actor", "critic"):
            config[model_name].update(hidden_dims=(512, 256, 128), obs_normalization=True)
        config["actor"].update(init_noise_std=1.0, latent_dim=64,
                                encoder_hidden=(512, 256), detach_student_ppo=True)
        config["algorithm"].update(learning_rate=1e-3, entropy_coef=0.01,
                                    alignment_lr=1e-3, alignment_epochs=5, clip_rewards=5.0)
    return config