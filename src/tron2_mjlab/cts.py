"""Concurrent teacher/student PPO with student-only history inference."""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

import torch
from torch import nn
from tensordict import TensorDict
from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, MLP

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

HISTORY_LENGTH = 5


def teacher_role(
    env: "ManagerBasedRlEnv", teacher_fraction: float = 0.75, play: bool = False,
) -> torch.Tensor:
    if not 0 < teacher_fraction < 1:
        raise ValueError("CTS teacher_fraction must be in (0, 1)")
    if play:
        return torch.zeros((env.num_envs, 1), device=env.device)
    if env.num_envs < 2:
        raise ValueError("CTS training requires at least two environments")
    count = max(1, min(env.num_envs - 1, int(env.num_envs * teacher_fraction)))
    return (torch.arange(env.num_envs, device=env.device) < count).float().unsqueeze(-1)


@dataclass
class CtsActorCfg(RslRlModelCfg):
    class_name: str = "tron2_mjlab.cts:CtsActor"
    latent_dim: int = 64
    encoder_hidden: tuple[int, ...] = (512, 256)
    history_length: int = HISTORY_LENGTH
    obs_normalization: bool = True


@dataclass
class CtsPpoCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "tron2_mjlab.cts:CtsPPO"
    alignment_coef: float = 1.0
    alignment_lr: float = 1e-3
    alignment_epochs: int = 5


class CtsActor(MLPModel):
    def __init__(
        self, obs: TensorDict, obs_groups, obs_set, output_dim, *,
        latent_dim=64, encoder_hidden=(512, 256), history_length=HISTORY_LENGTH, **kwargs,
    ):
        if latent_dim < 1 or history_length < 2:
            raise ValueError("CTS requires a positive latent dimension and at least two history frames")
        if kwargs.get("state_dependent_std", False):
            raise ValueError("CTS export requires state-independent action noise")
        self.context_dim = latent_dim
        self.history_length = history_length
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.student_encoder = nn.Sequential(
            MLP(history_length * self.obs_dim, latent_dim, encoder_hidden, "elu"), nn.Tanh(),
        )
        self.teacher_encoder = nn.Sequential(
            MLP(obs["critic"].shape[-1], latent_dim, encoder_hidden, "elu"), nn.Tanh(),
        )
        self.teacher_normalizer = (
            EmpiricalNormalization(obs["critic"].shape[-1])
            if self.obs_normalization else nn.Identity()
        )

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        if tuple(obs_groups[obs_set]) != ("actor",):
            raise ValueError("CTS actor requires exactly the actor history observation group")
        history = obs["actor"]
        if history.ndim != 3 or history.shape[1] != self.history_length:
            raise ValueError(f"CTS expects actor history shaped (batch, {self.history_length}, features)")
        return ["actor"], history.shape[-1]

    def _get_latent_dim(self):
        return self.obs_dim + self.context_dim

    def student_context(self, history: torch.Tensor) -> torch.Tensor:
        normalized = self.obs_normalizer(history.reshape(-1, self.obs_dim)).reshape(history.shape)
        return self.student_encoder(normalized.flatten(start_dim=1))

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None):
        current = self.obs_normalizer(obs["actor"][:, -1])
        student = self.student_context(obs["actor"])
        if self.training:
            teacher = self.teacher_encoder(self.teacher_normalizer(obs["critic"]))
            context = torch.where(obs["cts_role"] > 0.5, teacher, student.detach())
        else:
            context = student
        return torch.cat((current, context), dim=-1)

    def update_normalization(self, obs):
        if self.obs_normalization:
            self.obs_normalizer.update(obs["actor"][:, -1])
            self.teacher_normalizer.update(obs["critic"])

    def alignment_loss(self, obs):
        student = self.student_context(obs["actor"])
        with torch.no_grad():
            teacher = self.teacher_encoder(self.teacher_normalizer(obs["critic"]))
        return nn.functional.mse_loss(student, teacher)

    def as_jit(self):
        return StudentPolicy(self)

    def as_onnx(self, verbose=False):
        return StudentPolicy(self)


class StudentPolicy(nn.Module):
    """Export input is frame-major actor history, oldest to newest."""

    is_recurrent = False

    def __init__(self, actor: CtsActor):
        super().__init__()
        self.obs_dim = actor.obs_dim
        self.history_length = actor.history_length
        self.normalizer = deepcopy(actor.obs_normalizer)
        self.encoder = deepcopy(actor.student_encoder)
        self.policy = deepcopy(actor.mlp)

    def forward(self, actor_history: torch.Tensor) -> torch.Tensor:
        normalized = self.normalizer(actor_history.reshape(-1, self.obs_dim)).reshape(
            -1, self.history_length, self.obs_dim,
        )
        context = self.encoder(normalized.flatten(start_dim=1))
        return self.policy(torch.cat((normalized[:, -1], context), dim=-1))

    def get_dummy_inputs(self):
        return (torch.zeros(1, self.history_length, self.obs_dim),)

    @property
    def input_names(self):
        return ["actor_history"]

    @property
    def output_names(self):
        return ["actions"]


class CtsPPO(PPO):
    def __init__(
        self, *args, alignment_coef=1.0, alignment_lr=1e-3, alignment_epochs=5, **kwargs,
    ):
        if any(not isfinite(value) or value <= 0 for value in (alignment_coef, alignment_lr)):
            raise ValueError("CTS alignment coefficient and learning rate must be finite and positive")
        if not isinstance(alignment_epochs, int) or alignment_epochs < 1:
            raise ValueError("CTS alignment_epochs must be a positive integer")
        super().__init__(*args, **kwargs)
        if not isinstance(self.actor, CtsActor):
            raise ValueError("CtsPPO requires CtsActor")
        if self.is_multi_gpu or self.symmetry:
            raise ValueError("CTS currently supports single-process training without symmetry augmentation")
        self.alignment_coef = alignment_coef
        self.alignment_epochs = alignment_epochs
        self.student_optimizer = torch.optim.Adam(
            self.actor.student_encoder.parameters(), lr=alignment_lr,
        )

    def update(self):
        # PPO clears the rollout cursor, not its tensors; retain a view for distillation.
        observations = self.storage.observations.flatten(0, 1)
        losses = super().update()
        count = observations.batch_size[0]
        total, updates = 0.0, 0
        for _ in range(self.alignment_epochs):
            indices = torch.randperm(count, device=self.device)
            for batch_ids in indices.split(max(1, count // self.num_mini_batches)):
                loss = self.alignment_coef * self.actor.alignment_loss(observations[batch_ids])
                if not torch.isfinite(loss):
                    raise ValueError("CTS latent alignment loss is not finite")
                self.student_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.student_encoder.parameters(), self.max_grad_norm)
                self.student_optimizer.step()
                total += loss.item()
                updates += 1
        losses["latent_alignment"] = total / updates
        return losses

    def save(self):
        saved = super().save()
        saved["cts_student_optimizer"] = self.student_optimizer.state_dict()
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        result = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("optimizer", False):
            self.student_optimizer.load_state_dict(loaded_dict["cts_student_optimizer"])
        return result
