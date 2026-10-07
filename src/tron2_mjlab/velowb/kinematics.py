"""Batched nominal-model FK; runtime inputs are encoder and odometry data only."""

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply


def rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    axes = torch.eye(3, device=quaternion.device, dtype=quaternion.dtype)
    axes = axes.expand(*quaternion.shape[:-1], 3, 3)
    return quat_apply(quaternion.unsqueeze(-2).expand(*axes.shape[:-1], 4),
                      axes).transpose(-1, -2)


def axis_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    basis = torch.eye(3, dtype=axis.dtype, device=axis.device)
    skew = torch.linalg.cross(axis.expand(3, 3), basis, dim=-1).T
    angle = angle[..., None, None]
    return basis + angle.sin() * skew + (1 - angle.cos()) * (skew @ skew)


@dataclass
class KinematicState:
    position: torch.Tensor
    rotation: torch.Tensor
    jacobian: torch.Tensor
    arm_com: torch.Tensor


@dataclass
class Estimate:
    position: torch.Tensor
    rotation: torch.Tensor
    twist: torch.Tensor
    base_induced_twist: torch.Tensor
    jacobian: torch.Tensor
    arm_com: torch.Tensor


class ArmKinematics:
    """Extract a fixed nominal tree once; never read simulator state during FK."""

    def __init__(self, model: mujoco.MjModel, *, prefix: str = "",
                 device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.device, self.dtype = device, dtype
        self.base_id = model.body(prefix + "base_Link").id
        arm_root = model.body(prefix + "arm_base_Link").id
        site_id = model.site(prefix + "ee").id
        self.ee_body = int(model.site_bodyid[site_id])
        ancestors = set()
        body_id = self.ee_body
        while body_id != self.base_id:
            if body_id == 0:
                raise ValueError("EE must descend from base_Link")
            ancestors.add(body_id)
            body_id = int(model.body_parentid[body_id])
        arm_bodies = {arm_root}
        for body_id in range(arm_root + 1, model.nbody):
            if int(model.body_parentid[body_id]) in arm_bodies:
                arm_bodies.add(body_id)
        self.arm_body_ids = sorted(arm_bodies)
        body_ids = sorted(ancestors | arm_bodies)
        self.joint_ids = [int(model.body_jntadr[body_id]) for body_id in body_ids
                          if model.body_jntnum[body_id]]
        if not self.joint_ids:
            raise ValueError("The model has no active arm joints")
        self.joint_names = [model.joint(joint_id).name for joint_id in self.joint_ids]
        self.qpos_ids = model.jnt_qposadr[self.joint_ids].copy()
        self.dof_ids = model.jnt_dofadr[self.joint_ids].copy()
        self.limits = self._tensor(model.jnt_range[self.joint_ids])
        self.nodes = []
        for body_id in body_ids:
            joint_index = None
            axis, anchor, reference = None, None, 0.0
            if model.body_jntnum[body_id]:
                joint_id = int(model.body_jntadr[body_id])
                if (model.body_jntnum[body_id] != 1
                        or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE):
                    raise ValueError("Arm FK requires one revolute joint per body")
                joint_index = self.joint_ids.index(joint_id)
                axis = self._tensor(model.jnt_axis[joint_id])
                anchor = self._tensor(model.jnt_pos[joint_id])
                reference = float(model.qpos0[model.jnt_qposadr[joint_id]])
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.body_quat[body_id])
            self.nodes.append((body_id, int(model.body_parentid[body_id]),
                               self._tensor(model.body_pos[body_id]),
                               self._tensor(rotation.reshape(3, 3)), joint_index,
                               axis, anchor, reference))
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.site_quat[site_id])
        self.site_rotation = self._tensor(rotation.reshape(3, 3))
        self.site_position = self._tensor(model.site_pos[site_id])
        self.com_offsets = self._tensor(model.body_ipos[self.arm_body_ids])
        self.masses = self._tensor(model.body_mass[self.arm_body_ids])

    def _tensor(self, value) -> torch.Tensor:
        return torch.tensor(value, device=self.device, dtype=self.dtype)

    def forward(self, joint_position: torch.Tensor) -> KinematicState:
        if joint_position.ndim != 2 or joint_position.shape[1] != len(self.joint_ids):
            raise ValueError("Expected joint_position with shape (batch, arm_dofs)")
        batch = joint_position.shape[0]
        identity = torch.eye(3, device=self.device, dtype=self.dtype).expand(batch, 3, 3)
        zero = torch.zeros((batch, 3), device=self.device, dtype=self.dtype)
        transforms = {self.base_id: (zero, identity)}
        joint_axes, joint_anchors = [], []
        for body_id, parent_id, offset, local_rotation, index, axis, anchor, ref in self.nodes:
            parent_position, parent_rotation = transforms[parent_id]
            position = parent_position + (parent_rotation @ offset)
            rotation = parent_rotation @ local_rotation
            if index is not None:
                joint_axes.append(rotation @ axis)
                pivot = position + rotation @ anchor
                joint_anchors.append(pivot)
                rotation = rotation @ axis_rotation(axis, joint_position[:, index] - ref)
                position = pivot - rotation @ anchor
            transforms[body_id] = (position, rotation)
        position, rotation = transforms[self.ee_body]
        position = position + rotation @ self.site_position
        rotation = rotation @ self.site_rotation
        axes = torch.stack(joint_axes, dim=-1)
        offsets = position.unsqueeze(-1) - torch.stack(joint_anchors, dim=-1)
        jacobian = torch.cat((torch.linalg.cross(axes, offsets, dim=1), axes), dim=1)
        centers = torch.stack([
            transforms[body_id][0] + transforms[body_id][1] @ self.com_offsets[index]
            for index, body_id in enumerate(self.arm_body_ids)
        ], dim=1)
        arm_com = (centers * self.masses[None, :, None]).sum(dim=1) / self.masses.sum()
        return KinematicState(position, rotation, jacobian, arm_com)

    def estimate(self, joint_position: torch.Tensor, joint_velocity: torch.Tensor,
                 base_position: torch.Tensor, base_quaternion: torch.Tensor,
                 base_linear_velocity: torch.Tensor,
                 base_angular_velocity: torch.Tensor) -> Estimate:
        """All base quantities are world-frame; quaternion order is wxyz."""
        state = self.forward(joint_position)
        rotation = rotation_matrix(base_quaternion)
        offset = (rotation @ state.position.unsqueeze(-1)).squeeze(-1)
        jacobian = torch.cat((rotation @ state.jacobian[:, :3],
                              rotation @ state.jacobian[:, 3:]), dim=1)
        induced = torch.cat((base_linear_velocity + torch.linalg.cross(
            base_angular_velocity, offset, dim=-1), base_angular_velocity), dim=-1)
        twist = induced + (jacobian @ joint_velocity.unsqueeze(-1)).squeeze(-1)
        return Estimate(base_position + offset, rotation @ state.rotation, twist,
                        induced, jacobian, base_position +
                        (rotation @ state.arm_com.unsqueeze(-1)).squeeze(-1))