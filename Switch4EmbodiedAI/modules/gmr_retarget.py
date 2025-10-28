import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Ensure GMR submodule importable
REPO_ROOT = Path(__file__).resolve().parents[2]
GMR_ROOT = REPO_ROOT / "third_party" / "GMR"
if GMR_ROOT.exists():
    sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR # type: ignore
from general_motion_retargeting import KinematicsModel, RobotMotionViewer # type: ignore


@dataclass
class GMRConfig:
    robot: str = "unitree_g1"
    # Visualization flags
    visualize: bool = True
    step_full: bool = False
    record_video: bool = False
    video_path: str | None = None
    # Retargeting behavior flags
    rate_limit: bool = False
    joint_vel_limit: bool = True
    collision_avoid: bool = False
    offset_ground: bool = True


class GMRRetarget:
    def __init__(self, cfg: GMRConfig, motion_fps: int = 30):
        self.cfg = cfg
        self.motion_fps = motion_fps
        self.retarget = GMR(
            actual_human_height=None,
            src_human="smplx",
            tgt_robot=self.cfg.robot,
            use_velocity_limit=self.cfg.joint_vel_limit,
            use_collision_avoidance=self.cfg.collision_avoid,
        )
        self.viewer = None
        if self.cfg.visualize:
            self.viewer = RobotMotionViewer(
                robot_type=self.cfg.robot,
                motion_fps=self.motion_fps,
                transparent_robot=0,
                record_video=self.cfg.record_video,
                video_path=self.cfg.video_path,
            )

    def compute_local_body_pos(self, xml_file, dof_pos):
        """Compute local body positions with root at origin and identity rotation.

        Args:
            xml_file: Path to the robot MJCF file used by the retargeter.
            dof_pos: Numpy array of shape (T, dof_dim) with per-frame joint positions.

        Returns:
            local_body_pos: Numpy array (T, num_bodies, 3) of local body positions.
            body_names: List of body names corresponding to the second dimension.
        """
        device = torch.device("cpu")
        kinematics_model = KinematicsModel(xml_file, device=device)
        num_frames = dof_pos.shape[0]
        fk_root_pos = torch.zeros((num_frames, 3), device=device)
        fk_root_rot = torch.zeros((num_frames, 4), device=device)
        fk_root_rot[:, -1] = 1.0
        local_body_pos_t, _ = kinematics_model.forward_kinematics(
            fk_root_pos,
            fk_root_rot,
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
        )
        local_body_pos = local_body_pos_t.detach().cpu().numpy()
        body_names = kinematics_model.body_names
        return local_body_pos, body_names

    def step(self, smplx_data_frame: dict) -> dict:
        qpos = self.retarget.retarget(smplx_data_frame, self.cfg.offset_ground)
        # Build per-step motion_data like gvhmr_to_robot.py
        root_pos = qpos[:3]
        # convert wxyz -> xyzw for saving compatibility
        root_rot = qpos[3:7][[1, 2, 3, 0]]
        dof_pos = qpos[7:]
        motion_data = {
            "fps": self.motion_fps,
            "qpos": np.asarray(qpos)[None, ...],
            "root_pos": np.asarray(root_pos)[None, ...],
            "root_rot": np.asarray(root_rot)[None, ...],
            "dof_pos": np.asarray(dof_pos)[None, ...],
        }
        return motion_data
    
    def step_full(self, smplx_data_frame: dict) -> dict:
        qpos = self.retarget.retarget(smplx_data_frame, self.cfg.offset_ground)
        # Build per-step motion_data like gvhmr_to_robot.py
        root_pos = qpos[:3]
        # convert wxyz -> xyzw for saving compatibility
        root_rot = qpos[3:7][[1, 2, 3, 0]]
        dof_pos = qpos[7:]
        local_body_pos, body_names = self.compute_local_body_pos(self.retarget.xml_file, dof_pos)

        motion_data = {
            "fps": self.motion_fps,
            "qpos": np.asarray(qpos)[None, ...],
            "root_pos": np.asarray(root_pos)[None, ...],
            "root_rot": np.asarray(root_rot)[None, ...],
            "dof_pos": np.asarray(dof_pos)[None, ...],
            "local_body_pos": np.asarray(local_body_pos)[None, ...],
        }
        return motion_data

    def vis_step(self, smplx_data_frame: dict) -> np.ndarray:
        qpos = self.retarget.retarget(smplx_data_frame, self.cfg.offset_ground)
        if self.viewer is not None:
            self.viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=self.retarget.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=self.cfg.rate_limit,
            )

            motion_data = {
                "fps": self.motion_fps,
                "qpos": np.asarray(qpos)[None, ...],
                "root_pos": np.asarray(qpos[:3])[None, ...],
                "root_rot": np.asarray(qpos[3:7][[1, 2, 3, 0]])[None, ...], # xyzw
                "dof_pos": np.asarray(qpos[7:])[None, ...],
            }
            return motion_data
        return qpos

    def get_default_pose(self) -> dict:
        """
        Get default robot pose with zero joint positions.
        
        Returns:
            motion_data dict with default pose
        """
        # Default standing pose with zero joint positions
        root_pos = np.zeros(3)
        root_rot = np.array([0.0, 0.0, 0.0, 1.0])  # identity quaternion (xyzw)
        dof_pos = np.zeros(23)  # G1 has 23 DOF
        
        motion_data = {
            "fps": self.motion_fps,
            "root_pos": root_pos[None, ...],
            "root_rot": root_rot[None, ...],
            "dof_pos": dof_pos[None, ...],
        }
        return {"motion_data": motion_data}

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


