import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

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
    # Base pitch constraint
    base_pitch_limit: bool = True
    base_pitch_max_rad: float = 0.10


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

        if self.cfg.base_pitch_limit:
            from Switch4EmbodiedAI.utils.base_pitch_limit import BasePitchConfigurationLimit
            base_limit = BasePitchConfigurationLimit(
                self.retarget.model,
                joint_name=None,
                max_pitch_rad=self.cfg.base_pitch_max_rad,
                axis="y",
                gain=1.0,
            )
            # Append to the IK limits used by mink.solve_ik
            if hasattr(self.retarget, 'ik_limits') and isinstance(self.retarget.ik_limits, list):
                self.retarget.ik_limits.append(base_limit)
            print(f"[GMRRetarget] Base pitch limit applied with max pitch {self.cfg.base_pitch_max_rad} rad")

        # Debug flag: print base pitch a few times to verify constraint effect
        self._debug_pitch_prints_remaining = 20

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

        # #######################################################################
        # ### Debug: print base pitch a few times to verify constraint effect ###
        # if self._debug_pitch_prints_remaining > 0:
        #     wxyz = qpos[3:7]
        #     xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        #     euler_xyz = R.from_quat(xyzw).as_euler('xyz', degrees=False)
        #     pitch = float(euler_xyz[1])
        #     print(f"[BasePitch] pitch={pitch:.3f} rad (max={getattr(self.cfg,'base_pitch_max_rad',np.nan):.3f})")
        #     self._debug_pitch_prints_remaining -= 1
        # ### End debug ###
        # #################


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
        return None

    def get_default_pose(self) -> dict:
        """
        Get default robot pose with zero joint positions.
        
        Returns:
            motion_data dict with default pose
        """
        # Get DOF count from the robot model (nv is total DOF including root)
        # qpos format: [root_pos(3), root_rot(4), dof_pos(n)]
        # So dof_pos size = nv - 7 (3 for root_pos + 4 for root_rot)
        num_dof = self.retarget.model.nv - 7
        
        # Default standing pose with zero joint positions
        # Default height varies by robot, using a reasonable default
        # For most humanoids, 0.75-0.85m is typical
        default_heights = {
            "unitree_g1": 0.793,
            "unitree_g1_with_hands": 0.793,
            "unitree_h1": 0.85,
            "unitree_h1_2": 0.85,
            "booster_t1": 0.80,
            "booster_t1_29dof": 0.80,
            "booster_k1": 0.75,
            "stanford_toddy": 0.50,  # toddler robot
            "fourier_n1": 0.80,
            "engineai_pm01": 0.80,
            "kuavo_s45": 0.85,
            "hightorque_hi": 0.80,
            "galaxea_r1pro": 0.80,
            "berkeley_humanoid_lite": 0.75,
            "pnd_adam_lite": 0.80,
            "pnd_adam_inspire": 0.80,
            "tienkung": 0.80,
        }
        default_height = default_heights.get(self.cfg.robot, 0.80)
        
        root_pos = np.array([0.0, 0.0, default_height])
        root_rot = np.array([0.0, 0.0, 0.0, 1.0])  # identity quaternion (xyzw)
        dof_pos = np.zeros(num_dof)
        
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


