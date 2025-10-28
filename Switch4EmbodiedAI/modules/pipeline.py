import sys

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

from .stream_module import SimpleStreamModule, SimpleStreamModuleConfig
from .gvhmr_realtime import GVHMRRealtime, GVHMRRealtimeConfig
from .gmr_retarget import GMRRetarget, GMRConfig

# Import GMR utils for per-frame SMPLX-to-joint dict conversion
REPO_ROOT = Path(__file__).resolve().parents[2]
GMR_ROOT = REPO_ROOT / "third_party" / "GMR"
if GMR_ROOT.exists():
    sys.path.insert(0, str(GMR_ROOT))
from general_motion_retargeting.utils.smpl import get_smplx_data # type: ignore
import general_motion_retargeting.utils.lafan_vendor.utils as gmr_utils # type: ignore
from scipy.spatial.transform import Rotation as R
import time
import smplx


@dataclass
class PipelineConfig:
    use_stream: bool = True
    # UDP output
    udp_enabled: bool = True
    udp_ip: str = "127.0.0.1"
    udp_send_port: int = 54010
    stream: SimpleStreamModuleConfig = SimpleStreamModuleConfig()
    gvhmr: GVHMRRealtimeConfig = GVHMRRealtimeConfig()
    gmr: GMRConfig = GMRConfig()


class StreamToRobotPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.stream = SimpleStreamModule(cfg.stream) if cfg.use_stream else None
        self.gvhmr = GVHMRRealtime(cfg.gvhmr)
        self.gmr = GMRRetarget(cfg.gmr, motion_fps=30)
        # Prepare SMPLX body model for per-frame conversion (GMR assets path)
        self.cached_betas = None
        smplx_models_path = GMR_ROOT / "assets" / "body_models"
        self.body_model = smplx.create(
            str(smplx_models_path), "smplx", gender="neutral", use_pca=False
        )
        # store previous to compute finite differences (velocities)
        self._prev = {
            "t": None,
            "root_pos": None,
            "root_rot": None,
            "dof_pos": None,
        }

    def start(self):
        if self.stream is not None:
            self.stream.start()

    def _process_frame(self, frame) -> dict | None:
        pred = self.gvhmr.step(frame)
        if pred is None:
            return None

        # Convert last-frame GVHMR global params to the per-frame joint dict expected by GMR
        smpl_params_global = pred["smpl_params_global"]  # tensors on CPU
        body_pose = smpl_params_global["body_pose"].view(1, -1).numpy()
        global_orient = smpl_params_global["global_orient"].view(1, -1).numpy()
        transl = smpl_params_global["transl"].view(1, -1).numpy()
        if self.cached_betas is None:
            # Use first seen betas; pad to 16 as GMR expects
            betas = smpl_params_global.get("betas", torch.zeros(10)).detach().cpu().numpy()
            if betas.ndim == 2:
                betas = betas[0]
            if betas.shape[0] < 16:
                betas = np.pad(betas, (0, 16 - betas.shape[0]))
            self.cached_betas = betas
        # betas = np.pad(smpl_params_global['betas'][0], (0,6))

        smplx_data = {
            "pose_body": body_pose.reshape(-1, 63),
            "betas": self.cached_betas,
            # "betas": betas,
            "root_orient": global_orient.reshape(-1, 3),
            "trans": transl.reshape(-1, 3),
            "mocap_frame_rate": torch.tensor(30),
        }
        smplx_output = self.body_model(
            betas=torch.tensor(self.cached_betas).float().view(1, -1),
            # betas=torch.tensor(betas).float().view(1, -1),
            global_orient=torch.tensor(smplx_data["root_orient"]).float(),
            body_pose=torch.tensor(smplx_data["pose_body"]).float(),
            transl=torch.tensor(smplx_data["trans"]).float(),
            left_hand_pose=torch.zeros(1, 45).float(),
            right_hand_pose=torch.zeros(1, 45).float(),
            jaw_pose=torch.zeros(1, 3).float(),
            leye_pose=torch.zeros(1, 3).float(),
            reye_pose=torch.zeros(1, 3).float(),
            return_full_pose=True,
        )
        per_frame = get_smplx_data(smplx_data, self.body_model, smplx_output, curr_frame=0)

        # Align coordinate frames to match offline pipeline (rotate to robot frame)
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)
        for joint_name in per_frame.keys():
            orientation = gmr_utils.quat_mul(rotation_quat, per_frame[joint_name][1])
            position = per_frame[joint_name][0] @ rotation_matrix.T
            per_frame[joint_name] = (position, orientation)

        if self.cfg.gmr.visualize:
            # viewer mode → one retarget via vis_step
            motion_data = self.gmr.vis_step(per_frame)
        elif not self.cfg.gmr.visualize:
            # headless mode → one retarget via step()/step_full(); compute velocities from motion_data
            motion_data = self.gmr.step(per_frame) if not self.cfg.gmr.step_full else self.gmr.step_full(per_frame)
        
        now = time.time()
        root_pos = motion_data["root_pos"].reshape(-1)
        root_rot = motion_data["root_rot"].reshape(-1)
        dof_pos = motion_data["dof_pos"].reshape(-1)
        
        ## **Match with Offline format, make root pos x,y to 0 and root vel meaningless
        root_pos[0] = 0
        root_pos[1] = 0
        
        root_vel, root_ang_vel, dof_vel = self._compute_fd(now, root_pos, root_rot, dof_pos)
        motion_data["root_vel"] = None if root_vel is None else root_vel[None, ...]
        motion_data["root_ang_vel"] = None if root_ang_vel is None else root_ang_vel[None, ...]
        motion_data["dof_vel"] = None if dof_vel is None else dof_vel[None, ...]
        return {"motion_data": motion_data}

    def run_once(self) -> dict | None:
        if self.stream is None:
            return None
        frame = self.stream.read()
        if frame is None:
            return None
        return self._process_frame(frame)

    def run_once_with_frame(self, frame) -> dict | None:
        return self._process_frame(frame)

    def close(self):
        self.stream.close()
        self.gmr.close()

    def _compute_fd(self, now: float, root_pos: np.ndarray, root_rot: np.ndarray, dof_pos: np.ndarray):
        root_vel = None
        root_ang_vel = None
        dof_vel = None
        if self._prev["t"] is not None:
            dt = max(now - self._prev["t"], 1e-6)
            root_vel = (root_pos - self._prev["root_pos"]) / dt
            dof_vel = (dof_pos - self._prev["dof_pos"]) / dt
            # angular velocity from quaternion difference
            wxyz_prev = np.array([self._prev["root_rot"][3], *self._prev["root_rot"][:3]])
            wxyz_curr = np.array([root_rot[3], *root_rot[:3]])
            qd = R.from_quat(wxyz_curr) * R.from_quat(wxyz_prev).inv()
            root_ang_vel = qd.as_rotvec() / dt
        self._prev = {
            "t": now,
            "root_pos": root_pos.copy(),
            "root_rot": root_rot.copy(),
            "dof_pos": dof_pos.copy(),
        }
        return root_vel, root_ang_vel, dof_vel


