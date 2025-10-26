"""
Interpolator module to double output frequency by interpolating between consecutive outputs.

Timeline:
- t=0.0s: process frame_0 → f_prev (takes 0.2s)
- t=0.2s: process frame_1 → f_curr (takes 0.2s), OUTPUT f_prev
- t=0.3s: OUTPUT interpolated(f_prev, f_curr, alpha=0.5)
- t=0.4s: process frame_2 → f_next, OUTPUT f_curr, f_prev=f_curr, f_curr=f_next
- t=0.5s: OUTPUT interpolated(f_prev, f_curr, alpha=0.5)
- ...continues
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class OutputInterpolator:
    """Interpolates between consecutive pipeline outputs to double output frequency."""
    
    def __init__(self, interpolation_alpha: float = 0.5):
        """
        Args:
            interpolation_alpha: Interpolation factor (0.5 = midpoint between prev and curr)
        """
        self.alpha = interpolation_alpha
        self.f_prev = None
        self.f_curr = None
        self.prev_output_time = None
        self.curr_output_time = None
        self.interpolated_sent = False  # Track if we've sent interpolated value
        
    def update(self, new_output: dict) -> None:
        """
        Update with new pipeline output.
        
        Args:
            new_output: Output from pipeline.run_once() containing either:
                - {"qpos": ..., "derived": {...}} in visualize mode
                - {"motion_data": {...}} in headless mode
        """
        # Shift f_curr to f_prev, new output becomes f_curr
        self.f_prev = self.f_curr
        self.f_curr = new_output
        self.prev_output_time = self.curr_output_time
        self.curr_output_time = time.time()
        self.interpolated_sent = False  # Reset flag for new pair
        
    def should_send_interpolated(self, current_time: float) -> bool:
        """
        Check if we should send an interpolated output now.
        
        Returns True if:
        - We have both f_prev and f_curr
        - We haven't sent interpolated yet for this pair
        - Enough time has passed (alpha fraction of the interval from curr_output_time)
        
        The interpolated output should be sent at:
        curr_output_time + interval * alpha
        where interval = curr_output_time - prev_output_time
        """
        if self.f_prev is None or self.f_curr is None:
            return False
        if self.interpolated_sent:
            return False
        if self.prev_output_time is None or self.curr_output_time is None:
            return False
            
        # Calculate when to send interpolated output
        # It should be sent at alpha fraction of the interval AFTER curr_output_time
        interval = self.curr_output_time - self.prev_output_time
        send_time = self.curr_output_time + interval * self.alpha
        
        return current_time >= send_time
    
    def get_next_output(self) -> tuple[dict | None, str]:
        """
        Get the next output to send.
        
        Returns:
            (output_dict, output_type) where output_type is one of:
                - "actual": Real pipeline output (f_prev)
                - "interpolated": Interpolated output
                - "none": No output ready
        """
        current_time = time.time()
        
        # If we don't have f_prev yet, nothing to send
        if self.f_prev is None:
            return None, "none"
            
        # Check if we should send interpolated output
        if self.should_send_interpolated(current_time):
            interpolated = self._interpolate()
            self.interpolated_sent = True
            return interpolated, "interpolated"
        
        # Otherwise, return None (caller should wait or do other work)
        return None, "none"
    
    def get_actual_output(self) -> dict | None:
        """Get the actual (non-interpolated) output to send (f_prev)."""
        return self.f_prev
    
    def _interpolate(self) -> dict:
        """
        Interpolate between f_prev and f_curr.
        
        Handles both visualize mode (qpos) and headless mode (motion_data).
        """
        if "qpos" in self.f_prev:
            # Visualize mode: interpolate qpos and derived fields
            return self._interpolate_qpos_mode()
        else:
            # Headless mode: interpolate motion_data
            return self._interpolate_motion_data_mode()
    
    def _interpolate_qpos_mode(self) -> dict:
        """Interpolate for visualize mode output."""
        qpos_prev = self.f_prev["qpos"]
        qpos_curr = self.f_curr["qpos"]
        
        # qpos = [root_pos(3), root_rot_wxyz(4), dof_pos(N)]
        root_pos_interp = self._lerp(qpos_prev[:3], qpos_curr[:3])
        root_rot_interp = self._slerp_wxyz(qpos_prev[3:7], qpos_curr[3:7])
        dof_pos_interp = self._lerp(qpos_prev[7:], qpos_curr[7:])
        
        qpos_interp = np.concatenate([root_pos_interp, root_rot_interp, dof_pos_interp])
        
        # Interpolate derived fields
        derived_prev = self.f_prev.get("derived", {})
        derived_curr = self.f_curr.get("derived", {})
        
        derived_interp = {}
        derived_interp["root_pos"] = root_pos_interp.tolist()
        
        # Convert wxyz to xyzw for derived
        root_rot_xyzw_interp = np.array([
            root_rot_interp[1], root_rot_interp[2], root_rot_interp[3], root_rot_interp[0]
        ])
        derived_interp["root_rot_xyzw"] = root_rot_xyzw_interp.tolist()
        derived_interp["dof_pos"] = dof_pos_interp.tolist()
        
        # Interpolate velocities if available
        if derived_prev.get("root_vel") is not None and derived_curr.get("root_vel") is not None:
            derived_interp["root_vel"] = self._lerp(
                np.array(derived_prev["root_vel"]), 
                np.array(derived_curr["root_vel"])
            ).tolist()
        else:
            derived_interp["root_vel"] = None
            
        if derived_prev.get("root_ang_vel") is not None and derived_curr.get("root_ang_vel") is not None:
            derived_interp["root_ang_vel"] = self._lerp(
                np.array(derived_prev["root_ang_vel"]),
                np.array(derived_curr["root_ang_vel"])
            ).tolist()
        else:
            derived_interp["root_ang_vel"] = None
            
        if derived_prev.get("dof_vel") is not None and derived_curr.get("dof_vel") is not None:
            derived_interp["dof_vel"] = self._lerp(
                np.array(derived_prev["dof_vel"]),
                np.array(derived_curr["dof_vel"])
            ).tolist()
        else:
            derived_interp["dof_vel"] = None
        
        return {
            "qpos": qpos_interp,
            "derived": derived_interp
        }
    
    def _interpolate_motion_data_mode(self) -> dict:
        """Interpolate for headless mode output."""
        md_prev = self.f_prev["motion_data"]
        md_curr = self.f_curr["motion_data"]
        
        # Interpolate core fields
        root_pos_interp = self._lerp(
            md_prev["root_pos"].reshape(-1),
            md_curr["root_pos"].reshape(-1)
        )
        
        # root_rot is in xyzw format
        root_rot_interp = self._slerp_xyzw(
            md_prev["root_rot"].reshape(-1),
            md_curr["root_rot"].reshape(-1)
        )
        
        dof_pos_interp = self._lerp(
            md_prev["dof_pos"].reshape(-1),
            md_curr["dof_pos"].reshape(-1)
        )
        
        motion_data_interp = {
            "fps": md_prev.get("fps", 30),
            "root_pos": root_pos_interp[None, ...],
            "root_rot": root_rot_interp[None, ...],
            "dof_pos": dof_pos_interp[None, ...],
        }
        
        # Interpolate optional fields
        if md_prev.get("local_body_pos") is not None and md_curr.get("local_body_pos") is not None:
            local_body_pos_interp = self._lerp(
                md_prev["local_body_pos"].reshape(-1, 3),
                md_curr["local_body_pos"].reshape(-1, 3)
            )
            motion_data_interp["local_body_pos"] = local_body_pos_interp[None, ...]
        
        if md_prev.get("root_vel") is not None and md_curr.get("root_vel") is not None:
            root_vel_interp = self._lerp(
                md_prev["root_vel"].reshape(-1),
                md_curr["root_vel"].reshape(-1)
            )
            motion_data_interp["root_vel"] = root_vel_interp[None, ...]
        else:
            motion_data_interp["root_vel"] = None
            
        if md_prev.get("root_ang_vel") is not None and md_curr.get("root_ang_vel") is not None:
            root_ang_vel_interp = self._lerp(
                md_prev["root_ang_vel"].reshape(-1),
                md_curr["root_ang_vel"].reshape(-1)
            )
            motion_data_interp["root_ang_vel"] = root_ang_vel_interp[None, ...]
        else:
            motion_data_interp["root_ang_vel"] = None
            
        if md_prev.get("dof_vel") is not None and md_curr.get("dof_vel") is not None:
            dof_vel_interp = self._lerp(
                md_prev["dof_vel"].reshape(-1),
                md_curr["dof_vel"].reshape(-1)
            )
            motion_data_interp["dof_vel"] = dof_vel_interp[None, ...]
        else:
            motion_data_interp["dof_vel"] = None
        
        return {"motion_data": motion_data_interp}
    
    def _lerp(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Linear interpolation."""
        return (1 - self.alpha) * a + self.alpha * b
    
    def _slerp_wxyz(self, q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> np.ndarray:
        """Spherical linear interpolation for quaternions in wxyz format."""
        # scipy Rotation expects scalar-last (xyzw)
        q1_xyzw = np.array([q1_wxyz[1], q1_wxyz[2], q1_wxyz[3], q1_wxyz[0]])
        q2_xyzw = np.array([q2_wxyz[1], q2_wxyz[2], q2_wxyz[3], q2_wxyz[0]])
        
        r1 = R.from_quat(q1_xyzw)
        r2 = R.from_quat(q2_xyzw)
        
        slerp = Slerp([0, 1], R.concatenate([r1, r2]))
        r_interp = slerp([self.alpha])[0]
        
        quat_interp_xyzw = r_interp.as_quat()
        # Convert back to wxyz
        quat_interp_wxyz = np.array([
            quat_interp_xyzw[3], quat_interp_xyzw[0], quat_interp_xyzw[1], quat_interp_xyzw[2]
        ])
        return quat_interp_wxyz
    
    def _slerp_xyzw(self, q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
        """Spherical linear interpolation for quaternions in xyzw format."""
        r1 = R.from_quat(q1_xyzw)
        r2 = R.from_quat(q2_xyzw)
        
        slerp = Slerp([0, 1], R.concatenate([r1, r2]))
        r_interp = slerp([self.alpha])[0]
        
        return r_interp.as_quat()
    
    def has_outputs_ready(self) -> bool:
        """Check if we have both prev and curr outputs to work with."""
        return self.f_prev is not None and self.f_curr is not None

