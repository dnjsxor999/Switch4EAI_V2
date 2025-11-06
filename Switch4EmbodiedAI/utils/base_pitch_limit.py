import numpy as np
import mujoco
from typing import Optional

from mink.limits.limit import Limit, Constraint  # type: ignore
from mink.configuration import Configuration  # type: ignore


class BasePitchConfigurationLimit(Limit):
    """Configuration-like limit for the floating base pitch component.

    This constraint keeps the floating base pitch angle within [-max_pitch, +max_pitch]
    by constraining the tangent increment of the rotational DoF corresponding to the
    chosen axis of the free joint. It behaves similarly to a configuration limit but
    targets only the base pitch.

    Args:
        model: MuJoCo model.
        joint_name: Name of the floating base joint (mjJNT_FREE). If None, the first
            free joint in the model is used.
        max_pitch_rad: Symmetric absolute bound on the base pitch angle in radians.
        axis: One of {"x", "y", "z"} selecting which rotational DoF is considered pitch.
        gain: Scalar in (0, 1] to scale allowed approach toward the bound, similar to
            ConfigurationLimit gain.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_name: Optional[str] = None,
        max_pitch_rad: float = 0.3,
        axis: str = "y",
        gain: float = 1.0,
    ):
        if not 0.0 < gain <= 1.0:
            raise ValueError("gain must be in (0, 1]")
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be one of {'x','y','z'}")

        # Resolve free joint id
        if joint_name is not None:
            jid = model.joint(joint_name).id
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"Joint {joint_name} is not a free joint")
        else:
            jid = None
            for j in range(model.njnt):
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    jid = j
                    break
            if jid is None:
                raise ValueError("No free joint found in the model; cannot apply base pitch limit")

        vadr = model.jnt_dofadr[jid]
        # For a free joint, tangent dofs are [vx, vy, vz, wx, wy, wz]
        axis_map = {"x": 0, "y": 1, "z": 2}
        self._pitch_dof_index = vadr + 3 + axis_map[axis]

        # Store parameters
        self._model = model
        self._max_pitch = float(max_pitch_rad)
        self._gain = float(gain)
        # Single-row projection onto the selected DoF
        self._row = np.zeros((1, model.nv))
        self._row[0, self._pitch_dof_index] = 1.0

    def _get_base_quat_wxyz(self, configuration: Configuration) -> np.ndarray:
        # qpos layout for free joint: [x y z qw qx qy qz]
        # Find corresponding qpos address
        # We search again for robustness in case different model instances are used
        jid = None
        for j in range(self._model.njnt):
            if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                jid = j
                break
        assert jid is not None
        qadr = self._model.jnt_qposadr[jid]
        return configuration.q[qadr + 3 : qadr + 7]

    @staticmethod
    def _quat_to_euler_xyz_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
        # Convert MuJoCo wxyz to xyzw then compute euler xyz
        from scipy.spatial.transform import Rotation as R  # lazy import

        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        euler = R.from_quat(q_xyzw).as_euler('xyz', degrees=False)
        return euler

    def compute_qp_inequalities(self, configuration: Configuration, dt: float) -> Constraint:
        del dt  # configuration-like bound, independent of dt

        # Current pitch angle from base quaternion (xyz convention)
        q_wxyz = self._get_base_quat_wxyz(configuration)
        euler = self._quat_to_euler_xyz_wxyz(q_wxyz)

        # Map selected axis index (x=0,y=1,z=2) to pitch angle component
        # We constrained only that component
        axis_idx = np.array([0, 1, 2])[1]  # default y
        # But ensure match with the selected tangent index
        # Determine axis from chosen dof index modulo 3
        axis_idx = (self._pitch_dof_index - (self._pitch_dof_index // 3) * 3) % 3

        theta = float(euler[axis_idx])
        # Allowed increment bounds to remain within [-max, +max]
        upper = self._gain * (self._max_pitch - theta)
        lower = -self._gain * (self._max_pitch + theta)  # note sign

        G = np.vstack([self._row, -self._row])
        h = np.array([upper, -lower])  # since -Δq ≤ -lower ⇒ Δq ≥ lower
        return Constraint(G=G, h=h)


