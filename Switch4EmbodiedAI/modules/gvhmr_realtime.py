import os
import cv2

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

# Ensure submodules are importable when running from the wrapper repo
REPO_ROOT = Path(__file__).resolve().parents[2]
GVHMR_ROOT = REPO_ROOT / "third_party" / "GVHMR"
if GVHMR_ROOT.exists():
    sys.path.insert(0, str(GVHMR_ROOT))

import hydra
from hydra import initialize_config_module, compose
from ultralytics import YOLO

from hmr4d import PROJ_ROOT # type: ignore
from hmr4d.utils.preproc import Extractor # type: ignore
from Switch4EmbodiedAI.GVHMR_wrappers.vitpose_wrapper import VitPoseExtractor
from Switch4EmbodiedAI.GVHMR_wrappers.extractor_wrapper import Extractor
from hmr4d.configs import register_store_gvhmr # type: ignore

from hmr4d.utils.preproc.vitfeat_extractor import get_batch # type: ignore
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, create_camera_sensor # type: ignore
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL # type: ignore


@dataclass
class GVHMRRealtimeConfig:
    win_size: int = 120
    static_cam: bool = True
    use_dpvo: bool = False
    f_mm: int | None = None
    verbose: bool = False


def _detect_person_xyxy(yolo: YOLO, frame_np: np.ndarray) -> np.ndarray | None:
    res = yolo.predict(frame_np, device="cuda", conf=0.5, classes=0, verbose=False)
    if len(res) == 0:
        return None
    boxes = res[0].boxes
    if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    wh = xyxy[:, 2:] - xyxy[:, :2]
    area = wh[:, 0] * wh[:, 1]
    idx = int(area.argmax())
    return xyxy[idx]


class GVHMRRealtime:
    def __init__(self, cfg: GVHMRRealtimeConfig):
        self.frame_cnt = 0
        self.cfg = cfg

        # YOLO detector
        yolo_w = PROJ_ROOT / "inputs/checkpoints/yolo/yolo11x.pt"
        self.yolo = YOLO(str(yolo_w) if yolo_w.exists() else "yolo11x.pt")

        # Pose extractor and img feature extractor
        self.vitpose_extractor = VitPoseExtractor()
        self.vitpose_extractor.flip_test = False
        self.extractor = Extractor()

        # GVHMR model via hydra config (reuse demo config just to instantiate model + ckpt)
        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            overrides = [
                f"static_cam={int(self.cfg.static_cam)}",
                f"verbose={int(self.cfg.verbose)}",
                f"use_dpvo={int(self.cfg.use_dpvo)}",
            ]
            if self.cfg.f_mm is not None:
                overrides.append(f"f_mm={self.cfg.f_mm}")
            # Register GVHMR config store (mirrors demo script behavior)
            register_store_gvhmr()
            hydra_cfg = compose(config_name="demo", overrides=overrides)

        self.model: DemoPL = hydra.utils.instantiate(hydra_cfg.model, _recursive_=False)
        self.model.load_pretrained_model(GVHMR_ROOT / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt")
        self.model = self.model.eval().cuda()

        # Runtime buffers
        self.initialized_intrinsics = False
        self.K_fullimg_perframe = None
        self.frames_window: list[np.ndarray] = []
        self.bbx_xyxy_window: list[np.ndarray] = []
        self.vitpose_window: torch.Tensor | None = None
        self.f_imgseq_window: torch.Tensor | None = None
        self.last_xyxy: np.ndarray | None = None

    def _ensure_intrinsics(self, width: int, height: int):
        if self.initialized_intrinsics:
            return
        if self.cfg.f_mm is not None:
            self.K_fullimg_perframe = create_camera_sensor(width, height, self.cfg.f_mm)[2]
        else:
            self.K_fullimg_perframe = estimate_K(width, height)
        self.initialized_intrinsics = True

    def prefill_buffer_with_frame(self, frame_bgr: np.ndarray, count: int):
        """
        Prefill buffer with the same frame repeated.
        
        Args:
            frame_bgr: Frame to replicate
            count: Number of times to replicate
        """
        height, width = frame_bgr.shape[:2]
        self._ensure_intrinsics(width, height)
        
        xyxy = _detect_person_xyxy(self.yolo, frame_bgr)
        if xyxy is None:
            xyxy = np.array([0, 0, width - 1, height - 1], dtype=np.float32)
        
        for _ in range(count):
            self.frames_window.append(frame_bgr.copy())
            self.bbx_xyxy_window.append(xyxy.copy())
    
    def overwrite_buffer_with_frame(self, frame_bgr: np.ndarray):
        """
        Overwrite the entire buffer with the same frame repeated.
        
        Args:
            frame_bgr: Frame to replicate
        """
        height, width = frame_bgr.shape[:2]
        self._ensure_intrinsics(width, height)
        
        xyxy = _detect_person_xyxy(self.yolo, frame_bgr)
        if xyxy is None:
            xyxy = np.array([0, 0, width - 1, height - 1], dtype=np.float32)
        
        self.frames_window = [frame_bgr.copy() for _ in range(self.cfg.win_size)]
        self.bbx_xyxy_window = [xyxy.copy() for _ in range(self.cfg.win_size)]

    @torch.no_grad()
    def step(self, frame_bgr: np.ndarray) -> dict | None:
        height, width = frame_bgr.shape[:2]
        self._ensure_intrinsics(width, height)

        xyxy = _detect_person_xyxy(self.yolo, frame_bgr)
        if xyxy is None:
            if self.last_xyxy is None:
                xyxy = np.array([0, 0, width - 1, height - 1], dtype=np.float32)
            else:
                xyxy = self.last_xyxy
        self.last_xyxy = xyxy

        # === DEBUG SAVE BBOX CROP ===
        os.makedirs("dev/debug_bbx", exist_ok=True)
        x1, y1, x2, y2 = xyxy.astype(int)
        crop = frame_bgr[y1:y2, x1:x2]
        cv2.imwrite(f"dev/debug_bbx/frame_{self.frame_cnt:06d}.jpg", crop)
        self.frame_cnt += 1
        # ============================

        self.frames_window.append(frame_bgr)
        self.bbx_xyxy_window.append(xyxy)

        if len(self.frames_window) < self.cfg.win_size:
            return None
        if len(self.frames_window) > self.cfg.win_size:
            self.frames_window.pop(0)
            self.bbx_xyxy_window.pop(0)

        bbx_xyxy_t = torch.from_numpy(np.stack(self.bbx_xyxy_window)).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy_t, base_enlarge=1.2).float()

        if self.vitpose_window is None:
            imgs_t, _ = get_batch(
                np.stack(self.frames_window),
                bbx_xys,
                img_ds=1.0,
                img_dst_size=256,
                path_type="np",
            )
            self.vitpose_window = self.vitpose_extractor.extract(imgs_t, bbx_xys)
            self.f_imgseq_window = self.extractor.extract_video_features(imgs_t, bbx_xys)
        else:
            bbx_xys_last = get_bbx_xys_from_xyxy(
                torch.from_numpy(self.bbx_xyxy_window[-1]).float().unsqueeze(0), base_enlarge=1.2
            ).float()
            imgs_last_t, _ = get_batch(
                np.expand_dims(self.frames_window[-1], 0),
                bbx_xys_last,
                img_ds=1.0,
                img_dst_size=256,
                path_type="np",
            )
            vitpose_last = self.vitpose_extractor.extract(imgs_last_t, bbx_xys_last)
            self.vitpose_window = torch.cat([self.vitpose_window[1:], vitpose_last], dim=0)
            feat_last = self.extractor.extract_video_features(imgs_last_t, bbx_xys_last)
            self.f_imgseq_window = torch.cat([self.f_imgseq_window[1:], feat_last], dim=0)

        if self.cfg.static_cam:
            cam_angvel = torch.zeros((len(self.frames_window), 6)).float()
        else:
            raise NotImplementedError("Non-static camera is not implemented.")

        K_fullimg = self.K_fullimg_perframe.repeat(len(self.frames_window), 1, 1)

        data_window = {
            "length": torch.tensor(len(self.frames_window)),
            "bbx_xys": bbx_xys,
            "kp2d": self.vitpose_window,
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "f_imgseq": self.f_imgseq_window,
        }

        pred_w = self.model.predict(data_window, static_cam=self.cfg.static_cam)
        # pred_w = {k: {kk: vv.detach().cpu() for kk, vv in v.items()} for k, v in pred_w.items()}

        last_idx = -1
        smpl_params_global = {k: v[last_idx].detach().cpu() for k, v in pred_w["smpl_params_global"].items()}
        return {"smpl_params_global": smpl_params_global}


