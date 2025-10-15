import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Ensure GVHMR is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
GVHMR_ROOT = REPO_ROOT / "third_party" / "GVHMR"
if GVHMR_ROOT.exists():
    sys.path.insert(0, str(GVHMR_ROOT))

from hmr4d import PROJ_ROOT  # type: ignore
from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps  # type: ignore
from hmr4d.utils.geo_transform import cvt_p2d_from_pm1_to_i  # type: ignore
from hmr4d.utils.geo.flip_utils import flip_heatmap_coco17  # type: ignore
from hmr4d.utils.preproc.vitpose_pytorch import build_model  # type: ignore
from hmr4d.utils.preproc.vitfeat_extractor import get_batch  # type: ignore


class VitPoseExtractor:
    def __init__(self, tqdm_leave=True):
        ckpt_path = PROJ_ROOT / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
        self.pose = build_model("ViTPose_huge_coco_256x192", str(ckpt_path))
        self.pose.cuda().eval()

        self.flip_test = True
        self.tqdm_leave = tqdm_leave

    @torch.no_grad()
    def extract(self, video_path, bbx_xys, img_ds=0.5):
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        L, _, H, W = imgs.shape  # (L, 3, H, W)
        batch_size = 16
        vitpose = []
        # for j in tqdm(range(0, L, batch_size), desc="ViTPose", leave=self.tqdm_leave):
        for j in range(0, L, batch_size):
            # Heat map
            imgs_batch = imgs[j : j + batch_size, :, :, 32:224].cuda()
            if self.flip_test:
                heatmap, heatmap_flipped = self.pose(torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)).chunk(2)
                heatmap_flipped = flip_heatmap_coco17(heatmap_flipped)
                heatmap = (heatmap + heatmap_flipped) * 0.5
                del heatmap_flipped
            else:
                heatmap = self.pose(imgs_batch.clone())  # (B, J, 64, 48)

            if False:
                # Get joint
                bbx_xys_batch = bbx_xys[j : j + batch_size].cuda()
                method = "hard"
                if method == "hard":
                    kp2d_pm1, conf = get_heatmap_preds(heatmap)
                elif method == "soft":
                    kp2d_pm1, conf = get_heatmap_preds(heatmap, soft=True)

                # Convert 64, 48 to 64, 64
                kp2d_pm1[:, :, 0] *= 24 / 32
                kp2d = cvt_p2d_from_pm1_to_i(kp2d_pm1, bbx_xys_batch[:, None])
                kp2d = torch.cat([kp2d, conf], dim=-1)

            else:  # postprocess from mmpose
                bbx_xys_batch = bbx_xys[j : j + batch_size]
                heatmap = heatmap.clone().cpu().numpy()
                center = bbx_xys_batch[:, :2].numpy()
                scale = (torch.cat((bbx_xys_batch[:, [2]] * 24 / 32, bbx_xys_batch[:, [2]]), dim=1) / 200).numpy()
                preds, maxvals = keypoints_from_heatmaps(heatmaps=heatmap, center=center, scale=scale, use_udp=True)
                kp2d = np.concatenate((preds, maxvals), axis=-1)
                kp2d = torch.from_numpy(kp2d)

            vitpose.append(kp2d.detach().cpu().clone())

        vitpose = torch.cat(vitpose, dim=0).clone()  # (F, 17, 3)
        return vitpose


