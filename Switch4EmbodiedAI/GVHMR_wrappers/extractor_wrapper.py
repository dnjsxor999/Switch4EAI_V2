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

import torch
from hmr4d.network.hmr2 import load_hmr2, HMR2 # type: ignore
from hmr4d.utils.preproc.vitfeat_extractor import get_batch # type: ignore


from hmr4d.utils.video_io_utils import read_video_np # type: ignore
import cv2
import numpy as np

from hmr4d.network.hmr2.utils.preproc import crop_and_resize, IMAGE_MEAN, IMAGE_STD # type: ignore
# from tqdm import tqdm

class Extractor:
    def __init__(self, tqdm_leave=True):
        self.extractor: HMR2 = load_hmr2().cuda().eval()
        self.tqdm_leave = tqdm_leave

    def extract_video_features(self, video_path, bbx_xys, img_ds=0.5):
        """
        img_ds makes the image smaller, which is useful for faster processing
        """
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        F, _, H, W = imgs.shape  # (F, 3, H, W)
        imgs = imgs.cuda()
        batch_size = 16  # 5GB GPU memory, occupies all CUDA cores of 3090
        features = []
        # for j in tqdm(range(0, F, batch_size), desc="HMR2 Feature", leave=self.tqdm_leave):
        for j in range(0, F, batch_size):
            imgs_batch = imgs[j : j + batch_size]

            with torch.no_grad():
                feature = self.extractor({"img": imgs_batch})
                features.append(feature.detach().cpu())

        features = torch.cat(features, dim=0).clone()  # (F, 1024)
        return features