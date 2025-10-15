import sys
from pathlib import Path as _Path

# Ensure GVHMR submodule is importable when running from this repo wrapper
_REPO_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
_GVHMR_PATH = _REPO_ROOT / "third_party" / "GVHMR"
if _GVHMR_PATH.exists():
    sys.path.insert(0, str(_GVHMR_PATH))

import cv2
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from hmr4d.utils.pylogger import Log # type: ignore
import hydra
from hydra import initialize_config_module, compose
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix

from hmr4d.configs import register_store_gvhmr # type: ignore
from hmr4d.utils.video_io_utils import ( # type: ignore
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch # type: ignore
from Switch4EmbodiedAI.GVHMR_wrappers.vitpose_wrapper import VitPoseExtractor
from Switch4EmbodiedAI.GVHMR_wrappers.extractor_wrapper import Extractor

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor # type: ignore
from hmr4d.utils.geo_transform import compute_cam_angvel # type: ignore
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL # type: ignore
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda # type: ignore
from hmr4d.utils.smplx_utils import make_smplx # type: ignore
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points # type: ignore
from tqdm import tqdm
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay # type: ignore
from einops import einsum, rearrange

from hmr4d.utils.preproc.vitfeat_extractor import get_batch # type: ignore
from hmr4d import PROJ_ROOT # type: ignore
from ultralytics import YOLO


CRF = 23  # same as demo.py


class StepTimer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.total_ms = 0.0
        self.last_ms = 0.0

    def start(self):
        self._tic = Log.time()

    def stop(self):
        ms = (Log.time() - self._tic) * 1000.0
        self.last_ms = ms
        self.total_ms += ms
        self.count += 1
        return ms

    @property
    def avg_ms(self):
        return self.total_ms / max(1, self.count)


def parse_args_to_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--use_dpvo", action="store_true", help="If true, use DPVO. By default not using DPVO.")
    parser.add_argument(
        "--f_mm",
        type=int,
        default=None,
        help="Focal length of fullframe camera in mm. Leave it as None to use default values."
        "For iPhone 15p, the [0.5x, 1x, 2x, 3x] lens have typical values [13, 24, 48, 77]."
        "If the camera zoom in a lot, you can try 135, 200 or even larger values.",
    )
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    parser.add_argument("--win_size", type=int, required=True, help="Sliding window size (frames). Step=1, keep last-only after first window.")
    args = parser.parse_args()

    # Input
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")

    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"use_dpvo={args.use_dpvo}",
        ]
        if args.f_mm is not None:
            overrides.append(f"f_mm={args.f_mm}")
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # Output dirs
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to cfg.video_path
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=CRF)
        for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc=f"Copy"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    return cfg, args


def detect_person_xyxy(yolo: YOLO, frame_np: np.ndarray):
    """Detect a single person bbox (xyxy) on the frame. Returns None if not found.
    Picks the largest area person if multiple.
    """
    res = yolo.predict(frame_np, device="cuda", conf=0.5, classes=0, verbose=False)
    if len(res) == 0:
        return None
    boxes = res[0].boxes
    if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
    wh = xyxy[:, 2:] - xyxy[:, :2]
    area = wh[:, 0] * wh[:, 1]
    idx = int(area.argmax())
    return xyxy[idx]


@torch.no_grad()
def run_realtime(cfg, win_size: int):
    Log.info(f"[Realtime] Start!")
    tic = Log.time()

    # Reader & model
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)

    # Initialize detectors/extractors
    # Prefer local weights if available; fallback to default hub weights
    _yolo_w = (PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")
    yolo = YOLO(str(_yolo_w) if _yolo_w.exists() else "yolov8x.pt")
    vitpose_extractor = VitPoseExtractor()
    # Optimize ViTPose latency: disable flip testing (halve pose inference cost)
    vitpose_extractor.flip_test = False
    extractor = Extractor()

    # Model
    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    # model.load_pretrained_model(cfg.ckpt_path)
    model.load_pretrained_model(_GVHMR_PATH / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt")
    model = model.eval().cuda()

    # Intrinsics per frame (full image)
    if cfg.f_mm is not None:
        K_fullimg_perframe = create_camera_sensor(width, height, cfg.f_mm)[2]
    else:
        K_fullimg_perframe = estimate_K(width, height)

    # Outputs accumulator
    out_incam = None
    out_global = None

    # Sliding window buffers
    frames_window = []  # list of np.ndarray (H, W, 3) RGB
    bbx_xyxy_window = []  # list of (4,) np arrays
    vitpose_window = None  # torch.Tensor (W, 17, 3)
    f_imgseq_window = None  # torch.Tensor (W, 1024)
    last_xyxy = None

    reader = get_video_reader(video_path)
    total_len = length

    # Process frames with timing
    step_timer = StepTimer()
    infer_timer = StepTimer()
    yolo_timer = StepTimer()
    vitpose_timer = StepTimer()
    feat_timer = StepTimer()
    pbar = tqdm(reader, total=length, desc="Processing Frames")
    for frame_idx, frame in enumerate(pbar):
        step_timer.start()
        # Detect bbox on current frame
        yolo_timer.start()
        xyxy = detect_person_xyxy(yolo, frame)
        yolo_timer.stop()
        if xyxy is None:
            if last_xyxy is None:
                # fallback to full image bbox
                xyxy = np.array([0, 0, width - 1, height - 1], dtype=np.float32)
            else:
                xyxy = last_xyxy
        last_xyxy = xyxy

        frames_window.append(frame)
        bbx_xyxy_window.append(xyxy)

        # Maintain window size
        if len(frames_window) < win_size:
            continue
        if len(frames_window) > win_size:
            frames_window.pop(0)
            bbx_xyxy_window.pop(0)

        # Build bbx_xys tensor from xyxy window (cheap; recompute each step)
        bbx_xyxy_t = torch.from_numpy(np.stack(bbx_xyxy_window)).float()  # (W, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy_t, base_enlarge=1.2).float()  # (W, 3)

        if vitpose_window is None:
            # First window: compute full window once
            imgs_t, _ = get_batch(
                np.stack(frames_window),
                bbx_xys,
                img_ds=1.0,
                img_dst_size=256,
                path_type="np",
            )  # imgs_t: (W, 3, 256, 256)

            vitpose_timer.start()
            vitpose_window = vitpose_extractor.extract(imgs_t, bbx_xys)
            vitpose_timer.stop()

            feat_timer.start()
            f_imgseq_window = extractor.extract_video_features(imgs_t, bbx_xys)
            feat_timer.stop()
        else:
            # Subsequent windows: compute only the last frame and update ring buffers
            bbx_xys_last = get_bbx_xys_from_xyxy(
                torch.from_numpy(bbx_xyxy_window[-1]).float().unsqueeze(0), base_enlarge=1.2
            ).float()  # (1, 3)
            imgs_last_t, _ = get_batch(
                np.expand_dims(frames_window[-1], 0),
                bbx_xys_last,
                img_ds=1.0,
                img_dst_size=256,
                path_type="np",
            )  # (1, 3, 256, 256)

            vitpose_timer.start()
            vitpose_last = vitpose_extractor.extract(imgs_last_t, bbx_xys_last)  # (1, 17, 3)
            vitpose_timer.stop()
            vitpose_window = torch.cat([vitpose_window[1:], vitpose_last], dim=0)

            feat_timer.start()
            feat_last = extractor.extract_video_features(imgs_last_t, bbx_xys_last)  # (1, 1024)
            feat_timer.stop()
            f_imgseq_window = torch.cat([f_imgseq_window[1:], feat_last], dim=0)

        # cam_angvel for window
        if cfg.static_cam:
            cam_angvel = torch.zeros((len(frames_window), 6)).float()
        else:
            raise NotImplementedError("Realtime non-static cam (DPVO) is not implemented in this script.")

        # K_fullimg per frame for window
        K_fullimg = K_fullimg_perframe.repeat(len(frames_window), 1, 1)

        # Build data dict for model
        data_window = {
            "length": torch.tensor(len(frames_window)),
            "bbx_xys": bbx_xys,
            "kp2d": vitpose_window,
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "f_imgseq": f_imgseq_window,
        }

        # Predict
        infer_timer.start()
        pred_w = detach_to_cpu(model.predict(data_window, static_cam=cfg.static_cam))
        infer_timer.stop()

        # Initialize outputs on first window
        if out_incam is None:
            out_incam = {k: torch.zeros((total_len,) + v.shape[1:], dtype=v.dtype) for k, v in pred_w["smpl_params_incam"].items()}
            out_global = {k: torch.zeros((total_len,) + v.shape[1:], dtype=v.dtype) for k, v in pred_w["smpl_params_global"].items()}

            # Fill first window fully (0..W-1)
            for k, v in pred_w["smpl_params_incam"].items():
                out_incam[k][:win_size] = v
            for k, v in pred_w["smpl_params_global"].items():
                out_global[k][:win_size] = v
        else:
            # Fill only the last frame of this window
            last_idx = frame_idx
            for k, v in pred_w["smpl_params_incam"].items():
                out_incam[k][last_idx] = v[-1]
            for k, v in pred_w["smpl_params_global"].items():
                out_global[k][last_idx] = v[-1]

        step_timer.stop()
        # ms measure box for each step
        pbar.set_postfix({
            "cur_ms": f"{step_timer.last_ms:.1f}",
            "avg_ms": f"{step_timer.avg_ms:.1f}",
            "yolo_ms": f"{yolo_timer.last_ms:.1f}",
            "yolo_avg": f"{yolo_timer.avg_ms:.1f}",
            "pose_ms": f"{vitpose_timer.last_ms:.1f}",
            "pose_avg": f"{vitpose_timer.avg_ms:.1f}",
            "feat_ms": f"{feat_timer.last_ms:.1f}",
            "feat_avg": f"{feat_timer.avg_ms:.1f}",
            "infer_ms": f"{infer_timer.last_ms:.1f}",
            "infer_avg": f"{infer_timer.avg_ms:.1f}",
        })

    # Close reader
    reader.close()

    # Final pred dict
    if out_incam is None:
        raise RuntimeError("No predictions were produced. Check win_size and input video.")

    pred = {
        "smpl_params_incam": out_incam,
        "smpl_params_global": out_global,
        "K_fullimg": K_fullimg_perframe.repeat(total_len, 1, 1),
    }

    # Save
    torch.save(pred, cfg.paths.hmr4d_results)
    Log.info(f"[Realtime] End. Time elapsed: {Log.time()-tic:.2f}s")


def render_incam(cfg):
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load(str(PROJ_ROOT / "hmr4d/utils/body_model/smplx2smpl_sparse.pt")).cuda()
    faces_smpl = make_smplx("smpl").faces

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy

    # -- render mesh -- #
    verts_incam = pred_c_verts
    writer = get_writer(incam_video_path, fps=30, crf=CRF)
    for i, img_raw in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc=f"Rendering Incam"):
        img = renderer.render_mesh(verts_incam[i].cuda(), img_raw, [0.8, 0.8, 0.8])
        writer.write_frame(img)
    writer.close()
    reader.close()


if __name__ == "__main__":
    cfg, args_ns = parse_args_to_cfg()
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    run_realtime(cfg, args_ns.win_size)
    render_incam(cfg)


