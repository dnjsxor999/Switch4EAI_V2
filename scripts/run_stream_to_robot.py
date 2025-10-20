import sys
from pathlib import Path
import argparse
import cv2

# Ensure repo root is importable before importing our package modules
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Switch4EmbodiedAI.modules.pipeline import StreamToRobotPipeline, PipelineConfig
from Switch4EmbodiedAI.modules.UDPcomm_module import UDPComm


def probe_cameras(max_index: int = 10):
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None, help="Video capture index (e.g., 0,1,2)")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file to stream instead of camera")
    parser.add_argument("--list-cams", action="store_true", help="List available camera indices and exit")
    args = parser.parse_args()

    if args.list_cams:
        cams = probe_cameras(10)
        print(f"Available cameras: {cams if cams else 'None detected'}")
        return

    cfg = PipelineConfig()
    if args.camera is not None:
        cfg.stream.capture_card_index = args.camera
        cfg.stream.source = "camera"
    if args.video is not None:
        cfg.stream.source = "video"
        cfg.stream.video_path = args.video
    pipeline = StreamToRobotPipeline(cfg)
    client = None
    if cfg.udp_enabled:
        client = UDPComm(cfg.udp_ip, cfg.udp_send_port)
    pipeline.start()
    try:
        while True:
            out = pipeline.run_once()
            if out is None:
                continue
            if client is not None:
                if "qpos" in out:
                    arr = out["qpos"].astype("float32").flatten()
                else:
                    md = out["motion_data"]
                    # # pack root_pos (3), root_rot (4), dof_pos (rest)
                    # root_pos = md["root_pos"].reshape(-1)
                    # root_rot = md["root_rot"].reshape(-1)
                    # dof_pos = md["dof_pos"].reshape(-1)
                    # arr = np.concatenate([root_pos, root_rot, dof_pos]).astype("float32")
                client.send_message(md, cfg.udp_ip, cfg.udp_send_port)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()


