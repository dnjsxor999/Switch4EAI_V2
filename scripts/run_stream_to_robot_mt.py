import sys
import threading
import queue
import time
from dataclasses import dataclass
from pathlib import Path

import argparse
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Switch4EmbodiedAI.modules.pipeline import PipelineConfig, StreamToRobotPipeline
from Switch4EmbodiedAI.modules.stream_module import SimpleStreamModule
from Switch4EmbodiedAI.modules.gvhmr_realtime import GVHMRRealtimeConfig
from Switch4EmbodiedAI.modules.gmr_retarget import GMRConfig
from Switch4EmbodiedAI.modules.UDPcomm_module import UDPComm


@dataclass
class RunnerConfig:
    lag_frames: int = 6
    visualize: bool = True


class Worker(threading.Thread):
    def __init__(self, name, in_q: queue.Queue, result_q: queue.Queue, app_cfg: PipelineConfig):
        super().__init__(name=name, daemon=True)
        self.in_q = in_q
        self.result_q = result_q
        # pipeline without owning a stream, processes provided frames
        self.pipeline = StreamToRobotPipeline(app_cfg)
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                idx, frame = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            out = self.pipeline.run_once_with_frame(frame)
            if out is not None:
                self.result_q.put((idx, out))

    def stop(self):
        self._stop_evt.set()
        self.pipeline.close()


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

    app_cfg = PipelineConfig()
    app_cfg.use_stream = False
    runner_cfg = RunnerConfig()
    app_cfg.gmr.visualize = runner_cfg.visualize

    if args.video is not None:
        app_cfg.stream.source = "video"
        app_cfg.stream.video_path = args.video
    elif args.camera is not None:
        app_cfg.stream.capture_card_index = args.camera
        app_cfg.stream.source = "camera"
    stream = SimpleStreamModule(app_cfg.stream)
    stream.start()

    q0: queue.Queue = queue.Queue(maxsize=app_cfg.gvhmr.win_size * 2)
    q1: queue.Queue = queue.Queue(maxsize=app_cfg.gvhmr.win_size * 2)
    res_q: queue.Queue = queue.Queue()

    w0 = Worker("worker-0", q0, res_q, app_cfg)
    w1 = Worker("worker-1", q1, res_q, app_cfg)
    w0.start()
    w1.start()

    buffer = []
    next_idx = 0

    client = None
    if app_cfg.udp_enabled:
        client = UDPComm(app_cfg.udp_ip, app_cfg.udp_send_port)

    try:
        while True:
            frame = stream.read()
            if frame is None:
                time.sleep(0.005)
                continue
            buffer.append((next_idx, frame.copy()))
            # feed newest to worker-0
            try:
                q0.put_nowait((next_idx, frame))
            except queue.Full:
                pass
            # feed lagged to worker-1 when available
            if next_idx - runner_cfg.lag_frames >= 0:
                lag_idx = next_idx - runner_cfg.lag_frames
                lag_item = buffer[lag_idx]
                try:
                    q1.put_nowait(lag_item)
                except queue.Full:
                    pass

            next_idx += 1

            # drain results quickly (optional)
            drained = 0
            while True:
                try:
                    idx, out = res_q.get_nowait()
                except queue.Empty:
                    break
                drained += 1
                if client is not None:
                    if "qpos" in out:
                        arr = out["qpos"].astype("float32").flatten()
                    else:
                        md = out["motion_data"]
                        # root_pos = md["root_pos"].reshape(-1)
                        # root_rot = md["root_rot"].reshape(-1)
                        # dof_pos = md["dof_pos"].reshape(-1)
                        # arr = np.concatenate([root_pos, root_rot, dof_pos]).astype("float32")
                    client.send_message(md, app_cfg.udp_ip, app_cfg.udp_send_port)

            # keep buffer bounded
            if len(buffer) > app_cfg.gvhmr.win_size * 4:
                buffer = buffer[-app_cfg.gvhmr.win_size * 4 :]

    except KeyboardInterrupt:
        pass
    finally:
        w0.stop(); w1.stop()
        stream.close()


if __name__ == "__main__":
    main()


