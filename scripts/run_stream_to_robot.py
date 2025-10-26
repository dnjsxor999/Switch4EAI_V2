import sys
from pathlib import Path
import argparse
import cv2
import time
import threading

# Ensure repo root is importable before importing our package modules
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Switch4EmbodiedAI.modules.pipeline import StreamToRobotPipeline, PipelineConfig
from Switch4EmbodiedAI.modules.UDPcomm_module import UDPComm
from Switch4EmbodiedAI.modules.interpolator import OutputInterpolator


def probe_cameras(max_index: int = 10):
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


def send_output_via_udp(out: dict, client: UDPComm, cfg: PipelineConfig, is_interpolated: bool = False):
    """Helper function to send output via UDP."""
    if client is None:
        return
    
    if "qpos" in out:
        # visual mode: send JSON with derived fields
        payload = {
            "type": "qpos",
            "qpos": out["qpos"].astype("float32").flatten().tolist(),
            "interpolated": is_interpolated,
        }
        if "derived" in out:
            payload.update(out["derived"])  # already basic python lists
        client.send_json_message(payload, cfg.udp_ip, cfg.udp_send_port)
    else:
        md = out["motion_data"]
        payload = {
            "type": "motion_data",
            "fps": int(md.get("fps", 30)),
            "root_pos": md["root_pos"].reshape(-1).tolist(),
            "root_rot": md["root_rot"].reshape(-1).tolist(),
            "dof_pos": md["dof_pos"].reshape(-1).tolist(),
            "interpolated": is_interpolated,
        }
        if md.get("local_body_pos", None) is not None:
            payload["local_body_pos"] = md["local_body_pos"].reshape(-1, 3).tolist()
        if md.get("root_vel", None) is not None:
            payload["root_vel"] = md["root_vel"].reshape(-1).tolist()
        if md.get("root_ang_vel", None) is not None:
            payload["root_ang_vel"] = md["root_ang_vel"].reshape(-1).tolist()
        if md.get("dof_vel", None) is not None:
            payload["dof_vel"] = md["dof_vel"].reshape(-1).tolist()
        client.send_json_message(payload, cfg.udp_ip, cfg.udp_send_port)


def interpolation_sender_thread(interpolator: OutputInterpolator, client: UDPComm, cfg: PipelineConfig, stop_event: threading.Event):
    """Background thread that sends interpolated outputs at the right time."""
    while not stop_event.is_set():
        if interpolator.has_outputs_ready():
            interp_out, output_type = interpolator.get_next_output()
            if output_type == "interpolated":
                send_output_via_udp(interp_out, client, cfg, is_interpolated=True)
        time.sleep(0.01)  # Check every 10ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None, help="Video capture index (e.g., 0,1,2)")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file to stream instead of camera")
    parser.add_argument("--list-cams", action="store_true", help="List available camera indices and exit")
    parser.add_argument("--no-interpolation", action="store_true", help="Disable output interpolation (run at native 5Hz)")
    parser.add_argument("--interpolation-alpha", type=float, default=0.5, help="Interpolation alpha (0.5 = midpoint)")
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
    
    use_interpolation = not args.no_interpolation
    
    # Start background thread for interpolated outputs
    stop_event = threading.Event()
    sender_thread = None
    if use_interpolation and client is not None:
        # Initialize interpolator
        interpolator = OutputInterpolator(interpolation_alpha=args.interpolation_alpha)

        sender_thread = threading.Thread(
            target=interpolation_sender_thread,
            args=(interpolator, client, cfg, stop_event),
            daemon=True
        )
        sender_thread.start()
        print(f"Interpolation enabled (alpha={args.interpolation_alpha})")
    else:
        print("Running at native ~5Hz (no interpolation)")
    
    pipeline.start()
    
    try:
        while True:
            out = pipeline.run_once()
            if out is None:
                continue
            
            if use_interpolation:
                # Update interpolator with new output
                interpolator.update(out)
                
                # Send actual output (f_prev from interpolator)
                actual_out = interpolator.get_actual_output()
                if actual_out is not None:
                    send_output_via_udp(actual_out, client, cfg, is_interpolated=False)
                # Interpolated output will be sent by background thread
            else:
                # Direct output without interpolation
                send_output_via_udp(out, client, cfg, is_interpolated=False)
                
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if sender_thread is not None:
            sender_thread.join(timeout=1.0)
        pipeline.close()


if __name__ == "__main__":
    main()


