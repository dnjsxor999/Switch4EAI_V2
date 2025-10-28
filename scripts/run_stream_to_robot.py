import sys
from pathlib import Path
import argparse
import cv2
import time
import threading
import torch

# Ensure repo root is importable before importing our package modules
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Switch4EmbodiedAI.modules.pipeline import StreamToRobotPipeline, PipelineConfig
from Switch4EmbodiedAI.modules.UDPcomm_module import UDPComm
from Switch4EmbodiedAI.utils.interpolator import OutputInterpolator
from Switch4EmbodiedAI.utils import get_user_timing_inputs


def probe_cameras(max_index: int = 10):
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


# Global timing debug state
_debug_timing_state = {
    "enabled": False,
    "last_send_time": None,
    "send_count": 0,
    "target_interval_ms": 100,  # Will be updated based on num_interpolations
}


def send_output_via_udp(out: dict, client: UDPComm, cfg: PipelineConfig, is_interpolated: bool = False):
    """Helper function to send output via UDP."""
    if client is None:
        return
    
    # Debug timing measurement
    current_time = time.time()
    if _debug_timing_state["enabled"]:
        if _debug_timing_state["last_send_time"] is not None:
            interval = current_time - _debug_timing_state["last_send_time"]
            interval_ms = interval * 1000
            output_type = "INTERP" if is_interpolated else "ACTUAL"
            _debug_timing_state["send_count"] += 1
            target = _debug_timing_state["target_interval_ms"]
            print(f"[DEBUG #{_debug_timing_state['send_count']:04d}] {output_type:6s} | Δt = {interval_ms:6.1f}ms | Target: ~{target:.0f}ms")
        else:
            print(f"[DEBUG #0001] {'INTERP' if is_interpolated else 'ACTUAL':6s} | First output (no interval)")
        _debug_timing_state["last_send_time"] = current_time
    
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
    parser.add_argument("--num-interp", type=int, default=1, help="Number of interpolated frames (1=~10Hz, 2=~15Hz, 3=~20Hz, etc.)")
    parser.add_argument("--debug-timing", action="store_true", help="Print timing info for each UDP output (debug mode)")
    args = parser.parse_args()

    if args.list_cams:
        cams = probe_cameras(10)
        print(f"Available cameras: {cams if cams else 'None detected'}")
        return

    # Enable debug timing if requested
    if args.debug_timing:
        _debug_timing_state["enabled"] = True
        if args.no_interpolation:
            target_interval_ms = 200  # 5Hz
        else:
            target_interval_ms = 1000 / (5 * (args.num_interp + 1))
        _debug_timing_state["target_interval_ms"] = target_interval_ms
        print("=" * 70)
        print("DEBUG TIMING MODE ENABLED")
        print("=" * 70)
        print("Legend:")
        print("  ACTUAL = Real processing result from pipeline")
        print("  INTERP = Interpolated output between two actuals")
        print("  Δt     = Time interval since last output")
        print(f"  Target = Expected interval (~{target_interval_ms:.0f}ms)")
        print("=" * 70)

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
    
    use_interpolation = not args.no_interpolation
    
    # Start background thread for interpolated outputs
    stop_event = threading.Event()
    sender_thread = None
    if use_interpolation and client is not None:
        # Initialize interpolator
        interpolator = OutputInterpolator(num_interpolations=args.num_interp)
        
        sender_thread = threading.Thread(
            target=interpolation_sender_thread,
            args=(interpolator, client, cfg, stop_event),
            daemon=True
        )
        sender_thread.start()
        print(f"Interpolation enabled (num_interpolations={args.num_interp})")
    else:
        print("Running at native ~5Hz (no interpolation)")
    
    # Get user timing inputs before starting pipeline
    wait_time, song_duration = get_user_timing_inputs()
    
    # Get default pose from GMR
    default_pose = pipeline.gmr.get_default_pose()

    # Send default pose during wait time
    if client is not None and wait_time > 0:
        print(f"\nSending default pose for {wait_time:.1f} seconds...")
        wait_start = time.time()
        send_interval = 0.1  # Send at 10Hz
        next_send = wait_start

        # Prefill buffer if wait_time is insufficient
        first_frame = pipeline.stream.read()
        required_frames = cfg.gvhmr.win_size
        frames_during_wait = int(wait_time * 10)  # 10Hz step rate
        if frames_during_wait < required_frames:
            prefill_count = 20 + required_frames - frames_during_wait # 20 is safety guard (2sec)
            print(f"Prefilling buffer with {prefill_count} frames...")
            pipeline.gvhmr.prefill_buffer_with_frame(first_frame, prefill_count)
        
        while time.time() - wait_start < wait_time:
            current = time.time()
            if current >= next_send:
                last_pred = pipeline.run_once()
                send_output_via_udp(default_pose, client, cfg, is_interpolated=False)
                next_send += send_interval
            time.sleep(0.01)
        torch.cuda.synchronize()
        print("================================================")
        print("Wait time complete. Starting motion capture...")
        print("================================================")
    
    # pre-initialize interpolator
    # if use_interpolation and last_pred is not None:
    #     interpolator.update(last_pred)
        # first_actual_out = interpolator.get_actual_output()
        # # if first_actual_out is not None:
        # #     send_output_via_udp(first_actual_out, client, cfg, is_interpolated=False)
    
    try:
        capture_start = time.time()
        while time.time() - capture_start < song_duration:
            if last_pred is not None:
                out = last_pred
                last_pred = None
            else:
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
        
        print(f"\nMotion capture complete. Captured for {song_duration:.1f} seconds.")
                
    except KeyboardInterrupt:
        stop_event.set()
        if sender_thread is not None:
            sender_thread.join(timeout=1.0)
        pipeline.close()
    finally:
        stop_event.set()
        if sender_thread is not None:
            sender_thread.join(timeout=1.0)
        pipeline.close()


if __name__ == "__main__":
    main()


