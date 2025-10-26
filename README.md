# Switch4EAI_V2
Switch4EmbodiedAI
- [CoRL Open-Source Hardware in the Era of Robot Learning Workshop](https://easypapersniper.github.io/projects/Switch4EAI/Switch4EAI.html)
- previous version of Switch4EmbodiedAI is [Switch4EAI_V1](https://github.com/EasyPaperSniper/Switch4EmbodiedAI)

## Installation

1) Clone with submodules
```
git clone --recursive https://github.com/dnjsxor999/Switch4EAI_V2.git Switch4EAI
cd Switch4EAI
```

2) Create env
```
conda env create -f environment.yml
conda activate switch4eai
```

3) Install submodules in editable mode (keeps imports stable)
```
pip install -e third_party/GMR
pip install -e third_party/GVHMR
pip install -e . #our switch4eai
```

Notes:
- The env pins PyTorch 2.3.0+cu121 and Pytorch3D wheel for CUDA 12.1. Adjust if your CUDA differs.
- GMR expects body models smplx under `third_party/GMR/assets/body_models/` per its [README](https://github.com/YanjieZe/GMR/blob/master/README.md)
- GVHMR expects checkpoints (YOLO, model ckpt) under `third_party/GVHMR/inputs/checkpoints/...` per its [README](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md)


## Usage

**Online** GVHMR demo from video (single-thread):
```
python scripts/online_gvhmr_test.py --video=/path/to/video.mp4 --win_size=30 -s
```

**Offline** retarget GMR trajectory from GVHMR result(all hmr4d_results.pt):
```
./scripts/run_offline_gvhmr_to_gmr.sh \
     --src_folder /path/to/gvhmr_outputs \
     --tgt_folder /path/to/save_robot_dataset \
     --robot unitree_g1 --record_video --offset_ground --joint_vel_limit
```

**Real-time** demo: Stream -> GVHMR -> GMR (single-thread + interpolating thread):
```
python scripts/run_stream_to_robot.py                # default /dev/video0 (with hardware)
python scripts/run_stream_to_robot.py --list-cams    # list available cameras (with hardware)
python scripts/run_stream_to_robot.py --camera=1     # select /dev/video1 (with hardware)
python scripts/run_stream_to_robot.py --video=/path/to/video.mp4  # for test without hardware (unstable) -> only recommend to check process of pipeline
```

**Real-time** demo: Stream -> GVHMR -> GMR (multi-thread + interpolating thread, lag=6 frames):
```
python scripts/run_stream_to_robot_mt.py --list-cams
python scripts/run_stream_to_robot_mt.py --camera=1
python scripts/run_stream_to_robot_mt.py --video=/path/to/video.mp4
```

### Interpolation

By Original, the pipeline processes frames at ~5Hz due to ~0.2s processing time per frame. The **interpolation module** doubles the output frequency to ~10Hz by interpolating between consecutive outputs.

**How it works**:
- Actual outputs: ~5Hz (real processing results)
- Interpolated outputs: ~5Hz (smooth interpolation between consecutive frames)
- **Total: 10Hz output rate** with minimal overhead and preserved real-time performance

**Usage**:
```bash
# With interpolation (default, 10Hz output)
python scripts/run_stream_to_robot.py --camera 0

# Without interpolation (~5Hz output, original behavior)
python scripts/run_stream_to_robot.py --camera 0 --no-interpolation

# Multiple interpolations for higher frequency
python scripts/run_stream_to_robot.py --camera 0 --num-interp 1  # ~x hz
python scripts/run_stream_to_robot.py --camera 0 --num-interp 2  # ~x*2 hz
python scripts/run_stream_to_robot.py --camera 0 --num-interp 3  # ~x*3 hz

# Debug timing mode (verify output frequency)
python scripts/run_stream_to_robot.py --camera 0 --debug-timing
```

**Features**:
- Flexible output frequency: 10Hz, 15Hz, 20Hz, or higher (via `--num-interp`)
- Smooth interpolation: LERP for positions, SLERP for rotations
- Real-time performance preserved (no accumulating lag)
- One-time 0.2s initial lag (acceptable to test with game)

**Documentation**:
- Detailed guide: `INTERPOLATION_README.md`

### Real-time with Nintendo Switch (capture card)

1) Hardware hookup
- Connect Nintendo Switch HDMI OUT → capture card HDMI IN
- Connect capture card USB to your Ubuntu machine
- Verify device appears (e.g., `/dev/video0`, `/dev/video1`)

2) Discover the correct camera index
```
python scripts/run_stream_to_robot.py --list-cams
# Example: Available cameras: [1]
```

3) Run the pipeline
```
python scripts/run_stream_to_robot.py --camera=1
```

4) Troubleshooting
- If OpenCV fails to open the device, try another index or ensure no other app is using it
- Check permissions (your user in `video` group), and that `/dev/videoN` exists

### Configuration reference
- `SimpleStreamModuleConfig` in `Switch4EAI/modules/stream_modules.py`
  - capture_card_index (int): camera index matching `/dev/videoN`
  - source (str): "camera" or "video" (for "video" only recommend to check process of pipeline)
  - video_path (str|None): path to a test video when source=="video"
  - loop_video (bool): loop video when it ends
- `GVHMRRealtimeConfig` in `Switch4EAI/modules/gvhmr_realtime.py`
  - win_size (int): sliding window length, buffer size (higher = more latency, stabler)
  - static_cam (bool): only static cam is supported here (leave True)
  - use_dpvo (bool): not wired in this repo (leave False)
  - f_mm (int|None): focal length override (leave False)
  - verbose (bool): (leave False)
- `GMRConfig` in `Switch4EAI/modules/gmr_retarget.py`
  - robot (str): target robot name
  - visualize (bool): open viewer and return qpos per step; else headless
  - step_full (bool): in headless mode, include local_body_pos
  - record_video (bool), video_path (str|None)
  - rate_limit, joint_vel_limit, collision_avoid, offset_ground (bool)

### Outputs per step
- If `cfg.gmr.visualize = True` (viewer on):
  - Pipeline returns `{ "qpos": np.ndarray }`
  - qpos layout: `[root_pos(3), root_rot(4, wxyz), dof_pos(...)]`
- If `cfg.gmr.visualize = False` (headless):
  - Pipeline returns `{ "motion_data": dict }`
  - motion_data contains:
    - `fps`: int
    - `root_pos`: shape (1, 3)
    - `root_rot`: shape (1, 4) as xyzw
    - `dof_pos`: shape (1, dof_dim)
    - if `step_full=True`: `local_body_pos`: shape (1, num_bodies, 3)

To enable UDP output, edit `Switich4EmbodiedAI/modules/pipeline.py` defaults or set in code:
```python
from Switich4EmbodiedAI.modules.pipeline import PipelineConfig
cfg = PipelineConfig()
cfg.udp_enabled = True
cfg.udp_ip = "xxx.0.0.x"
cfg.udp_send_port = 11111
```

## Project structure

- `third_party/GVHMR`: upstream GVHMR
- `third_party/GMR`: upstream GMR
- `Switich4EmbodiedAI/modules`: modules
  - `stream_module.py`: camera capture
  - `gvhmr_realtime.py`: sliding-window GVHMR per-frame inference
  - `gmr_retarget.py`: per-frame retargeting and optional visualization
  - `pipeline.py`: orchestrates stream->GVHMR->GMR, UDP config
  - `UDPcomm_module.py`: UDP client utilities
- `scripts`:
  - `online_gvhmr_test.py`: imported and adapted from GVHMR
  - `run_stream_to_robot.py`: single-thread streaming pipeline
  - `run_stream_to_robot_mt.py`: dual-worker pipeline to hide ~200ms GVHMR latency
  - `run_offline_gvhmr_to_gmr.sh`: batch convert GVHMR outputs via GMR

## Tips

- If imports like `hmr4d` or `general_motion_retargeting` fail, ensure you ran the `pip install -e` steps above.
- On first run, ultralytics may download YOLO weights if not found at `third_party/GVHMR/inputs/checkpoints/yolo/yolov8x.pt`.
- For non-static cameras, additional VO integration would be needed (not provided by default).

## UDP JSON payloads and receiver

The runners send JSON over UDP each step.

- Visualize=True (viewer on):
  ```json
  {
    "type": "qpos",
    "qpos": [ ... ],
    "root_pos": [x, y, z],
    "root_rot_xyzw": [x, y, z, w],
    "root_vel": [vx, vy, vz] | null,
    "root_ang_vel": [wx, wy, wz] | null,
    "dof_pos": [ ... ],
    "dof_vel": [ ... ] | null
  }
  ```

- Visualize=False (headless):
  ```json
  {
    "type": "motion_data",
    "fps": 30,
    "root_pos": [x, y, z],
    "root_rot": [x, y, z, w],   
    "dof_pos": [ ... ],
    "local_body_pos": [[x,y,z], ...],   // present if step_full=True
    "root_vel": [vx, vy, vz] | null,
    "root_ang_vel": [wx, wy, wz] | null,
    "dof_vel": [ ... ] | null
  }
  ```

Minimal JSON UDP receiver:
```python
import socket, json

UDP_IP = "xxx.0.0.x"
UDP_PORT = 11111
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("listening...")
while True:
  data, addr = sock.recvfrom(65535)
  msg = json.loads(data.decode("utf-8"))
  print(msg.get("type"), msg.keys())
```
