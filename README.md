# Switch4EAI
Unified wrapper around GVHMR (human motion recovery) and GMR (robot retargeting) with live streaming, multi-threaded execution, and UDP output.

## Installation (single conda env)

1) Clone with submodules
```
git clone --recursive <this-repo-url> Switch4EAI
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

Online GVHMR demo (single-thread):
```
python scripts/online_gvhmr_test.py --video=/path/to/video.mp4 --win_size=30 -s
```

Stream -> GVHMR -> GMR (single-thread):
```
python scripts/run_stream_to_robot.py
```

Stream -> GVHMR -> GMR (multi-thread, lag=6 frames):
```
python scripts/run_stream_to_robot_mt.py
```

To enable UDP output, edit `Switich4EmbodiedAI/modules/pipeline.py` defaults or set in code:
```python
from Switich4EmbodiedAI.modules.pipeline import PipelineConfig
cfg = PipelineConfig()
cfg.udp_enabled = True
cfg.udp_ip = "127.0.0.1"
cfg.udp_send_port = 54010
```

## Project structure

- `third_party/GVHMR`: upstream GVHMR
- `third_party/GMR`: upstream GMR
- `Switich4EmbodiedAI/modules`: wrapper modules
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