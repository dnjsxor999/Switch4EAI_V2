# Environment Setup - Installing GVHMR Checkpoints

```bash
# Navigate to GVHMR submodule root directory
cd /home/[user_name]/workspace/Switch4EAI/third_party/GVHMR

# Create directory structure
mkdir -p inputs/checkpoints/{body_models/smpl,body_models/smplx,dpvo,gvhmr,hmr2,vitpose,yolo}

# Recommended: aria2c (install if not available: sudo apt-get install -y aria2)
# Download SMPL body model
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
  https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_NEUTRAL.pkl \
  -d inputs/checkpoints/body_models/smpl -o SMPL_NEUTRAL.pkl

# Download SMPLX body model
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
  https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_NEUTRAL.npz \
  -d inputs/checkpoints/body_models/smplx -o SMPLX_NEUTRAL.npz

# Download DPVO checkpoint
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
  https://huggingface.co/camenduru/GVHMR/resolve/main/dpvo/dpvo.pth \
  -d inputs/checkpoints/dpvo -o dpvo.pth

# Download GVHMR checkpoint
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
  https://huggingface.co/camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt \
  -d inputs/checkpoints/gvhmr -o gvhmr_siga24_release.ckpt

# Download HMR2 checkpoint
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
  "https://huggingface.co/camenduru/GVHMR/resolve/main/hmr2/epoch%3D10-step%3D25000.ckpt" \
  -d inputs/checkpoints/hmr2 -o "epoch=10-step=25000.ckpt"

# Download ViTPose checkpoint
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
  https://huggingface.co/camenduru/GVHMR/resolve/main/vitpose/vitpose-h-multi-coco.pth \
  -d inputs/checkpoints/vitpose -o vitpose-h-multi-coco.pth

# Download YOLOv8 checkpoint
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
  https://huggingface.co/camenduru/GVHMR/resolve/main/yolo/yolov8x.pt \
  -d inputs/checkpoints/yolo -o yolov8x.pt

# Download YOLO11 checkpoint
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt \
  -d inputs/checkpoints/yolo -o yolo11x.pt