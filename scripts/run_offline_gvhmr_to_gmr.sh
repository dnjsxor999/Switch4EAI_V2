#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (this script lives in REPO_ROOT/scripts)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)

# Ensure submodules are on PYTHONPATH (GMR imports general_motion_retargeting; GVHMR utils may be needed)
export PYTHONPATH="${REPO_ROOT}/third_party/GMR:${REPO_ROOT}/third_party/GVHMR:${PYTHONPATH:-}"

# Defaults
DEFAULT_TGT_FOLDER="${REPO_ROOT}/outputs/gvhmr_retargeted"

# Add default --tgt_folder if user didn't provide one
ARGS=("$@")
TGT_SET=false
for a in "${ARGS[@]:-}"; do
  case "$a" in
    --tgt_folder|--tgt_folder=*) TGT_SET=true; break;;
  esac
done
if [ "$TGT_SET" = false ]; then
  ARGS+=("--tgt_folder" "${DEFAULT_TGT_FOLDER}")
fi

# Delegate to GMR script
python3 "${REPO_ROOT}/third_party/GMR/scripts/gvhmr_to_robot_dataset.py" "${ARGS[@]}"

# Example:
#   ./scripts/run_offline_gvhmr_to_gmr.sh \
#     --src_folder ${REPO_ROOT}/third_party/GVHMR/outputs/demo \
#     --robot unitree_g1 --record_video --offset_ground --joint_vel_limit
