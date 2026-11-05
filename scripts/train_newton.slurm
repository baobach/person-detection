#!/bin/bash
# Fine-tune YOLOv8s on the COCO person subset.
# Submit from the repo root: sbatch scripts/train_newton.slurm
#SBATCH --job-name=person-det
#SBATCH --account=course_cap5415
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# PyTorch's CUDA libraries fail to mmap directly off Lustre. Stage the env on node-local disk.
LOCAL="/tmp/${USER}/person-det-${SLURM_JOB_ID}"
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
echo "staging venv at $LOCAL"
rsync -a .venv/ "$LOCAL/venv/"
# shellcheck disable=SC1091
source "$LOCAL/venv/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "HOST $(hostname) JOB ${SLURM_JOB_ID:-local}"
nvidia-smi
python - << 'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
x = torch.randn(2, 3, device="cuda")
print("matmul_ok", (x @ x.T).shape)
PY

python src/prepare_coco.py --from-zip
python src/train.py --model yolov8s.pt --epochs 50 --batch 32 --workers 8 --name yolov8s_coco_person
