#!/bin/bash
#SBATCH --job-name=sam3_viz
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module purge
module load brics/apptainer-multi-node

# Paths (adapted from your train slurm script)
home_dir="/home/u5dm/pravsels.u5dm"
scratch_dir="/scratch/u5dm/pravsels.u5dm"
repo_dir="${home_dir}/rewact"
data_dir="${scratch_dir}/rewact"
container="${data_dir}/container/rewact_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"

# Config for visualization
DATASET_REPO="danaaubakirova/so100_task_2"
EPISODE_ID=0
SAM3_WEIGHTS="${scratch_dir}/rewact/weights/sam3.pt"
TASK_DESC="Pick up the red cube and place it in the box."
OUTPUT_FILE="${repo_dir}/outputs/sam3_focus_ep${EPISODE_ID}.mp4"

mkdir -p "$(dirname "${OUTPUT_FILE}")"

VIZ_CMD="python scripts/visualise_sam3_focus.py \
    --dataset-repo-id ${DATASET_REPO} \
    --episode-id ${EPISODE_ID} \
    --sam3-weights ${SAM3_WEIGHTS} \
    --task-description \"${TASK_DESC}\" \
    --output ${OUTPUT_FILE}"

srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -c "export PYTHONPATH=${repo_dir}:\$PYTHONPATH && ${VIZ_CMD}"

