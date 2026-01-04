#!/bin/bash
#SBATCH --job-name=rewact_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --requeue

module purge
module load brics/apptainer-multi-node

# Paths
home_dir="/home/u5dm/pravsels.u5dm"
scratch_dir="/scratch/u5dm/pravsels.u5dm"
repo_dir="${home_dir}/rewact"
data_dir="${scratch_dir}/rewact"
container="${data_dir}/container/rewact_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"

# Training config
CONFIG_FILE="configs/train_sam3.yaml"

# Extract job_name from config file
JOB_NAME=$(grep "job_name:" "${CONFIG_FILE}" | awk '{print $NF}' | tr -d '"'\'' ')
LAST_CHECKPOINT="${data_dir}/outputs/train/${JOB_NAME}/checkpoints/last"

mkdir -p "${HF_CACHE}" "${data_dir}/outputs"

start_time="$(date -Is --utc)"

# Auto-resume logic: check if a "last" checkpoint exists on scratch
if [ -d "${LAST_CHECKPOINT}" ]; then
    echo "Found existing checkpoint at ${LAST_CHECKPOINT}. Resuming training..."
    TRAIN_CMD="python scripts/train.py \
        --config=${CONFIG_FILE} \
        --output_dir=${data_dir}/outputs/train/${JOB_NAME} \
        --checkpoint_path=${LAST_CHECKPOINT} \
        --resume=true"
else
    echo "No checkpoint found. Starting fresh training..."
    TRAIN_CMD="python scripts/train.py \
        --config=${CONFIG_FILE} \
        --output_dir=${data_dir}/outputs/train/${JOB_NAME} \
        --policy.sam3.weights=${data_dir}/weights/sam3.pt"
fi

srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -c "export PYTHONPATH=${repo_dir}:\$PYTHONPATH && ${TRAIN_CMD}"

end_time="$(date -Is --utc)"
echo
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
