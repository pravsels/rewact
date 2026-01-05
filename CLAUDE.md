# RewACT Project Guide

## Overview

RewACT (Reward-Augmented Action Chunking with Transformers) is a monorepo for reward-based learning in robotics, built on LeRobot. It extends ACT policies with reward prediction capabilities.

## Package Structure

```
rewact/
├── lerobot_policy_rewact/    # RewACT policy - ACT with distributional value function
├── lerobot_policy_actvantage/# ACTvantage policy - advantage-conditioned ACT (pi*0.6 style)
├── rewact_tools/             # Plugins for robocandywrapper, dataset utilities
├── vision/                   # Vision encoders (ResNet, DINOv3, V-JEPA 2, SAM 3)
├── scripts/                  # Training and visualization scripts (not installable)
├── configs/                  # Training configuration YAML files
└── docker/                   # Docker setup
```

Each package uses `src/` layout with code in `src/<package_name>/`.

## Installation

```bash
# Install packages in editable mode
pip install -e lerobot_policy_rewact/
pip install -e lerobot_policy_actvantage/
pip install -e rewact_tools/
pip install -e vision/
```

## Common Commands

### Training
```bash
# Train RewACT policy
python scripts/train.py --config=configs/train_sam3.yaml

# Train with LeRobot CLI
lerobot-train --policy.type rewact --env.type pusht --steps 200000
```

### Visualization
```bash
python scripts/visualise_reward_predictions.py --dataset-repo-id <repo> --episode-id <id> --policy-path <path>
python scripts/visualise_advantages.py
python scripts/visualise_sam3_focus.py
```

### Slurm (HPC)
```bash
sbatch rewact_train_slurm.sh
sbatch rewact_sam3_viz_slurm.sh
```

## Key Dependencies

- `lerobot >= 0.4` - Core robotics framework
- `torch`, `torchvision` - Deep learning
- `einops` - Tensor operations
- `robocandywrapper` - Mixing multiple dataset types (LeRobot v2.1 and v3)

## Architecture Notes

- Policy plugins integrate with LeRobot's training tools via entry points
- Vision encoders in `vision/` package support multiple backbones: ResNet, DINOv3, V-JEPA 2, SAM 3
- Reward labels can be linear interpolation (start=0, end=1) or manually labeled keypoints
- Training configs are YAML files in `configs/`
