# Vision Encoders

Pluggable vision encoder implementations for robot learning policies.

## Supported Encoders

- **ResNet** - Standard torchvision ResNet backbone
- **DINOv3** - Self-supervised ViT backbone with optional patch merging
- **V-JEPA 2** - Video ViT with temporal context
- **SAM 3** - Segment Anything Model perception encoder

## Installation

```bash
pip install -e .
```

## Usage

```python
from vision import make_vision_encoder

# Create vision encoder from policy config
vision_encoder = make_vision_encoder(config)
```

## External Dependencies

Some encoders require separate installations:
- **DINOv3**: Install from the DINOv3 repository
- **V-JEPA 2**: Install from the V-JEPA 2 repository
- **SAM 3**: Install from the SAM 3 repository

The vision package will only load encoders for which dependencies are installed.
