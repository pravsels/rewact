#!/usr/bin/env python

# Quickstart:
# python scripts/smoke_test_vision_encoders.py --vjepa2-weights ./vjepa2_vitl.pt --vjepa2-variant vit_large --batch-size 1 --num-cameras 1

# Options: --dinov3-weights, --dinov3-variant, --device, --h, --w, --batch-size, --num-cameras

import argparse
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lerobot_policy_rewact.vision import make_vision_encoder
from lerobot.configs.types import FeatureType, PolicyFeature


def smoke_test_resnet(
    device: str, 
    batch_size: int, 
    h: int, 
    w: int, 
    dim_model: int,
    num_cameras: int,
) -> None:
    cfg = SimpleNamespace(
        vision_encoder_type="resnet",
        freeze_vision_encoder=False,
        dim_model=dim_model,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        replace_final_stride_with_dilation=False,
    )
    enc = make_vision_encoder(cfg).to(device)
    
    for cam_idx in range(num_cameras):
        # Generate synthetic image data (random noise) for shape testing
        img = torch.randn(batch_size, 3, h, w, device=device)
        tokens, pos = enc(img, cam_idx=cam_idx)

        assert pos.shape[0] == tokens.shape[0] and pos.shape[2] == tokens.shape[2], (tokens.shape, pos.shape)
        assert pos.shape[1] in (1, batch_size), pos.shape
        assert tokens.shape[1] == batch_size and tokens.shape[2] == dim_model, tokens.shape
        print(f"[resnet cam={cam_idx}] image tokens: {tuple(tokens.shape)}  pos tokens: {tuple(pos.shape)}")


def smoke_test_dinov3(
    device: str,
    batch_size: int,
    h: int,
    w: int,
    dim_model: int,
    num_cameras: int,
    dinov3_variant: str,
    dinov3_weights: str | None,
) -> None:
    input_features = {
        f"observation.images.cam{i}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w))
        for i in range(num_cameras)
    }
    cfg = SimpleNamespace(
        vision_encoder_type="dinov3",
        freeze_vision_encoder=False,
        dim_model=dim_model,
        image_features=[f"cam{i}" for i in range(num_cameras)],
        input_features=input_features,
        dinov3=SimpleNamespace(
            variant=dinov3_variant,
            weights=dinov3_weights,
            patch_size=16,
        ),
    )

    if not cfg.dinov3.weights:
        print("[dinov3] skipped (pass --dinov3-weights /path/to/ckpt.pth to run)")
        return

    try:
        enc = make_vision_encoder(cfg).to(device)
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"[dinov3] skipped: {e}")
        return

    for cam_idx in range(num_cameras):
        # Generate synthetic image data (random noise) for shape testing
        img = torch.randn(batch_size, 3, h, w, device=device)
        tokens, pos = enc(img, cam_idx=cam_idx)
        assert pos.shape[0] == tokens.shape[0] and pos.shape[2] == tokens.shape[2], (tokens.shape, pos.shape)
        assert pos.shape[1] in (1, batch_size), pos.shape
        assert tokens.shape[1] == batch_size and tokens.shape[2] == dim_model, tokens.shape
        print(f"[dinov3 cam={cam_idx}] tokens: {tuple(tokens.shape)}  pos: {tuple(pos.shape)}")


def smoke_test_vjepa2(
    device: str,
    batch_size: int,
    h: int,
    w: int,
    dim_model: int,
    num_cameras: int,
    vjepa2_variant: str,
    vjepa2_weights: str | None,
) -> None:
    input_features = {
        f"observation.images.cam{i}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w))
        for i in range(num_cameras)
    }
    cfg = SimpleNamespace(
        vision_encoder_type="vjepa2",
        freeze_vision_encoder=False,
        dim_model=dim_model,
        image_features=[f"cam{i}" for i in range(num_cameras)],
        input_features=input_features,
        vjepa2=SimpleNamespace(
            variant=vjepa2_variant,
            weights=vjepa2_weights,
            patch_size=16,
        ),
    )

    if not cfg.vjepa2.weights:
        print("[vjepa2] skipped (pass --vjepa2-weights /path/to/ckpt.pt to run)")
        return

    try:
        enc = make_vision_encoder(cfg).to(device)
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"[vjepa2] skipped: {e}")
        return

    for cam_idx in range(num_cameras):
        # Generate synthetic video data: expand image to (B, 3, T, H, W) with T=tubelet_size
        img = torch.randn(batch_size, 3, h, w, device=device)
        video = img.unsqueeze(2).expand(-1, -1, 2, -1, -1)  # repeat frame for tubelet_size=2
        tokens, pos = enc(video, cam_idx=cam_idx)
        assert pos.shape[0] == tokens.shape[0] and pos.shape[2] == tokens.shape[2], (tokens.shape, pos.shape)
        assert pos.shape[1] in (1, batch_size), pos.shape
        assert tokens.shape[1] == batch_size and tokens.shape[2] == dim_model, tokens.shape
        print(f"[vjepa2 cam={cam_idx}] tokens: {tuple(tokens.shape)}  pos: {tuple(pos.shape)}")


def smoke_test_sam3(
    device: str,
    batch_size: int,
    h: int,
    w: int,
    dim_model: int,
    num_cameras: int,
    sam3_weights: str | None,
    sam3_fpn_level: int,
    sam3_pos_embed_type: str,
    sam3_patch_merge_stages: int,
    sam3_input_resolution: int,
) -> None:
    input_features = {
        f"observation.images.cam{i}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w))
        for i in range(num_cameras)
    }
    cfg = SimpleNamespace(
        vision_encoder_type="sam3",
        freeze_vision_encoder=True,  # typical usage; adapter supports training too
        dim_model=dim_model,
        image_features=[f"cam{i}" for i in range(num_cameras)],
        input_features=input_features,
        sam3=SimpleNamespace(
            variant="vit_l",
            weights=sam3_weights,
            fpn_level=sam3_fpn_level,
            pos_embed_type=sam3_pos_embed_type,
            patch_merge_stages=int(sam3_patch_merge_stages),
            input_resolution=sam3_input_resolution,
        ),
    )

    if not cfg.sam3.weights:
        print("[sam3] skipped (pass --sam3-weights /path/to/sam3.pt to run)")
        return

    try:
        enc = make_vision_encoder(cfg).to(device)
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"[sam3] skipped: {e}")
        return

    for cam_idx in range(num_cameras):
        img = torch.randn(batch_size, 3, h, w, device=device)
        tokens, pos = enc(img, cam_idx=cam_idx)
        assert pos.shape[0] == tokens.shape[0] and pos.shape[2] == tokens.shape[2], (tokens.shape, pos.shape)
        assert pos.shape[1] in (1, batch_size), pos.shape
        assert tokens.shape[1] == batch_size and tokens.shape[2] == dim_model, tokens.shape
        print(f"[sam3 cam={cam_idx}] tokens: {tuple(tokens.shape)}  pos: {tuple(pos.shape)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Quick shape smoke test for RewACT vision encoders.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dim-model", type=int, default=512)
    p.add_argument("--h", type=int, default=480)
    p.add_argument("--w", type=int, default=640)
    p.add_argument("--num-cameras", type=int, default=2)
    p.add_argument(
        "--dinov3-variant",
        type=str,
        default="vitb16",
        choices=["vitb16", "vitl16", "convnext_base", "convnext_large"],
        help="Which DINOv3 backbone variant to instantiate (must match the checkpoint).",
    )
    p.add_argument(
        "--dinov3-weights",
        type=str,
        default=None,
        help="Local path to a DINOv3 checkpoint (must match --dinov3-variant). If provided, also runs the dinov3 check.",
    )
    p.add_argument(
        "--vjepa2-variant",
        type=str,
        default="vit_large",
        choices=["vit_large", "vit_huge", "vit_giant"],
        help="Which V-JEPA 2 backbone variant to instantiate (must match the checkpoint).",
    )
    p.add_argument(
        "--vjepa2-weights",
        type=str,
        default=None,
        help="Local path to a V-JEPA 2 checkpoint (must match --vjepa2-variant). If provided, also runs the vjepa2 check.",
    )
    p.add_argument(
        "--sam3-weights",
        type=str,
        default=None,
        help="Local path to a SAM3 checkpoint (.pt). If provided, also runs the sam3 check.",
    )
    p.add_argument(
        "--sam3-fpn-level",
        type=int,
        default=2,
        help="SAM3 feature pyramid level index to use as tokens (default 2 ~= scale=1.0).",
    )
    p.add_argument(
        "--sam3-pos-embed-type",
        type=str,
        default="act_sinusoidal",
        choices=["sam3", "act_sinusoidal"],
        help="Positional embedding source for SAM3 adapter.",
    )
    p.add_argument(
        "--sam3-use-patch-merge",
        action="store_true",
        help="Legacy flag. For SAM3, patch-merge defaults to 2 stages already; this flag sets --sam3-patch-merge-stages=2 unless explicitly overridden.",
    )
    p.add_argument(
        "--sam3-patch-merge-stages",
        type=int,
        default=2,
        help="Number of 2x2 patch-merge stages for SAM3 (0=none, 1=4x fewer tokens, 2=16x fewer tokens).",
    )
    p.add_argument(
        "--sam3-input-resolution",
        type=int,
        default=1008,
        help="Square canvas size used for SAM3 preprocessing (aspect-ratio resize + letterbox pad). Default 1008.",
    )
    args = p.parse_args()
    # Legacy flag behavior: set to 2 stages unless the user explicitly set a different stage count.
    if args.sam3_use_patch_merge:
        if "--sam3-patch-merge-stages" not in sys.argv:
            args.sam3_patch_merge_stages = 2

    smoke_test_resnet(
        args.device, 
        args.batch_size, 
        args.h, 
        args.w, 
        args.dim_model, 
        args.num_cameras
    )
    
    if args.dinov3_weights:
        smoke_test_dinov3(
            args.device,
            args.batch_size,
            args.h,
            args.w,
            args.dim_model,
            args.num_cameras,
            args.dinov3_variant,
            args.dinov3_weights,
        )
    if args.vjepa2_weights:
        smoke_test_vjepa2(
            args.device,
            args.batch_size,
            args.h,
            args.w,
            args.dim_model,
            args.num_cameras,
            args.vjepa2_variant,
            args.vjepa2_weights,
        )
    if args.sam3_weights:
        smoke_test_sam3(
            args.device,
            args.batch_size,
            args.h,
            args.w,
            args.dim_model,
            args.num_cameras,
            args.sam3_weights,
            args.sam3_fpn_level,
            args.sam3_pos_embed_type,
            args.sam3_patch_merge_stages,
            args.sam3_input_resolution,
        )


if __name__ == "__main__":
    main()
