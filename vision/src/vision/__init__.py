from __future__ import annotations

from .base import VisionEncoder, _infer_pos_base_hw


def make_vision_encoder(config) -> VisionEncoder:
    """Factory for RewACT/ACT configs.

    Kept permissive w.r.t. config type so we can call it from RewACTConfig (subclass of ACTConfig).
    """

    vision_type = getattr(config, "vision_encoder_type", "resnet")
    freeze = bool(getattr(config, "freeze_vision_encoder", False))

    if vision_type == "resnet":
        from .resnet import ResNetVisionEncoder
        vision_encoder: VisionEncoder = ResNetVisionEncoder(
            dim_model=int(config.dim_model),
            vision_backbone=str(config.vision_backbone),
            pretrained_backbone_weights=getattr(config, "pretrained_backbone_weights", None),
            replace_final_stride_with_dilation=bool(getattr(config, "replace_final_stride_with_dilation", False)),
        )
    elif vision_type == "dinov3":
        from .dinov3 import DinoV3VisionEncoder
        num_cameras = len(getattr(config, "image_features", []))
        c = config.dinov3
        if c is None:
            raise ValueError("vision_encoder_type='dinov3' requires config.dinov3 to be set.")
        if c.weights is None:
            raise ValueError("vision_encoder_type='dinov3' requires config.dinov3.weights (local checkpoint path).")
        vision_encoder = DinoV3VisionEncoder(
            dim_model=int(config.dim_model),
            num_cameras=num_cameras,
            variant=c.variant,
            weights=str(c.weights),
            vit_patch_size=c.patch_size,
            pos_base_hw=_infer_pos_base_hw(config, vit_patch_size=c.patch_size),
            use_learned_pos_embed=getattr(c, "use_learned_pos_embed", False),
            use_patch_merge=getattr(c, "use_patch_merge", False),
        )
    elif vision_type == "vjepa2":
        from .vjepa2 import VJepa2VisionEncoder
        num_cameras = len(getattr(config, "image_features", []))
        c = config.vjepa2
        if c is None:
            raise ValueError("vision_encoder_type='vjepa2' requires config.vjepa2 to be set.")
        if c.weights is None:
            raise ValueError("vision_encoder_type='vjepa2' requires config.vjepa2.weights (local checkpoint path).")
        vision_encoder = VJepa2VisionEncoder(
            dim_model=int(config.dim_model),
            num_cameras=num_cameras,
            variant=c.variant,
            weights=str(c.weights),
            vit_patch_size=c.patch_size,
            pos_base_hw=_infer_pos_base_hw(config, vit_patch_size=c.patch_size),
            use_learned_pos_embed=getattr(c, "use_learned_pos_embed", False),
            use_patch_merge=getattr(c, "use_patch_merge", False),
        )
    elif vision_type == "sam3":
        from .sam3 import SAM3VisionEncoder
        c = config.sam3
        if c is None:
            raise ValueError("vision_encoder_type='sam3' requires config.sam3 to be set.")
        if c.weights is None:
            raise ValueError("vision_encoder_type='sam3' requires config.sam3.weights (local checkpoint path).")
        patch_merge_stages = int(getattr(c, "patch_merge_stages", 2))
        # Legacy boolean: for SAM3, `use_patch_merge=True` means "use the default merge policy" (2 stages).
        # If users want 0/1/2 explicitly, they should set `patch_merge_stages`.
        if bool(getattr(c, "use_patch_merge", False)) and not hasattr(c, "patch_merge_stages"):
            patch_merge_stages = 2
        vision_encoder = SAM3VisionEncoder(
            dim_model=int(config.dim_model),
            weights=str(c.weights),
            fpn_level=int(getattr(c, "fpn_level", 2)),
            input_resolution=int(getattr(c, "input_resolution", 1008)),
            pos_embed_type=str(getattr(c, "pos_embed_type", "sam3")),
            patch_merge_stages=patch_merge_stages,
            compile_backbone=bool(getattr(c, "compile_backbone", False)),
        )
    else:
        raise ValueError(f"Unknown vision_encoder_type: {vision_type}")

    if freeze:
        vision_encoder.freeze_backbone(True)

    return vision_encoder

