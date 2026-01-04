from __future__ import annotations

import os

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACTSinusoidalPositionEmbedding2d

from .base import VisionEncoder


class SAM3VisionEncoder(VisionEncoder):
    """SAM3 vision backbone -> feature map tokens.

    Returns flattened spatial tokens + positional tokens, matching `VisionEncoder` contract.

    Notes:
    - SAM3 expects square 1008×1008 inputs with Normalize(mean=.5,std=.5) after scaling to [0,1].
    - We use SAM3's own `vision_pos_enc` by default, with an option to use ACT sinusoidal 2D pos instead.
    - Token-count control is exposed via an optional 2×2 patch-merge (space-to-depth) on the chosen feature map.
    """

    def __init__(
        self,
        *,
        dim_model: int,
        weights: str,
        # Which feature pyramid level to use from SAM3 neck outputs.
        # SAM3 neck uses scale_factors=[4,2,1,0.5] (in that order), so the "native" scale=1.0 is index 2.
        fpn_level: int = 2,
        input_resolution: int = 1008,
        pos_embed_type: str = "sam3",  # "sam3" | "act_sinusoidal"
        patch_merge_stages: int = 2,
        compile_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.dim_model = int(dim_model)
        self.fpn_level = int(fpn_level)
        self.input_resolution = int(input_resolution)
        self.pos_embed_type = str(pos_embed_type)
        self.patch_merge_stages = int(patch_merge_stages)

        if self.input_resolution <= 0:
            raise ValueError(f"input_resolution must be positive. Got {self.input_resolution}.")
        if self.pos_embed_type not in ("sam3", "act_sinusoidal"):
            raise ValueError(f"pos_embed_type must be 'sam3' or 'act_sinusoidal'. Got {self.pos_embed_type}.")
        if self.patch_merge_stages < 0:
            raise ValueError(f"patch_merge_stages must be >= 0. Got {self.patch_merge_stages}.")
        if self.patch_merge_stages > 3:
            raise ValueError(
                f"patch_merge_stages too large ({self.patch_merge_stages}). "
                "Use 0-2 in practice; >3 is likely to over-downsample."
            )
        if not isinstance(weights, str) or len(weights) == 0:
            raise ValueError("sam3.weights must be a non-empty local checkpoint path.")
        weights = os.path.expanduser(weights)
        if not os.path.exists(weights):
            raise FileNotFoundError(f"sam3.weights not found at: {weights}")

        # Lazy import so non-SAM3 users don't need the dependency stack.
        try:
            from sam3.model_builder import build_sam3_image_model
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "SAM3 is not available. Install the local repo, e.g. `pip install -e /home/user/Desktop/code/sam3`, "
                "and ensure its dependencies are installed."
            ) from e

        # Build full SAM3 image model, but we only use `model.backbone.forward_image(...)`.
        # Avoid HF downloads by disabling load_from_HF and requiring a local checkpoint.
        self.sam3_model = build_sam3_image_model(
            checkpoint_path=weights,
            load_from_HF=False,
            eval_mode=True,
            enable_segmentation=False,
            enable_inst_interactivity=False,
            compile=bool(compile_backbone),
        )

        # SAM3 neck outputs are 256-d features/pos-enc maps.
        self._sam3_dim = 256
        self.feat_proj = nn.Conv2d(self._sam3_dim, self.dim_model, kernel_size=1)
        self.pos_proj = nn.Conv2d(self._sam3_dim, self.dim_model, kernel_size=1)

        if self.patch_merge_stages > 0:
            # After each 2×2 merge, channels become 4*D; project back to D.
            self.merge_proj = nn.Conv2d(self.dim_model * 4, self.dim_model, kernel_size=1)

        if self.pos_embed_type == "act_sinusoidal":
            self.pos_embed_2d = ACTSinusoidalPositionEmbedding2d(self.dim_model // 2)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze only the SAM3 backbone/model. Keep projection/merge layers trainable."""
        self.sam3_model.requires_grad_(not freeze)
        if freeze:
            self.sam3_model.eval()

    @staticmethod
    def _to_float01(x: Tensor) -> Tensor:
        """Best-effort conversion to float32 in [0, 1]."""
        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32) / 255.0
        else:
            x = x.to(dtype=torch.float32)
            # Heuristic: if values look like [0,255], rescale to [0,1].
            if torch.isfinite(x).all() and x.max().item() > 1.5:
                x = x / 255.0
        return x

    def _preprocess_for_sam3(self, img: Tensor) -> Tensor:
        """Aspect-ratio preserving resize + letterbox pad to SAM3's square input, then normalize to [-1,1]."""
        img01 = self._to_float01(img)
        _, _, h, w = img01.shape

        # Resize so the longer side fits `input_resolution`.
        scale = float(self.input_resolution) / float(max(h, w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        img01 = F.interpolate(img01, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # Pad to a square canvas (letterbox). Use 0.5 in [0,1] so normalized value is 0 (neutral).
        pad_h = self.input_resolution - new_h
        pad_w = self.input_resolution - new_w
        if pad_h < 0 or pad_w < 0:  # pragma: no cover
            raise RuntimeError(
                f"Unexpected negative padding: input_resolution={self.input_resolution}, resized={(new_h, new_w)}."
            )
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        img01 = F.pad(img01, (pad_left, pad_right, pad_top, pad_bottom), value=0.5)

        # Normalize(mean=.5,std=.5) => map [0,1] -> [-1,1]
        return (img01 - 0.5) / 0.5

    @staticmethod
    def _merge_2x2(feat: Tensor, *, merge_proj: nn.Module) -> Tensor:
        """2×2 patch merge: (B,D,H,W) -> (B,D,H/2,W/2) with learned projection."""
        b, d, h, w = feat.shape
        # Crop to even spatial sizes (simplest, stable behavior).
        h2 = (h // 2) * 2
        w2 = (w // 2) * 2
        feat = feat[:, :, :h2, :w2]
        feat = einops.rearrange(feat, "b d (h p1) (w p2) -> b (d p1 p2) h w", p1=2, p2=2)
        feat = merge_proj(feat)
        return feat

    def forward(self, img: Tensor, *, cam_idx: int = 0) -> tuple[Tensor, Tensor]:
        # 1) Resize + normalize to SAM3 expected input format.
        img = self._preprocess_for_sam3(img)

        # 2) Run SAM3 backbone on image.
        # If the backbone is frozen (all params require_grad=False), avoid autograd overhead.
        if any(p.requires_grad for p in self.sam3_model.parameters()):
            backbone_out = self.sam3_model.backbone.forward_image(img)
        else:
            with torch.no_grad():
                backbone_out = self.sam3_model.backbone.forward_image(img)
        feats = backbone_out["backbone_fpn"]
        sam3_pos = backbone_out["vision_pos_enc"]

        feat_map = feats[self.fpn_level]  # (B, 256, H', W')
        if self.pos_embed_type == "sam3":
            # `vision_pos_enc` is effectively position-only; keep it broadcastable across batch like other encoders.
            pos_map = sam3_pos[self.fpn_level][:1]  # (1, 256, H', W')
            pos_map = self.pos_proj(pos_map).to(dtype=feat_map.dtype)
        else:
            # ACT sinusoidal position embedding ignores cam_idx.
            _, _, hp, wp = feat_map.shape
            dummy = torch.zeros((1, self.dim_model, hp, wp), device=feat_map.device, dtype=feat_map.dtype)
            pos_map = self.pos_embed_2d(dummy)  # (1, D, Hp, Wp)

        feat_map = self.feat_proj(feat_map)

        for _ in range(self.patch_merge_stages):
            feat_map = self._merge_2x2(feat_map, merge_proj=self.merge_proj)
            pos_map = self._merge_2x2(pos_map, merge_proj=self.merge_proj)

        tokens = einops.rearrange(feat_map, "b d h w -> (h w) b d").contiguous()
        pos_tokens = einops.rearrange(pos_map, "b d h w -> (h w) b d").contiguous()

        return tokens, pos_tokens

