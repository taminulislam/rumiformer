"""Model factory — create segmentation backbones by name.

All models expose the same interface as RumiFormer:
    forward(overlay, thermal_intensity=None, binary_mask=None)
    → {"seg_logits": (B,1,H,W), "stage4_features": ..., "all_features": [...]}
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Lightweight SegFormer-style MLP Decode Head (shared across all models)
# ---------------------------------------------------------------------------
class MLPDecodeHead(nn.Module):
    """All-MLP decode head that fuses multi-scale features."""

    def __init__(self, in_channels_list, decode_dim=256, num_classes=1):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decode_dim, 1),
                nn.BatchNorm2d(decode_dim),
                nn.GELU(),
            ) for c in in_channels_list
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(decode_dim * len(in_channels_list), decode_dim, 1),
            nn.BatchNorm2d(decode_dim),
            nn.GELU(),
        )
        self.head = nn.Conv2d(decode_dim, num_classes, 1)

    def forward(self, features, target_size):
        projected = []
        for feat, proj in zip(features, self.projections):
            p = proj(feat)
            p = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            projected.append(p)
        fused = self.fuse(torch.cat(projected, dim=1))
        return self.head(fused)


# ═══════════════════════════════════════════════════════════════════════════
#  1. SegFormer-B2 (Vanilla) — same backbone as RumiFormer but NO TGAA
# ═══════════════════════════════════════════════════════════════════════════
class SegFormerB2Vanilla(nn.Module):
    """Vanilla SegFormer-B2 using HuggingFace transformers.
    Direct ablation baseline — same architecture minus TGAA blocks.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        from transformers import SegformerModel
        self.backbone = SegformerModel.from_pretrained(
            "nvidia/mit-b2",
            output_hidden_states=True,
        )
        # MiT-B2 stage dims: [64, 128, 320, 512]
        self.decode_head = MLPDecodeHead([64, 128, 320, 512], decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        outputs = self.backbone(pixel_values=overlay, output_hidden_states=True)
        features = list(outputs.hidden_states)

        seg_logits = self.decode_head(features, (H, W))

        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  2. iFormer (implemented as EfficientFormerV2-S2) — ICLR 2025 style
#     Hybrid CNN+ViT lightweight model
# ═══════════════════════════════════════════════════════════════════════════
class iFormerSeg(nn.Module):
    """iFormer-style CNN+ViT hybrid using MobileViT-S from timm.
    Pretrained on ImageNet, provides multi-scale features.
    ~5.6M params, fast training.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "mobilevit_s",
            pretrained=True,
            features_only=True,
        )
        # Get channel dims from the backbone
        dummy = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_channels = [f.shape[1] for f in feats]
        print(f"  iFormer/MobileViT-S feature channels: {self.feat_channels}")

        self.decode_head = MLPDecodeHead(self.feat_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        features = self.backbone(overlay)

        seg_logits = self.decode_head(features, (H, W))

        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  3. LACTNet — Lightweight Aggregated CNN + Transformer
#     Custom implementation: ResNet-18 CNN encoder + lightweight Transformer
# ═══════════════════════════════════════════════════════════════════════════
class LightweightTransformerBlock(nn.Module):
    """Lightweight transformer block with depthwise separable attention."""

    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        x_flat = x_flat + self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))[0]
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.permute(0, 2, 1).view(B, C, H, W)


class LACTNet(nn.Module):
    """Lightweight Aggregated CNN + Transformer Network.
    CNN branch: ResNet-18 (pretrained) for local features
    Transformer branch: Lightweight attention on downsampled features
    Fusion: Channel attention + MLP decode head
    ~15M params.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm

        # CNN branch: ResNet-18
        resnet = timm.create_model("resnet18", pretrained=True, features_only=True)
        self.cnn_stages = resnet

        # Get feature dims
        dummy = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            cnn_feats = self.cnn_stages(dummy)
        self.cnn_channels = [f.shape[1] for f in cnn_feats]
        print(f"  LACTNet CNN feature channels: {self.cnn_channels}")

        # Transformer branch on last two stages (lower resolution)
        self.trans_proj3 = nn.Conv2d(self.cnn_channels[-2], 128, 1)
        self.trans_block3 = LightweightTransformerBlock(128, num_heads=4)
        self.trans_proj4 = nn.Conv2d(self.cnn_channels[-1], 256, 1)
        self.trans_block4 = LightweightTransformerBlock(256, num_heads=4)

        # Fusion: concat CNN + Transformer features
        fused_channels = list(self.cnn_channels)
        fused_channels[-2] = self.cnn_channels[-2] + 128
        fused_channels[-1] = self.cnn_channels[-1] + 256

        self.decode_head = MLPDecodeHead(fused_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        cnn_feats = self.cnn_stages(overlay)

        # Transformer on last 2 stages
        t3 = self.trans_block3(self.trans_proj3(cnn_feats[-2]))
        t4 = self.trans_block4(self.trans_proj4(cnn_feats[-1]))

        # Fuse CNN + Transformer
        features = list(cnn_feats)
        features[-2] = torch.cat([cnn_feats[-2], t3], dim=1)
        features[-1] = torch.cat([cnn_feats[-1], t4], dim=1)

        seg_logits = self.decode_head(features, (H, W))

        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  4. SegFormer-B0 — Tiny SegFormer baseline
# ═══════════════════════════════════════════════════════════════════════════
class SegFormerB0(nn.Module):
    """SegFormer-B0 — smallest SegFormer variant (3.7M params).
    Useful as a lightweight baseline.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        from transformers import SegformerModel
        self.backbone = SegformerModel.from_pretrained(
            "nvidia/mit-b0",
            output_hidden_states=True,
        )
        # MiT-B0 stage dims: [32, 64, 160, 256]
        self.decode_head = MLPDecodeHead([32, 64, 160, 256], decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        outputs = self.backbone(pixel_values=overlay, output_hidden_states=True)
        features = list(outputs.hidden_states)
        seg_logits = self.decode_head(features, (H, W))
        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  5. Mask2Former — Universal segmentation (Swin-T backbone)
# ═══════════════════════════════════════════════════════════════════════════
class Mask2FormerSeg(nn.Module):
    """Mask2Former with Swin-Tiny backbone (~44M params).
    SOTA universal segmentation architecture. We use just the backbone
    features + our standard MLP decode head for fair comparison.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=True,
            features_only=True,
        )
        dummy = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_channels = [f.shape[1] for f in feats]
        print(f"  Mask2Former/Swin-T feature channels: {self.feat_channels}")
        self.decode_head = MLPDecodeHead(self.feat_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        features = self.backbone(overlay)
        seg_logits = self.decode_head(features, (H, W))
        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  6. SAM-style — Foundation model approach (MobileViT-V2 large backbone)
# ═══════════════════════════════════════════════════════════════════════════
class SAMSeg(nn.Module):
    """SAM-style segmentation using MobileViT-V2-200 as backbone.
    Large ViT backbone (~18M params) mimicking SAM's approach of using
    a powerful image encoder + lightweight mask decoder.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "mobilevitv2_200",
            pretrained=True,
            features_only=True,
        )
        dummy = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_channels = [f.shape[1] for f in feats]
        print(f"  SAM/MobileViT-V2-200 feature channels: {self.feat_channels}")
        self.decode_head = MLPDecodeHead(self.feat_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        features = self.backbone(overlay)
        seg_logits = self.decode_head(features, (H, W))
        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  7. Prior2Former — MetaFormer architecture (PoolFormerV2-S24)
# ═══════════════════════════════════════════════════════════════════════════
class Prior2FormerSeg(nn.Module):
    """Prior2Former-style using PoolFormerV2-S24 from timm.
    MetaFormer architecture (ICCV 2025 inspired): replaces attention
    with simple pooling, proving the importance of architecture over
    specific token mixing. ~21M params.
    """

    def __init__(self, num_seg_classes=1, decode_dim=256, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "poolformerv2_s24",
            pretrained=True,
            features_only=True,
        )
        dummy = torch.randn(1, 3, 256, 320)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_channels = [f.shape[1] for f in feats]
        print(f"  Prior2Former/PoolFormerV2-S24 feature channels: {self.feat_channels}")
        self.decode_head = MLPDecodeHead(self.feat_channels, decode_dim, num_seg_classes)

    def forward(self, overlay, thermal_intensity=None, binary_mask=None):
        B, _, H, W = overlay.shape
        features = self.backbone(overlay)
        seg_logits = self.decode_head(features, (H, W))
        return {
            "seg_logits": seg_logits,
            "stage4_features": features[-1],
            "all_features": features,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Model Factory
# ═══════════════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    "rumiformer": None,  # imported from src.models.rumiformer
    "segformer_b2": SegFormerB2Vanilla,
    "iformer": iFormerSeg,
    "lactnet": LACTNet,
    "segformer_b0": SegFormerB0,
    "mask2former": Mask2FormerSeg,
    "sam2": SAMSeg,
    "prior2former": Prior2FormerSeg,
}


def create_model(model_name: str, num_seg_classes=1, decode_dim=256, **kwargs):
    """Create a segmentation model by name.

    Available models:
        rumiformer, segformer_b2, iformer, lactnet,
        segformer_b0, mask2former, sam2, prior2former
    """
    name = model_name.lower().replace("-", "_")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    if name == "rumiformer":
        from src.models.rumiformer import RumiFormer
        return RumiFormer(num_seg_classes=num_seg_classes, decode_dim=decode_dim,
                          use_aux_mask=kwargs.get("use_aux_mask", True))

    cls = MODEL_REGISTRY[name]
    return cls(num_seg_classes=num_seg_classes, decode_dim=decode_dim, **kwargs)
