"""
Dual-branch DCE-MRI fusion model with temporal Mamba and MIL head.

This module implements:
    Frame-wise twin backbones (ConvNeXt/Swin)
    Temporal Mamba stacks per branch
    Cross-model gated attention fusion 
    Mamba-based MIL head for bag-level classification
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 



def _exists(x) -> bool:
    return x is not None


class SinCosTimeEmbedding(nn.Module):
    """Standard sinusoidal positional encoding along the temporal axis."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if length <= 0:
            raise ValueError("Temporal length must be positive.")
        half_dim = self.dim // 2
        positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(half_dim, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, half_dim))
        )
        emb = torch.zeros(length, self.dim, device=device, dtype=dtype)
        emb[:, 0::2] = torch.sin(positions * div_term)
        emb[:, 1::2] = torch.cos(positions * div_term)
        if self.dim % 2 == 1:
            emb[:, -1] = 0.0
        return emb


class SimpleConvFeatureExtractor(nn.Module):
    """
    Lightweight CNN feature extractor used when timm is unavailable.

    Produces multi-scale feature maps; the last feature is default output.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        layers: int = 4,
    ) -> None:
        super().__init__()
        blocks = []
        channels = in_channels
        for idx in range(layers):
            out_channels = base_channels * (2**idx)
            stride = 2 if idx > 0 else 1
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            )
            channels = out_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats


class FrameBackbone(nn.Module):
    """
    Wrapper that applies a 2D backbone frame-by-frame and aligns channels.

    Added features:
        * in_channels argument to support grayscale inputs
        * min_size parameter to upsample tiny ROIs before passing into large backbones
    """

    def __init__(
        self,
        backbone_name: Optional[str],
        out_channels: int,
        feature_index: int = -2,
        pretrained: bool = False,
        in_channels: int = 3,
        min_size: Optional[int] = 64,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.feature_index = feature_index
        self.min_size = min_size if min_size and min_size > 0 else None

        use_simple = backbone_name in {None, "", "simple"} or timm is None
        if use_simple:
            base = max(16, out_channels // 4)
            self.backbone = SimpleConvFeatureExtractor(
                in_channels=in_channels,
                base_channels=base,
                layers=4,
            )
            feature_channels = [
                base,
                base * 2,
                base * 4,
                base * 8,
            ]
            idx = feature_index % len(feature_channels)
            self.proj = nn.Conv2d(feature_channels[idx], out_channels, kernel_size=1)
            self._use_timm = False
        else:
            if timm is None:
                raise ImportError(
                    "timm is required for the requested backbone. Install with `pip install timm`."
                )
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                in_chans=in_channels,
            )
            feat_info = self.backbone.feature_info  # type: ignore[attr-defined]
            if hasattr(feat_info, "channels"):
                num_chs = feat_info.channels()[feature_index]
            else:
                num_chs = feat_info[feature_index]["num_chs"]  # type: ignore[index]
            self.proj = nn.Conv2d(num_chs, out_channels, kernel_size=1)
            self._use_timm = True

    def _ensure_min_size(self, tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if self.min_size is None or min(h, w) >= self.min_size:
            return tensor
        scale = self.min_size / float(min(h, w))
        new_h = int(math.ceil(h * scale))
        new_w = int(math.ceil(w * scale))
        tensor = F.interpolate(
            tensor,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        return tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shaped [B, T, C, H, W]
        Returns:
            Tensor shaped [B, T, out_channels, Hs, Ws]
        """
        if x.dim() != 5:
            raise ValueError("Expected input shape [B, T, C, H, W].")
        b, t, c, h, w = x.shape
        x_2d = x.view(b * t, c, h, w)
        x_2d = self._ensure_min_size(x_2d, h, w)
        feats = self.backbone(x_2d)  # list of [B*T, C_i, H_i, W_i]
        feat = feats[self.feature_index]
        feat = self.proj(feat)
        hs, ws = feat.shape[-2:]
        feat = feat.view(b, t, self.out_channels, hs, ws)
        return feat


class SimpleMambaBlock(nn.Module):
    """
    Simplified Mamba-style block with depthwise 1D convolution and gating.
    Operates on sequences of shape [N, L, C].
    """

    def __init__(
        self,
        dim: int,
        expansion: int = 2,
        dropout: float = 0.0,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.dw_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.act = nn.SiLU(inplace=True)
        self.down_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        gate = torch.sigmoid(self.gate_proj(x))
        x = self.up_proj(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.down_proj(x)
        x = self.dropout(x * gate)
        return residual + x


class TemporalMamba(nn.Module):
    """
    Applies a stack of SimpleMambaBlocks along the temporal dimension for each spatial location.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        dropout: float = 0.0,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SimpleMambaBlock(
                    dim=dim,
                    expansion=2,
                    dropout=dropout,
                    kernel_size=kernel_size,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, T, C, H, W]
        Returns:
            Tensor [B, T, C, H, W]
        """
        b, t, c, h, w = x.shape
        seq = x.permute(0, 3, 4, 1, 2).contiguous().view(b * h * w, t, c)
        for layer in self.layers:
            seq = layer(seq)
        seq = self.norm(seq)
        seq = seq.view(b, h, w, t, c).permute(0, 3, 4, 1, 2).contiguous()
        return seq


class CrossModalGatedAttention(nn.Module):
    """Cross-modal gated attention applied per spatial location over time."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query/context: [B, T, C, H, W]
        Returns:
            Tensor [B, T, C, H, W]
        """
        if query.shape != context.shape:
            raise ValueError("Query and context must share the same shape.")
        b, t, c, h, w = query.shape
        q = query.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        k = context.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)

        qn = self.norm_q(q)
        kn = self.norm_k(k)
        attn_out, _ = self.attn(
            self.q_proj(qn),
            self.k_proj(kn),
            self.v_proj(kn),
            need_weights=False,
        )
        gate_input = torch.cat([qn, kn], dim=-1)
        gate = self.gate(gate_input)
        out = q + self.dropout(self.out_proj(attn_out) * gate)
        out = self.final_norm(out)
        out = out.view(b, h, w, t, c).permute(0, 3, 4, 1, 2).contiguous()
        return out


class CrossModalFusion(nn.Module):
    """Bidirectional cross-modal attention with configurable fusion strategy."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        fusion_mode: str = "avg",
    ) -> None:
        super().__init__()
        self.branch_ab = CrossModalGatedAttention(dim, num_heads, dropout)
        self.branch_ba = CrossModalGatedAttention(dim, num_heads, dropout)
        self.fusion_mode = fusion_mode
        if fusion_mode == "concat":
            self.fusion_proj = nn.Conv2d(dim * 2, dim, kernel_size=1)
        elif fusion_mode == "avg":
            self.fusion_proj = None
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    def forward(self, fcnn: torch.Tensor, fmamba: torch.Tensor) -> torch.Tensor:
        a2 = self.branch_ab(fcnn, fmamba)
        b2 = self.branch_ba(fmamba, fcnn)
        if self.fusion_mode == "avg":
            fused = 0.5 * (a2 + b2)
        else:  # concat
            fused = torch.cat([a2, b2], dim=2)  # [B, T, 2C, H, W]
            bsz, t, ch2, h, w = fused.shape
            fused = fused.view(bsz * t, ch2, h, w)
            fused = self.fusion_proj(fused)
            fused = fused.view(bsz, t, -1, h, w)
        return fused


class MambaMILHead(nn.Module):
    """
    MIL head with Mamba-style encoder and optional CLS pooling.

    Produces logits and optional attention weights (for attn pooling).
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        depth: int = 2,
        grid_size: int = 4,
        dropout: float = 0.1,
        aggregator: str = "cls",
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.temporal_embed = SinCosTimeEmbedding(dim)
        self.use_cls_token = aggregator == "cls"
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [
                SimpleMambaBlock(dim=dim, expansion=2, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.attn_score = None
        else:
            self.cls_token = None
            self.attn_score = nn.Linear(dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            x: Tensor [B, T, C, H, W]
        Returns:
            dict with logits, bag_repr, attention (if available)
        """
        b, t, c, h, w = x.shape
        pooled = self.pool(x.view(b * t, c, h, w))
        pooled = pooled.view(b, t, c, self.grid_size, self.grid_size)
        tokens = (
            pooled.flatten(3)  # [B, T, C, S*S]
            .permute(0, 1, 3, 2)  # [B, T, S*S, C]
            .reshape(b, t * self.grid_size * self.grid_size, c)
        )

        time_embed = self.temporal_embed(
            t,
            device=x.device,
            dtype=x.dtype,
        )  # [T, C]
        time_embed = time_embed.unsqueeze(1).repeat(1, self.grid_size * self.grid_size, 1)
        time_embed = time_embed.reshape(t * self.grid_size * self.grid_size, c)
        tokens = tokens + time_embed.unsqueeze(0)
        tokens = self.dropout(tokens)

        if self.use_cls_token:
            cls = self.cls_token.expand(b, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        for layer in self.encoder:
            tokens = layer(tokens)

        tokens = self.norm(tokens)
        attn_weights = None

        if self.use_cls_token:
            bag_repr = tokens[:, 0]
        else:
            logits = self.attn_score(tokens)  # [B, L, 1]
            attn_weights = torch.softmax(logits, dim=1)
            bag_repr = torch.sum(attn_weights * tokens, dim=1)

        logits = self.classifier(bag_repr)
        return {
            "logits": logits,
            "bag_repr": bag_repr,
            "attention": attn_weights,
        }


class DCEFusionModel(nn.Module):
    """
    Full DCE-MRI model combining twin backbones, temporal Mamba, cross-modal fusion, and MIL head.

    Important new args for tiny grayscale ROIs:
        * input_channels: set to 1 for single-channel data.
        * cnn_min_size / ssm_min_size: controls internal upsampling threshold for each branch.
    """

    def __init__(
        self,
        cnn_backbone: Optional[str] = "convnext_tiny",
        ssm_backbone: Optional[str] = "swin_tiny_patch4_window7_224",
        embed_dim: int = 256,
        temporal_depth: int = 2,
        temporal_dropout: float = 0.0,
        temporal_kernel: int = 3,
        fusion_heads: int = 4,
        fusion_dropout: float = 0.0,
        fusion_mode: str = "avg",
        mil_depth: int = 2,
        mil_grid_size: int = 4,
        mil_dropout: float = 0.1,
        mil_aggregator: str = "cls",
        num_classes: int = 2,
        pretrained_backbones: bool = False,
        input_channels: int = 1,
        cnn_min_size: Optional[int] = 64,
        ssm_min_size: Optional[int] = 96,
    ) -> None:
        super().__init__()
        self.cnn_branch = FrameBackbone(
            backbone_name=cnn_backbone,
            out_channels=embed_dim,
            feature_index=-2,
            pretrained=pretrained_backbones,
            in_channels=input_channels,
            min_size=cnn_min_size,
        )
        self.ssm_branch = FrameBackbone(
            backbone_name=ssm_backbone,
            out_channels=embed_dim,
            feature_index=-2,
            pretrained=pretrained_backbones,
            in_channels=input_channels,
            min_size=ssm_min_size,
        )
        self.temporal_cnn = TemporalMamba(
            dim=embed_dim,
            depth=temporal_depth,
            dropout=temporal_dropout,
            kernel_size=temporal_kernel,
        )
        self.temporal_ssm = TemporalMamba(
            dim=embed_dim,
            depth=temporal_depth,
            dropout=temporal_dropout,
            kernel_size=temporal_kernel,
        )
        self.fusion = CrossModalFusion(
            dim=embed_dim,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
            fusion_mode=fusion_mode,
        )
        self.head = MambaMILHead(
            dim=embed_dim,
            num_classes=num_classes,
            depth=mil_depth,
            grid_size=mil_grid_size,
            dropout=mil_dropout,
            aggregator=mil_aggregator,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        features_cnn = self.cnn_branch(x)
        features_ssm = self.ssm_branch(x)

        features_cnn = self.temporal_cnn(features_cnn)
        features_ssm = self.temporal_ssm(features_ssm)

        fused = self.fusion(features_cnn, features_ssm)
        head_out = self.head(fused)

        return {
            "logits": head_out["logits"],
            "bag_repr": head_out["bag_repr"],
            "attention": head_out["attention"],
            "fused_features": fused,
            "cnn_features": features_cnn,
            "ssm_features": features_ssm,
        }

    def freeze_backbones(self) -> None:
        for module in (self.cnn_branch, self.ssm_branch):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbones(self) -> None:
        for module in (self.cnn_branch, self.ssm_branch):
            for param in module.parameters():
                param.requires_grad = True


if __name__ == "__main__":
    # Minimal sanity check with tiny grayscale inputs (e.g., 42×39)
    model = DCEFusionModel(
        cnn_backbone="simple",
        ssm_backbone="simple",
        embed_dim=128,
        num_classes=2,
        input_channels=1,
        cnn_min_size=48,
        ssm_min_size=48,
    )
    dummy = torch.randn(2, 9, 1, 42, 39)
    out = model(dummy)
    print("Logits:", out["logits"].shape)
    if out["attention"] is not None:
        print("Attention:", out["attention"].shape)