import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange, repeat
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

class TemporalSEAttention(nn.Module):
    """时序SE注意力模块"""
    def __init__(self, num_frames, channels, reduction=4):
        super().__init__()
        self.num_frames = num_frames
        self.channels = channels

        self.avg_pool = nn.AdaptiveAvgPool3d((num_frames, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((num_frames, 1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(num_frames * 2, num_frames // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_frames // reduction, num_frames, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        avg_out = self.avg_pool(x.transpose(1, 2)).squeeze(-1).squeeze(-1)  # [B, C, T]
        max_out = self.max_pool(x.transpose(1, 2)).squeeze(-1).squeeze(-1)  # [B, C, T]
        avg_out = avg_out.mean(dim=1)  # [B, T]
        max_out = max_out.mean(dim=1)  # [B, T]

        gate = self.fc(torch.cat([avg_out, max_out], dim=1))  # [B, T]
        gate = gate.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, T, 1, 1, 1]
        
        return x * gate

class TemporalDecomposedAttention(nn.Module):
    """分解式时序注意力"""
    def __init__(self, dim, num_heads=8, num_frames=9):
        super().__init__()
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.scale = (dim // num_heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.temporal_proj = nn.Linear(num_frames, num_frames)
        self.output_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x: [B, T, N, C] where N is flattened spatial dimension
        B, T, N, C = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b t n (h d) -> b h t n d', h=self.num_heads), qkv)
        dots = torch.einsum('b h t n d, b h s n d -> b h t s n', q, k) * self.scale
        dots = dots.mean(dim=-1)  
        attn = dots.softmax(dim=-1)
        temporal_attn = self.temporal_proj(attn.mean(dim=1))  # [B, T, T]
        
        # Apply attention
        temporal_attn = temporal_attn.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        out = torch.einsum('b h t s n, b h s n d -> b h t n d', temporal_attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)')
        
        return self.output_proj(out)

class CrossModalGatedCoAttention(nn.Module):
    """跨模态门控共注意力"""
    def __init__(self, cnn_dim, transformer_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.cnn_proj = nn.Linear(cnn_dim, transformer_dim)
        self.transformer_proj = nn.Linear(transformer_dim, transformer_dim)
        self.cnn_to_trans = nn.MultiheadAttention(transformer_dim, num_heads, batch_first=True)
        self.trans_to_cnn = nn.MultiheadAttention(transformer_dim, num_heads, batch_first=True)
        self.gate_mlp = nn.Sequential(
            nn.Linear(transformer_dim * 2, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.Sigmoid()
        )
        self.output_cnn = nn.Linear(transformer_dim, cnn_dim)
        self.output_trans = nn.Linear(transformer_dim, transformer_dim)
        
    def forward(self, cnn_feat, trans_feat, global_context=None):
        # cnn_feat: [B, N1, C1], trans_feat: [B, N2, C2]
        B = cnn_feat.shape[0]
        cnn_proj = self.cnn_proj(cnn_feat)
        trans_proj = self.transformer_proj(trans_feat)

        if global_context is None:
            cnn_global = cnn_proj.mean(dim=1)
            trans_global = trans_proj.mean(dim=1)
            global_context = torch.cat([cnn_global, trans_global], dim=-1)

        gate = self.gate_mlp(global_context).unsqueeze(1)
        cnn_attended, _ = self.cnn_to_trans(cnn_proj, trans_proj, trans_proj)
        trans_attended, _ = self.trans_to_cnn(trans_proj, cnn_proj, cnn_proj)
        cnn_attended = cnn_attended * gate
        trans_attended = trans_attended * gate
        cnn_out = cnn_feat + self.output_cnn(cnn_attended)
        trans_out = trans_feat + self.output_trans(trans_attended)

        return cnn_out, trans_out

class MultiScaleFPN(nn.Module):
    """多尺度特征金字塔网络"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
        
    def forward(self, features):
        # features: list of feature maps from different levels
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(features[i]))

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], 
                size=laterals[i - 1].shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )

        outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            outs.append(fpn_conv(laterals[i]))
        
        return outs

class MILTransformerHead(nn.Module):
    """多实例学习Transformer头"""
    def __init__(self, feature_dim, num_classes=2, num_layers=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.instance_proj = nn.Linear(feature_dim, feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.instance_attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
    def forward(self, instance_features):
        # instance_features: [B, N_instances, feature_dim]
        B, N, D = instance_features.shape
        instances = self.instance_proj(instance_features)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, instances], dim=1)
        x = self.transformer(x)

        instance_scores = self.instance_attn(x[:, 1:]).squeeze(-1)
        instance_weights = F.softmax(instance_scores, dim=1)

        cls_feat = x[:, 0]
        instance_feat = (x[:, 1:] * instance_weights.unsqueeze(-1)).sum(dim=1)
        final_feat = cls_feat + instance_feat
        logits = self.classifier(final_feat)
        
        return logits, instance_weights

class TMS_CMAN(nn.Module):
    """时空多尺度跨模态注意力网络"""
    def __init__(
        self,
        num_classes=2,
        num_frames=9,
        cnn_backbone='convnext_tiny',
        transformer_backbone='swin_tiny_patch4_window7_224',
        use_subtraction=True,
        pretrained=True
    ):
        super().__init__()
        self.num_frames = num_frames
        self.use_subtraction = use_subtraction
        self.cnn_branch = timm.create_model(
            cnn_backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4]
        )
        cnn_channels = self.cnn_branch.feature_info.channels()  # [96, 192, 384, 768]
        self.transformer_branch = timm.create_model(
            transformer_backbone,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        trans_dim = self.transformer_branch.num_features  # 768
        self.temporal_se = TemporalSEAttention(num_frames, cnn_channels[-1])
        self.temporal_decomposed = TemporalDecomposedAttention(trans_dim, num_frames=num_frames)
        self.fpn = MultiScaleFPN(cnn_channels, out_channels=256)

        self.cm_gca_1 = CrossModalGatedCoAttention(256, trans_dim)
        self.cm_gca_2 = CrossModalGatedCoAttention(256, trans_dim)
        
        if use_subtraction:
            self.subtraction_branch = timm.create_model(
                'efficientnet_b0',
                pretrained=pretrained,
                num_classes=0
            )
            self.sub_proj = nn.Linear(1280, trans_dim)

        self.mil_head = MILTransformerHead(trans_dim + 256, num_classes)
        self.aux_classifier = nn.Linear(trans_dim, num_classes)
        
    def extract_frame_features(self, x, frame_idx):
        """提取单个时相的特征"""
        cnn_feats = self.cnn_branch(x)
        trans_feat = self.transformer_branch.forward_features(x)
        if len(trans_feat.shape) == 3:  # [B, N, C]
            trans_feat = trans_feat.mean(dim=1)  # Global pooling
        
        return cnn_feats, trans_feat
    
    def forward(self, x, subtraction=None):
        """
        x: [B, T, C, H, W] - 多时相输入
        subtraction: [B, C_sub, H, W] - 减影图像
        """
        B, T, C, H, W = x.shape
        all_cnn_feats = []
        all_trans_feats = []
        
        for t in range(T):
            frame = x[:, t]
            cnn_feats, trans_feat = self.extract_frame_features(frame, t)
            all_cnn_feats.append(cnn_feats)
            all_trans_feats.append(trans_feat)

        cnn_temporal = []
        for level in range(len(all_cnn_feats[0])):
            level_feats = torch.stack([f[level] for f in all_cnn_feats], dim=1)
            if level == len(all_cnn_feats[0]) - 1: 
                level_feats = self.temporal_se(level_feats)
            cnn_temporal.append(level_feats.mean(dim=1))  
        trans_temporal = torch.stack(all_trans_feats, dim=1)  # [B, T, C]
        trans_temporal = trans_temporal.unsqueeze(2)  # [B, T, 1, C]
        trans_temporal = self.temporal_decomposed(trans_temporal)
        trans_temporal = trans_temporal.squeeze(2).mean(dim=1)  # [B, C]

        fpn_features = self.fpn(cnn_temporal)
        
        cnn_feat_1 = fpn_features[1].flatten(2).transpose(1, 2)  # [B, N, 256]
        cnn_feat_2 = fpn_features[2].flatten(2).transpose(1, 2)
        trans_feat_expanded = trans_temporal.unsqueeze(1).expand(-1, cnn_feat_1.shape[1], -1)
        cnn_feat_1, trans_feat_1 = self.cm_gca_1(cnn_feat_1, trans_feat_expanded)
        cnn_feat_2, trans_feat_2 = self.cm_gca_2(cnn_feat_2, trans_feat_expanded)
        
        if self.use_subtraction and subtraction is not None:
            sub_feat = self.subtraction_branch(subtraction)
            sub_feat = self.sub_proj(sub_feat).unsqueeze(1)
            trans_feat_2 = torch.cat([trans_feat_2, sub_feat], dim=1)

        cnn_global = torch.cat([
            cnn_feat_1.mean(dim=1, keepdim=True),
            cnn_feat_2.mean(dim=1, keepdim=True)
        ], dim=1)  # [B, 2, 256]
        trans_global = trans_feat_2.mean(dim=1, keepdim=True)  # [B, 1, trans_dim]

        all_instances = torch.cat([
            cnn_global,
            trans_global.expand(-1, 1, -1)
        ], dim=-1)  # [B, N_instances, feature_dim]
        logits, instance_weights = self.mil_head(all_instances)
        aux_logits = self.aux_classifier(trans_temporal)
        
        return {
            'logits': logits,
            'aux_logits': aux_logits,
            'instance_weights': instance_weights,
            'features': {
                'cnn': cnn_temporal,
                'transformer': trans_temporal,
                'fpn': fpn_features
            }
        }

class TMS_CMAN_WithSegmentation(nn.Module):
    """带分割头的TMS-CMAN（多任务学习）"""
    def __init__(self, tms_cman_model, num_classes_seg=2):
        super().__init__()
        self.backbone = tms_cman_model
        self.seg_decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        ])
        self.seg_head = nn.Conv2d(32, num_classes_seg, kernel_size=1)
        
    def forward(self, x, subtraction=None):
        outputs = self.backbone(x, subtraction)
        fpn_features = outputs['features']['fpn']
        seg_feat = fpn_features[0]
        
        for decoder in self.seg_decoder:
            seg_feat = decoder(seg_feat)

        seg_logits = self.seg_head(seg_feat)
        outputs['seg_logits'] = seg_logits
        return outputs