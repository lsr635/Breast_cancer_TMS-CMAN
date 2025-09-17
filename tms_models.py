import torch
import torch.nn as nn
import torch.nn.functional as F

# 采用 torchvision 的 ConvNeXt-T 与 Swin-T
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import swin_t, Swin_T_Weights

def conv1x1(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)

class TemporalSE(nn.Module):
    """
    对输入 [B,T,C,H,W] 做时序SE门控
    """
    def __init__(self, C, r=8):
        super().__init__()
        self.fc1 = nn.Conv1d(C, C // r, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(C // r, C, kernel_size=1, bias=True)

    def forward(self, x):
        # x: [B,T,C,H,W]
        B,T,C,H,W = x.shape
        x_ = x.mean(dim=[3,4])             # [B,T,C]
        x_ = x_.permute(0,2,1)             # [B,C,T]
        y = F.adaptive_avg_pool1d(x_, 1)   # [B,C,1]
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))     # [B,C,1]
        y = y.expand(-1,-1,T)              # [B,C,T]
        y = y.permute(0,2,1)               # [B,T,C]
        y = y.view(B,T,C,1,1)
        return x * y

class TemporalLinearAttn(nn.Module):
    """
    简化线性时序注意力：对每个空间位置的时序向量做线性注意力
    输入: [B,T,C,H,W]，输出同形状
    """
    def __init__(self, C, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (C // heads) ** -0.5
        self.to_q = nn.Conv1d(C, C, 1, bias=False)
        self.to_k = nn.Conv1d(C, C, 1, bias=False)
        self.to_v = nn.Conv1d(C, C, 1, bias=False)
        self.proj = nn.Conv1d(C, C, 1, bias=False)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x_flat = x.permute(0,3,4,1,2).contiguous().view(B*H*W, T, C)  # [BHW, T, C]
        x_flat = x_flat.permute(0,2,1)                                 # [BHW, C, T]

        q = self.to_q(x_flat)       # [BHW,C,T]
        k = self.to_k(x_flat)
        v = self.to_v(x_flat)

        C_h = C // self.heads
        q = q.view(-1, self.heads, C_h, T) * self.scale
        k = k.view(-1, self.heads, C_h, T)
        v = v.view(-1, self.heads, C_h, T)

        k_soft = F.softmax(k, dim=-1)
        v_weighted = torch.einsum('bhct,bhdt->bhcd', k_soft, v)  # sum over T

        q_soft = F.softmax(q, dim=-2)  # over C_h
        out = torch.einsum('bhct,bhcd->bhdt', q_soft, v_weighted)  # [BHW,heads,C_h,T]
        out = out.contiguous().view(-1, C, T)
        out = self.proj(out)          # [BHW,C,T]
        out = out.permute(0,2,1).contiguous().view(B,H,W,T,C)  # [B,H,W,T,C]
        out = out.permute(0,3,4,1,2).contiguous()              # [B,T,C,H,W]
        return out

class CMGCA(nn.Module):
    """
    Cross-Modal Gated Co-Attention
    输入: A=[B,C,H,W], B=[B,C,H,W] （空间对齐，通道对齐）
    输出: 两支融合后的 A', B'
    """
    def __init__(self, C, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (C // heads) ** -0.5
        self.qA = conv1x1(C, C)
        self.kA = conv1x1(C, C)
        self.vA = conv1x1(C, C)
        self.qB = conv1x1(C, C)
        self.kB = conv1x1(C, C)
        self.vB = conv1x1(C, C)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C*2, C//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C//4, C, 1),
            nn.Sigmoid()
        )
        self.projA = conv1x1(C, C)
        self.projB = conv1x1(C, C)

    def _attn(self, q, k, v):
        B,C,H,W = q.shape
        h = self.heads
        c = C // h
        q = q.view(B,h,c,H*W) * self.scale  # [B,h,c,N]
        k = k.view(B,h,c,H*W)               # [B,h,c,N]
        v = v.view(B,h,c,H*W)
        attn = torch.einsum('bhcn,bhdn->bhcd', q, k)  # [B,h,c,c]
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhcd,bhdn->bhcn', attn, v)  # [B,h,c,N]
        out = out.contiguous().view(B,C,H,W)
        return out

    def forward(self, A, B):
        # 门控
        g = self.gate(torch.cat([A,B], dim=1))   # [B,C,1,1]
        # A<-B
        qA, kB, vB = self.qA(A), self.kB(B), self.vB(B)
        msgA = self._attn(qA, kB, vB)
        A2 = A + self.projA(msgA * g)
        # B<-A
        qB, kA, vA = self.qB(B), self.kA(A), self.vA(A)
        msgB = self._attn(qB, kA, vA)
        B2 = B + self.projB(msgB * g)
        return A2, B2

class MILHead(nn.Module):
    """
    简易 TransMIL：将多时相/多实例特征 [B,T,C,H,W] -> tokens [B, N_tokens, C] -> Transformer -> CLS
    """
    def __init__(self, C, num_layers=2, num_heads=4, mlp_ratio=4.0, dropout=0.0, num_classes=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=C, nhead=num_heads, dim_feedforward=int(C*mlp_ratio),
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1,1,C))
        self.norm = nn.LayerNorm(C)
        self.head = nn.Sequential(
            nn.Linear(C, C),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(C, num_classes)
        )
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: [B,T,C,H,W] -> average pool: [B,T,C]
        B,T,C,H,W = x.shape
        x = x.mean(dim=[3,4])   # [B,T,C]
        cls = self.cls_token.expand(B,-1,-1)     # [B,1,C]
        x = torch.cat([cls, x], dim=1)           # [B,1+T,C]
        x = self.encoder(x)                      # [B,1+T,C]
        x = self.norm(x[:,0])                    # [B,C]
        logits = self.head(x)                    # [B,num_classes]
        return logits

class TMS_CMAN(nn.Module):
    """
    TMS-CMAN 最小可行模型：
    - CNN: ConvNeXt-Tiny
    - Transformer: Swin-Tiny
    - TemporalSE + 线性时序注意力
    - 双向门控共注意力融合
    - MILHead
    输入支持：
    - 单帧： [B,3,H,W] -> 转为 [B,1,3,H,W]
    - 多时相： [B,T,3,H,W]
    """
    def __init__(self, num_classes=2, img_size=224, pretrained=True, T_max=9):
        super().__init__()
        self.T_max = T_max
        cnn_weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.cnn = convnext_tiny(weights=cnn_weights)
        self.cnn_out_c = 768

        swin_weights = Swin_T_Weights.DEFAULT if pretrained else None
        self.tr = swin_t(weights=swin_weights)   # 输出通道 768
        self.tr_out_c = 768

        self.C = 512
        self.cnn_proj = conv1x1(self.cnn_out_c, self.C)
        self.tr_proj  = conv1x1(self.tr_out_c,  self.C)

        self.tse_cnn = TemporalSE(self.C)
        self.tla_cnn = TemporalLinearAttn(self.C, heads=4)
        self.tse_tr  = TemporalSE(self.C)
        self.tla_tr  = TemporalLinearAttn(self.C, heads=4)

        self.cmgca = CMGCA(self.C, heads=4)

        self.mil = MILHead(self.C, num_layers=2, num_heads=4, num_classes=num_classes)

    def _extract_cnn(self, x_2d):
        # x_2d: [B,3,H,W] -> feature map [B,C,H',W']
        # convnext_tiny.forward 返回 logits，需取中间层特征。这里用features方法（torchvision>=0.13）
        # 兼容处理：直接用 self.cnn.features
        f = self.cnn.features(x_2d)               # [B,768,Hc,Wc]
        f = self.cnn_proj(f)                      # [B,512,Hc,Wc]
        return f

    def _extract_tr(self, x_2d):
        # swin_t 返回 logits，取中间特征：forward_features
        f = self.tr.features(x_2d)                # [B,768,Ht,Wt]
        f = self.tr_proj(f)                       # [B,512,Ht,Wt]
        return f

    def forward(self, x):
        # x: [B,3,H,W] 或 [B,T,3,H,W]
        if x.dim() == 4:
            x = x.unsqueeze(1)  # [B,1,3,H,W]
        B,T,C,H,W = x.shape

        cnn_feats = []
        tr_feats  = []
        for t in range(T):
            xt = x[:,t]                          # [B,3,H,W]
            fc = self._extract_cnn(xt)           # [B,512,Hc,Wc]
            ft = self._extract_tr(xt)            # [B,512,Ht,Wt]
            # 空间对齐到共同尺度（取较小者，或统一到14x14）
            Ht,Wt = ft.shape[-2:]
            Hc,Wc = fc.shape[-2:]
            Hs,Ws = min(Hc,Ht), min(Wc,Wt)
            fc = F.interpolate(fc, size=(Hs,Ws), mode='bilinear', align_corners=False)
            ft = F.interpolate(ft, size=(Hs,Ws), mode='bilinear', align_corners=False)
            cnn_feats.append(fc)
            tr_feats.append(ft)

        cnn_feats = torch.stack(cnn_feats, dim=1)  # [B,T,512,Hs,Ws]
        tr_feats  = torch.stack(tr_feats,  dim=1)  # [B,T,512,Hs,Ws]

        cnn_feats = self.tse_cnn(cnn_feats)
        cnn_feats = self.tla_cnn(cnn_feats)
        tr_feats  = self.tse_tr(tr_feats)
        tr_feats  = self.tla_tr(tr_feats)

        out_feats = []
        for t in range(T):
            A = cnn_feats[:,t]   # [B,512,Hs,Ws]
            B_ = tr_feats[:,t]   # [B,512,Hs,Ws]
            A2,B2 = self.cmgca(A,B_)
            F_t = 0.5*(A2 + B2)
            out_feats.append(F_t)
        out = torch.stack(out_feats, dim=1)   # [B,T,512,Hs,Ws]
        logits = self.mil(out)                # [B,num_classes]
        return logits