import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C] -> LN -> [B, C, H, W]
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int, stride: int, padding: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class MixFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientSelfAttention(nn.Module):
    """
    Efficient self-attention with spatial reduction.
    Input / output shape: [B, C, H, W]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=qkv_bias)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = LayerNorm2d(dim)
        else:
            self.sr = None
            self.norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w

        # q: [B, heads, N, head_dim]
        q = self.q(x).reshape(b, self.num_heads, self.head_dim, n).permute(0, 1, 3, 2)

        kv_in = x
        if self.sr is not None:
            kv_in = self.sr(kv_in)
            kv_in = self.norm(kv_in)

        hk, wk = kv_in.shape[-2:]
        nk = hk * wk

        # kv: [2, B, heads, Nk, head_dim]
        kv = self.kv(kv_in).reshape(b, 2, self.num_heads, self.head_dim, nk)
        kv = kv.permute(1, 0, 2, 4, 3)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_prob: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = EfficientSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

        self.norm2 = LayerNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN(dim=dim, hidden_dim=hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MixTransformerEncoder(nn.Module):
    """
    MiT encoder for SegFormer-B0.
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: List[int] = [32, 64, 160, 256],
        num_heads: List[int] = [1, 2, 5, 8],
        mlp_ratios: List[int] = [4, 4, 4, 4],
        depths: List[int] = [2, 2, 2, 2],
        sr_ratios: List[int] = [8, 4, 2, 1],
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(in_chans, embed_dims[0], patch_size=7, stride=4, padding=3),
            OverlapPatchEmbed(embed_dims[0], embed_dims[1], patch_size=3, stride=2, padding=1),
            OverlapPatchEmbed(embed_dims[1], embed_dims[2], patch_size=3, stride=2, padding=1),
            OverlapPatchEmbed(embed_dims[2], embed_dims[3], patch_size=3, stride=2, padding=1),
        ])

        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(
                    TransformerBlock(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path_prob=dpr[cur + j],
                        sr_ratio=sr_ratios[i],
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

    def forward(self, x):
        outs = []
        for patch_embed, stage in zip(self.patch_embeds, self.stages):
            x = patch_embed(x)
            x = stage(x)
            outs.append(x)
        return outs


class SegFormerHead(nn.Module):
    """
    Practical all-MLP decoder implemented with 1x1 projections.
    """

    def __init__(self, in_channels: List[int], decoder_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, decoder_dim, kernel_size=1) for c in in_channels])

        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]):
        target_size = features[0].shape[-2:]
        projected = []
        for feat, proj in zip(features, self.projections):
            x = proj(feat)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)

        x = torch.cat(projected, dim=1)
        x = self.fuse(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class SegFormerB0(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_chans: int = 3,
        decoder_dim: int = 256,
        drop_path_rate: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MixTransformerEncoder(
            in_chans=in_chans,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
        )
        self.decode_head = SegFormerHead(
            in_channels=[32, 64, 160, 256],
            decoder_dim=decoder_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.encoder(x)
        logits = self.decode_head(features)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


def build_segformer_b0(num_classes: int, in_chans: int = 3, decoder_dim: int = 256, drop_path_rate: float = 0.1):
    return SegFormerB0(
        num_classes=num_classes,
        in_chans=in_chans,
        decoder_dim=decoder_dim,
        drop_path_rate=drop_path_rate,
    )


if __name__ == "__main__":
    model = build_segformer_b0(num_classes=19)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {num_params / 1e6:.2f}M")
