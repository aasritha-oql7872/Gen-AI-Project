"""
UNet for noise prediction.
Takes: noisy image (B,3,64,64), timestep (B,), attributes (B,10)
Returns: predicted noise (B,3,64,64)
"""

import math
import torch
import torch.nn as nn
import config


# ---- Time Embedding ----

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding → MLP → vector."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        # t: (B,) integer timesteps
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb = t[:, None].float() * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        return self.mlp(emb)


# ---- Attribute Embedding ----

class AttrEmbedding(nn.Module):
    """Multi-label attribute vector → embedding vector."""

    def __init__(self, num_attrs, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_attrs, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, attrs):
        # attrs: (B, num_attrs) float vector of 0s and 1s
        return self.mlp(attrs)


# ---- ResNet Block ----

class ResBlock(nn.Module):
    """Conv → Norm → SiLU → Conv → Norm → SiLU + skip connection.
    Time and attribute embeddings are added after first norm."""

    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        # Project embedding to match channel dim
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch),
        )

        # Skip connection if channels change
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        # emb: (B, emb_dim) — combined time + attribute embedding
        h = self.act(self.norm1(self.conv1(x)))

        # Add embedding (broadcast over spatial dims)
        emb_out = self.emb_proj(emb)[:, :, None, None]
        h = h + emb_out

        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


# ---- Attention Block ----

class SelfAttention(nn.Module):
    """Simple self-attention on spatial features."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale  # (B, HW, HW)
        attn = attn.softmax(dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, HW)
        out = out.reshape(B, C, H, W)
        return x + self.out(out)


# ---- Downsample / Upsample ----

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---- Full UNet ----

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=config.CHANNELS,
        base_channels=config.BASE_CHANNELS,
        channel_mults=config.CHANNEL_MULTS,
        num_attrs=config.NUM_CLASSES,
        time_dim=config.TIME_DIM,
        attr_dim=config.ATTR_DIM,
    ):
        super().__init__()

        emb_dim = time_dim  # combined embedding dimension

        # Embeddings
        self.time_emb = TimeEmbedding(time_dim)
        self.attr_emb = AttrEmbedding(num_attrs, attr_dim)
        self.comb_proj = nn.Linear(time_dim + attr_dim, emb_dim)

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Build channel list: e.g., [64, 128, 256, 512]
        channels = [base_channels * m for m in channel_mults]

        # ---- Encoder (downsampling) ----
        self.downs = nn.ModuleList()
        ch_in = base_channels
        self.skip_channels = []  # track for decoder

        for i, ch_out in enumerate(channels):
            self.downs.append(nn.ModuleList([
                ResBlock(ch_in, ch_out, emb_dim),
                ResBlock(ch_out, ch_out, emb_dim),
                SelfAttention(ch_out) if i >= 2 else nn.Identity(),  # attention at lower resolutions
                Downsample(ch_out) if i < len(channels) - 1 else nn.Identity(),
            ]))
            self.skip_channels.append(ch_out)
            ch_in = ch_out

        # ---- Middle ----
        mid_ch = channels[-1]
        self.mid = nn.ModuleList([
            ResBlock(mid_ch, mid_ch, emb_dim),
            SelfAttention(mid_ch),
            ResBlock(mid_ch, mid_ch, emb_dim),
        ])

        # ---- Decoder (upsampling) ----
        self.ups = nn.ModuleList()
        for i, ch_out in enumerate(reversed(channels)):
            ch_skip = self.skip_channels[len(channels) - 1 - i]
            ch_in_up = ch_in + ch_skip  # concatenated skip connection
            self.ups.append(nn.ModuleList([
                ResBlock(ch_in_up, ch_out, emb_dim),
                ResBlock(ch_out, ch_out, emb_dim),
                SelfAttention(ch_out) if (len(channels) - 1 - i) >= 2 else nn.Identity(),
                Upsample(ch_out) if i < len(channels) - 1 else nn.Identity(),
            ]))
            ch_in = ch_out

        # Final output
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 1),
        )

    def forward(self, x, t, attrs):
        """
        x: (B, 3, 64, 64) noisy image
        t: (B,) timestep
        attrs: (B, 10) attribute vector
        """
        # Compute embeddings
        t_emb = self.time_emb(t)                         # (B, time_dim)
        a_emb = self.attr_emb(attrs)                     # (B, attr_dim)
        emb = self.comb_proj(torch.cat([t_emb, a_emb], dim=-1))  # (B, emb_dim)

        # Initial conv
        x = self.init_conv(x)

        # Encoder
        skips = []
        for res1, res2, attn, down in self.downs:
            x = res1(x, emb)
            x = res2(x, emb)
            x = attn(x)
            skips.append(x)
            x = down(x)

        # Middle
        x = self.mid[0](x, emb)
        x = self.mid[1](x)
        x = self.mid[2](x, emb)

        # Decoder
        for res1, res2, attn, up in self.ups:
            x = torch.cat([x, skips.pop()], dim=1)  # skip connection
            x = res1(x, emb)
            x = res2(x, emb)
            x = attn(x)
            x = up(x)

        return self.final(x)


if __name__ == "__main__":
    # Quick test
    model = UNet()
    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    attrs = torch.zeros(2, 10)
    out = model(x, t, attrs)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
