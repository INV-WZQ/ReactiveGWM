"""Wan2.2 VAE (z_dim=48, upsampling_factor=16). Direct port of `WanVideoVAE38`.

Loads `Wan2.2_VAE.pth` cleanly (state_dict keys match diffsynth's layout).
Encoder reduces (T, H, W) → ((T+3)//4, H/16, W/16) by patchify-2 + 3 down stages.
"""
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

CACHE_T = 2


def _is_instance(m, cls):
    return isinstance(m, cls) or (hasattr(m, "module") and isinstance(m.module, cls))


def _count_conv3d(model):
    return sum(1 for m in model.modules() if isinstance(m, CausalConv3d))


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcast = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcast) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        # bf16-safe nearest-neighbor
        return super().forward(x.float()).type_as(x)


def patchify(x, p):
    if p == 1:
        return x
    if x.dim() == 5:
        return rearrange(x, "b c f (h q) (w r) -> b (c r q) f h w", q=p, r=p)
    return rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=p, r=p)


def unpatchify(x, p):
    if p == 1:
        return x
    if x.dim() == 5:
        return rearrange(x, "b (c r q) f h w -> b c f (h q) (w r)", q=p, r=p)
    return rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=p, r=p)


class Resample38(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0], x[:, 1]), 3).reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x, feat_cache, feat_idx


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if _is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h, feat_cache, feat_idx


class AttentionBlock(nn.Module):
    """Causal self-attention with a single head (per-frame)."""
    def __init__(self, dim):
        super().__init__()
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x + identity


class AvgDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        self.out_channels = out_channels
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x):
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        B, C, T, H, W = x.shape
        x = x.view(B, C, T // self.factor_t, self.factor_t,
                   H // self.factor_s, self.factor_s,
                   W // self.factor_s, self.factor_s)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // self.factor_t,
                   H // self.factor_s, W // self.factor_s)
        x = x.view(B, self.out_channels, self.group_size,
                   T // self.factor_t, H // self.factor_s, W // self.factor_s)
        return x.mean(dim=2)


class DupUp3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        self.out_channels = out_channels
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x, first_chunk=False):
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(x.size(0), self.out_channels, self.factor_t,
                   self.factor_s, self.factor_s,
                   x.size(2), x.size(3), x.size(4))
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(x.size(0), self.out_channels,
                   x.size(2) * self.factor_t,
                   x.size(4) * self.factor_s,
                   x.size(6) * self.factor_s)
        if first_chunk:
            x = x[:, :, self.factor_t - 1:, :, :]
        return x


class Down_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False):
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim, out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        ds = []
        for _ in range(mult):
            ds.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            ds.append(Resample38(out_dim, mode=mode))
        self.downsamples = nn.Sequential(*ds)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for module in self.downsamples:
            x, feat_cache, feat_idx = module(x, feat_cache, feat_idx)
        return x + self.avg_shortcut(x_copy), feat_cache, feat_idx


class Up_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_upsample=False, up_flag=False):
        super().__init__()
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim, out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None
        us = []
        for _ in range(mult):
            us.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            us.append(Resample38(out_dim, mode=mode))
        self.upsamples = nn.Sequential(*us)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_main = x.clone()
        for module in self.upsamples:
            x_main, feat_cache, feat_idx = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_short = self.avg_shortcut(x, first_chunk)
            return x_main + x_short, feat_cache, feat_idx
        return x_main, feat_cache, feat_idx


def _conv3d_with_cache(layer, x, feat_cache, feat_idx):
    """Apply a CausalConv3d while updating diffsynth's chunked-inference cache."""
    idx = feat_idx[0]
    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
        cache_x = torch.cat([
            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
            cache_x], dim=2)
    x = layer(x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return x


class Encoder3d_38(nn.Module):
    def __init__(self, dim=160, z_dim=96, dim_mult=(1, 2, 4, 4),
                 num_res_blocks=2, attn_scales=(),
                 temperal_downsample=(False, True, True), dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [1] + list(dim_mult)]
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)
        downs = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down = temperal_downsample[i] if i < len(temperal_downsample) else False
            downs.append(Down_ResidualBlock(
                in_dim=in_dim, out_dim=out_dim, dropout=dropout,
                mult=num_res_blocks, temperal_downsample=t_down,
                down_flag=i != len(dim_mult) - 1,
            ))
        self.downsamples = nn.Sequential(*downs)
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            x = _conv3d_with_cache(self.conv1, x, feat_cache, feat_idx)
        else:
            x = self.conv1(x)
        for layer in self.downsamples:
            if feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                x = _conv3d_with_cache(layer, x, feat_cache, feat_idx)
            else:
                x = layer(x)
        return x, feat_cache, feat_idx


class Decoder3d_38(nn.Module):
    def __init__(self, dim=256, z_dim=48, dim_mult=(1, 2, 4, 4),
                 num_res_blocks=2, attn_scales=(),
                 temperal_upsample=(True, True, False), dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + list(dim_mult[::-1])]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )
        ups = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up = temperal_upsample[i] if i < len(temperal_upsample) else False
            ups.append(Up_ResidualBlock(
                in_dim=in_dim, out_dim=out_dim, dropout=dropout,
                mult=num_res_blocks + 1, temperal_upsample=t_up,
                up_flag=i != len(dim_mult) - 1,
            ))
        self.upsamples = nn.Sequential(*ups)
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        if feat_cache is not None:
            x = _conv3d_with_cache(self.conv1, x, feat_cache, feat_idx)
        else:
            x = self.conv1(x)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.upsamples:
            if feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                x = _conv3d_with_cache(layer, x, feat_cache, feat_idx)
            else:
                x = layer(x)
        return x, feat_cache, feat_idx


class _VideoVAE38(nn.Module):
    """Inner Wan2.2 VAE: chunked encode/decode with cached causal convs."""
    def __init__(self, dim=160, z_dim=48, dec_dim=256, dim_mult=(1, 2, 4, 4),
                 num_res_blocks=2, temperal_downsample=(False, True, True), dropout=0.0):
        super().__init__()
        self.z_dim = z_dim
        self.temperal_downsample = list(temperal_downsample)
        self.temperal_upsample = self.temperal_downsample[::-1]
        self.encoder = Encoder3d_38(dim, z_dim * 2, dim_mult, num_res_blocks,
                                    [], self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d_38(dec_dim, z_dim, dim_mult, num_res_blocks,
                                    [], self.temperal_upsample, dropout)

    def _clear(self):
        self._conv_num = _count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = _count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def encode(self, x, scale):
        self._clear()
        x = patchify(x, 2)
        T = x.shape[2]
        iters = 1 + (T - 1) // 4
        out = None
        for i in range(iters):
            self._enc_conv_idx = [0]
            chunk = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1): 1 + 4 * i]
            o, self._enc_feat_map, self._enc_conv_idx = self.encoder(
                chunk, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            out = o if out is None else torch.cat([out, o], 2)
        mu, _ = self.conv1(out).chunk(2, dim=1)
        mean, inv_std = scale
        if isinstance(mean, torch.Tensor):
            mean = mean.to(dtype=mu.dtype, device=mu.device)
            inv_std = inv_std.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - mean.view(1, self.z_dim, 1, 1, 1)) * inv_std.view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - mean) * inv_std
        self._clear()
        return mu

    def decode(self, z, scale):
        self._clear()
        mean, inv_std = scale
        if isinstance(mean, torch.Tensor):
            mean = mean.to(dtype=z.dtype, device=z.device)
            inv_std = inv_std.to(dtype=z.dtype, device=z.device)
            z = z / inv_std.view(1, self.z_dim, 1, 1, 1) + mean.view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / inv_std + mean
        x = self.conv2(z)
        out = None
        for i in range(z.shape[2]):
            self._conv_idx = [0]
            o, self._feat_map, self._conv_idx = self.decoder(
                x[:, :, i:i + 1], feat_cache=self._feat_map, feat_idx=self._conv_idx,
                first_chunk=(i == 0))
            out = o if out is None else torch.cat([out, o], 2)
        out = unpatchify(out, 2)
        self._clear()
        return out


# Wan2.2 VAE normalization stats (mean/std per latent channel)
_WAN22_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]
_WAN22_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]


class WanVideoVAE38(nn.Module):
    """Wan2.2-TI2V-5B VAE: 16x spatial + 4x temporal compression, z_dim=48."""
    def __init__(self, z_dim=48, dim=160):
        super().__init__()
        self.z_dim = z_dim
        self.upsampling_factor = 16
        self.mean = torch.tensor(_WAN22_MEAN)
        self.std = torch.tensor(_WAN22_STD)
        self.scale = [self.mean, 1.0 / self.std]
        self.model = _VideoVAE38(z_dim=z_dim, dim=dim).eval().requires_grad_(False)

    @staticmethod
    def _build_1d_mask(length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def _build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self._build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self._build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])
        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)
        return rearrange(torch.stack([h, w]).min(dim=0).values, "H W -> 1 1 1 H W")

    def _tiled_decode(self, hidden, device, tile_size, tile_stride):
        _, _, T, H, W = hidden.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                tasks.append((h, h + size_h, w, w + size_w))
        out_T = T * 4 - 3
        weight = torch.zeros((1, 1, out_T, H * 16, W * 16), dtype=hidden.dtype, device="cpu")
        values = torch.zeros((1, 3, out_T, H * 16, W * 16), dtype=hidden.dtype, device="cpu")
        for h, h_, w, w_ in tqdm(tasks, desc="VAE decoding"):
            hb = hidden[:, :, :, h:h_, w:w_].to(device)
            hb = self.model.decode(hb, self.scale).to("cpu")
            mask = self._build_mask(hb,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=((size_h - stride_h) * 16, (size_w - stride_w) * 16),
            ).to(dtype=hidden.dtype, device="cpu")
            th, tw = h * 16, w * 16
            values[:, :, :, th:th + hb.shape[3], tw:tw + hb.shape[4]] += hb * mask
            weight[:, :, :, th:th + hb.shape[3], tw:tw + hb.shape[4]] += mask
        return (values / weight).clamp_(-1, 1)

    def _tiled_encode(self, video, device, tile_size, tile_stride):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                tasks.append((h, h + size_h, w, w + size_w))
        out_T = (T + 3) // 4
        weight = torch.zeros((1, 1, out_T, H // 16, W // 16), dtype=video.dtype, device="cpu")
        values = torch.zeros((1, self.z_dim, out_T, H // 16, W // 16), dtype=video.dtype, device="cpu")
        for h, h_, w, w_ in tqdm(tasks, desc="VAE encoding"):
            hb = video[:, :, :, h:h_, w:w_].to(device)
            hb = self.model.encode(hb, self.scale).to("cpu")
            mask = self._build_mask(hb,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=((size_h - stride_h) // 16, (size_w - stride_w) // 16),
            ).to(dtype=video.dtype, device="cpu")
            th, tw = h // 16, w // 16
            values[:, :, :, th:th + hb.shape[3], tw:tw + hb.shape[4]] += hb * mask
            weight[:, :, :, th:th + hb.shape[3], tw:tw + hb.shape[4]] += mask
        return values / weight

    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        videos = [v.to("cpu") for v in videos]
        outs = []
        for v in videos:
            v = v.unsqueeze(0)
            if tiled:
                ts = (tile_size[0] * 16, tile_size[1] * 16)
                td = (tile_stride[0] * 16, tile_stride[1] * 16)
                z = self._tiled_encode(v, device, ts, td)
            else:
                z = self.model.encode(v.to(device), self.scale)
            outs.append(z.squeeze(0))
        return torch.stack(outs)

    def decode(self, hiddens, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        hiddens = [h.to("cpu") for h in hiddens]
        outs = []
        for h in hiddens:
            h = h.unsqueeze(0)
            if tiled:
                v = self._tiled_decode(h, device, tile_size, tile_stride)
            else:
                v = self.model.decode(h.to(device), self.scale).clamp_(-1, 1)
            outs.append(v.squeeze(0))
        return torch.stack(outs)
