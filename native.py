"""Native in-process FireRedTTS3 runtime.

Ports the upstream FireRedTTS3 model code (RedAE, PatchEncoder, DiT flow head,
CAM++ speaker encoder, Qwen3 backbones) to a ComfyUI-friendly form:

- no einops, no HF from_pretrained; weights load straight from safetensors with
  the original key names
- Linear/Embedding/Conv modules are swapped for ComfyUI castable variants so
  AIMDO DynamicVRAM can page weights
- dtype policy is explicit: backbone LLM + RedAE encoder match the official
  bf16-autocast numerics in bf16 mode, flow/decoder weights stay fp32
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import torchaudio
from safetensors import safe_open
from torch import nn

try:
    from comfy.ops import cast_bias_weight, uncast_bias_weight
except Exception:
    def cast_bias_weight(model, input=None, **kwargs):
        return model.weight, getattr(model, "bias", None), None

    def uncast_bias_weight(model, weight, bias, offload_stream):
        return None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger("FireRedTTS3")

SAMPLE_RATE = 24_000
REDAE_SCALE = 0.4
TEXT_EOT_ID = 151677
AUDIO_SOS_ID = 151669

QWEN3_1_7B_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 40960,
    "max_window_layers": 28,
    "model_type": "qwen3",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


def fix_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def _check_interrupted() -> None:
    try:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _autocast_bf16(device: torch.device, enabled: bool):
    if enabled and torch.device(device).type in ("cuda", "xpu"):
        return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


# --------------------------------------------------------------------------- #
# Rotary embedding (upstream llm/rotary_embedding.py without einops)
# --------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, interpolation_factor=1.0, base=10000, base_rescale_factor=1.0):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))
        self._rope_dim = dim
        self._rope_base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor
        self._seq_len_cache = {}

    def rope_init(self):
        self.inv_freq = 1.0 / (self._rope_base ** (torch.arange(0, self._rope_dim, 2).float() / self._rope_dim))
        self._seq_len_cache = {}

    def forward_from_seq_len(self, seq_len):
        key = (int(seq_len), str(self.inv_freq.device), self.inv_freq.dtype)
        cached = self._seq_len_cache.get(key)
        if cached is None:
            t = torch.arange(seq_len, device=self.inv_freq.device)
            cached = self.forward(t)
            self._seq_len_cache[key] = cached
        return cached

    @torch.autocast("cuda", enabled=False)
    def forward(self, t, offset=0):
        if t.ndim == 1:
            t = t.unsqueeze(0)
        freqs = t.type_as(self.inv_freq).unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        freqs = freqs / self.interpolation_factor
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(-2)
        return freqs, 1.0


def _rotate_half(x):
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


@torch.autocast("cuda", enabled=False)
def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype
    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if isinstance(scale, torch.Tensor) else scale
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (_rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t, t_unrotated), dim=-1).type(orig_dtype)


# --------------------------------------------------------------------------- #
# DiT building blocks (upstream llm/modules.py + llm/dit.py)
# --------------------------------------------------------------------------- #
class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep).to(timestep.dtype)
        return self.time_mlp(time_hidden)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim), nn.Dropout(dropout)])

    def forward(self, x, mask=None, rope=None):
        batch_size = x.shape[0]
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)
        head_dim = key.shape[-1] // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        if rope is not None:
            freqs, xpos_scale = rope
            q_scale, k_scale = (xpos_scale, xpos_scale ** -1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_scale)
            key = apply_rotary_pos_emb(key, freqs, k_scale)
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        x = x.to(query.dtype)
        x = self.to_out[1](self.to_out[0](x))
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x


class PatchDiTBlock(nn.Module):
    """Upstream modules.py DiTBlock: attention + FFN (used by PatchEncoder)."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, **kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(dim=hidden_size, heads=num_heads, dim_head=hidden_size // num_heads, dropout=dropout)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate="tanh")

    def forward(self, x, mask, rope):
        x = x + self.attn(self.norm1(x), mask=mask, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """Upstream modules.py FinalLayer (used by PatchEncoder out_proj)."""

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        return self.linear(self.norm_final(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.Mish(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        x = self.block(x.transpose(1, 2)).transpose(1, 2)
        if mask is not None:
            x = x * mask
        return x


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FlowDiTBlock(nn.Module):
    """Upstream dit.py DiTBlock: attention + conv + FFN with 9-way AdaLN."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, **kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(dim=hidden_size, heads=num_heads, dim_head=hidden_size // num_heads, dropout=dropout)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.conv = ConvBlock(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3)
        self.norm3 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate="tanh")
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size))

    def forward(self, x, c, mask, rope, ada=None):
        if ada is None:
            ada = self.adaLN_modulation(c).chunk(9, dim=-1)
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp,
         shift_conv, scale_conv, gate_conv) = ada
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa), mask=mask, rope=rope)
        x = x + gate_conv * self.conv(_modulate(self.norm2(x), shift_conv, scale_conv), mask=mask)
        x = x + gate_mlp * self.mlp(_modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class AdaLNFinalLayer(nn.Module):
    """Upstream dit.py FinalLayer (used by the flow DiT)."""

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x, c, ada=None):
        if ada is None:
            ada = self.adaLN_modulation(c).chunk(2, dim=-1)
        shift, scale = ada
        return self.linear(_modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_ratio=4.0, depth=28, num_heads=8, hidden_size=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_proj = nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = nn.ModuleList([
            FlowDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=0.0) for _ in range(depth)
        ])
        self.final_layer = AdaLNFinalLayer(hidden_size, self.out_channels)

    def forward(self, x, t):
        t = self.t_embedder(t.view(-1)).unsqueeze(1)
        x = self.in_proj(x)
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])
        for block in self.blocks:
            x = block(x, t, mask=None, rope=rope)
        return self.final_layer(x, t)


class PatchEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, patch_size=4, hidden_size=1024, mlp_ratio=3, depth=8, num_heads=8):
        super().__init__()
        self.in_dim = in_dim
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = nn.ModuleList([
            PatchDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.in_proj = nn.Linear(in_dim, hidden_size) if in_dim != hidden_size else nn.Identity()
        self.out_proj = FinalLayer(hidden_size, out_dim)

    def forward(self, inputs_embeds):
        if inputs_embeds.shape[1] % self.patch_size != 0:
            raise ValueError(f"latent length {inputs_embeds.shape[1]} is not a multiple of patch_size {self.patch_size}")
        inputs_embeds = self.in_proj(inputs_embeds)
        hidden_states = inputs_embeds.reshape(-1, self.patch_size, self.hidden_size)
        cls_tok = self.cls_tok.expand(hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat([cls_tok, hidden_states], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(hidden_states.shape[1])
        for block in self.blocks:
            hidden_states = block(hidden_states, None, rope)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states[:, 0].unsqueeze(0)


# --------------------------------------------------------------------------- #
# RedAE continuous audio autoencoder (upstream redae/redae.py)
# --------------------------------------------------------------------------- #
def _qwen3_config(**kwargs) -> "Any":
    from transformers import Qwen3Config

    return Qwen3Config(**kwargs)


def _qwen3_model(config) -> "Any":
    from transformers import Qwen3Model

    return Qwen3Model(config)


class Qwen3ClsDownsample(nn.Module):
    def __init__(self, in_dim, out_dim, downsample_rate=2, hidden_size=896, intermediate_size=896 * 4,
                 num_hidden_layers=4, max_position_embeddings=32768, num_attention_heads=14,
                 num_key_value_heads=2, attn_implementation="sdpa"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.downsample_rate = downsample_rate
        self.qwen3_config = _qwen3_config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attn_implementation=attn_implementation,
        )
        self.qwen3 = _qwen3_model(self.qwen3_config)
        self.cls_tok = nn.Parameter(torch.ones(1, 1, hidden_size))
        self.in_proj = nn.Linear(in_dim, hidden_size) if in_dim != hidden_size else nn.Identity()
        self.out_proj = nn.Linear(hidden_size, out_dim) if out_dim != hidden_size else nn.Identity()

    def forward(self, xs):
        if xs.shape[1] % self.downsample_rate != 0:
            raise ValueError(f"latent length {xs.shape[1]} not divisible by {self.downsample_rate}")
        b = xs.shape[0]
        xs = xs.reshape(-1, self.downsample_rate, self.in_dim)
        xs = self.in_proj(xs)
        xs = torch.cat([xs, self.cls_tok.expand(xs.shape[0], -1, -1)], dim=1)
        ys = []
        for xs_chunk in torch.split(xs, 32768, dim=0):
            outs_chunk = self.qwen3(inputs_embeds=xs_chunk)
            ys.append(outs_chunk.last_hidden_state[:, -1])
        ys = torch.cat(ys, dim=0)
        return self.out_proj(ys).reshape(b, -1, self.out_dim)


class RedAEAudioEncoder(nn.Module):
    def __init__(self, out_dim=1024, audio_patch_size=480, audio_sample_rate=24000, hidden_size=896,
                 intermediate_size=896 * 4, num_hidden_layers=24, max_position_embeddings=32768,
                 max_window_layers=0, num_attention_heads=14, num_key_value_heads=2, sliding_window=64,
                 use_sliding_window=True, extra_downsample_rate=2, downsample_num_hidden_layers=4,
                 attn_implementation="sdpa"):
        super().__init__()
        self.audio_patch_size = audio_patch_size
        self.audio_sample_rate = audio_sample_rate
        self.extra_downsample_rate = extra_downsample_rate
        self.out_dim = out_dim
        self.in_proj = nn.Sequential(nn.Linear(self.audio_patch_size, hidden_size), nn.Linear(hidden_size, hidden_size))
        self.qwen3_config = _qwen3_config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            max_window_layers=max_window_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            sliding_window=sliding_window,
            use_sliding_window=use_sliding_window,
            attn_implementation=attn_implementation,
        )
        self.qwen3 = _qwen3_model(self.qwen3_config)
        if self.extra_downsample_rate > 1:
            self.downsample = Qwen3ClsDownsample(
                in_dim=hidden_size,
                out_dim=hidden_size,
                downsample_rate=extra_downsample_rate,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=downsample_num_hidden_layers,
                max_position_embeddings=max_position_embeddings,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                attn_implementation=attn_implementation,
            )
        self.out_proj = nn.Linear(hidden_size, out_dim)

    @property
    def downsample_rate(self):
        return int(self.audio_patch_size * self.extra_downsample_rate)

    def forward(self, audio):
        if audio.shape[1] % self.downsample_rate != 0:
            raise ValueError(f"audio length {audio.shape[1]} not divisible by {self.downsample_rate}")
        xs = audio.unfold(1, self.audio_patch_size, self.audio_patch_size)
        xs = self.in_proj(xs)
        xs = self.qwen3(inputs_embeds=xs, attention_mask=None).last_hidden_state
        if self.extra_downsample_rate > 1:
            xs = self.downsample(xs)
        return self.out_proj(xs)


class ISTFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, padding="same"):
        super().__init__()
        assert padding in ("center", "same")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, spec):
        if self.padding == "center":
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        pad = (self.win_length - self.hop_length) // 2
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward") * self.window[None, :, None]
        output_size = (spec.shape[2] - 1) * self.hop_length + self.win_length
        y = F.fold(ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length))[:, 0, 0, pad:-pad]
        window_sq = self.window.square().expand(1, spec.shape[2], -1).transpose(1, 2)
        window_envelope = F.fold(window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length)).squeeze()[pad:-pad]
        return y / window_envelope


class ISTFTHead(nn.Module):
    def __init__(self, dim, n_fft, hop_length, padding="same"):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x):
        x_pred = self.out(x).transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        spec = mag * (torch.cos(p) + 1j * torch.sin(p))
        return self.istft(spec)


class RedAEAudioDecoder(nn.Module):
    def __init__(self, in_dim=64, upsample_rate=2, audio_patch_size=480, audio_sample_rate=24000,
                 hidden_size=896, intermediate_size=896 * 4, num_hidden_layers=18, max_position_embeddings=32768,
                 max_window_layers=0, num_attention_heads=14, num_key_value_heads=2, sliding_window=64,
                 use_sliding_window=True, attn_implementation="sdpa"):
        super().__init__()
        self.upsample_rate = upsample_rate
        self.audio_patch_size = audio_patch_size
        self.audio_sample_rate = audio_sample_rate
        self.in_proj = nn.Linear(in_dim, upsample_rate * hidden_size)
        self.qwen3_config = _qwen3_config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            max_window_layers=max_window_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            sliding_window=sliding_window,
            use_sliding_window=use_sliding_window,
            attn_implementation=attn_implementation,
        )
        self.qwen3 = _qwen3_model(self.qwen3_config)
        self.istft_head = ISTFTHead(dim=hidden_size, n_fft=audio_patch_size * 4, hop_length=audio_patch_size, padding="same")

    def forward(self, xs):
        xs = self.in_proj(xs)
        xs = xs.reshape(xs.shape[0], -1, self.qwen3_config.hidden_size)
        xs = self.qwen3(inputs_embeds=xs, attention_mask=None).last_hidden_state
        return self.istft_head(xs)


class RedAE(nn.Module):
    """RedAE codec wrapper matching upstream RedAE encode/decode semantics."""

    def __init__(self, config: dict[str, Any], attn_implementation: str = "sdpa"):
        super().__init__()
        self.encoder = RedAEAudioEncoder(
            out_dim=config.get("bottleneck_dim", 64),
            audio_patch_size=config.get("audio_patch_size", 480),
            audio_sample_rate=config.get("audio_sample_rate", 24000),
            hidden_size=config.get("enc_hidden_size", 896),
            intermediate_size=config.get("enc_intermediate_size", 3584),
            num_hidden_layers=config.get("enc_num_hidden_layers", 18),
            max_position_embeddings=config.get("enc_max_position_embeddings", 32768),
            max_window_layers=config.get("enc_max_window_layers", 0),
            num_attention_heads=config.get("enc_num_attention_heads", 14),
            num_key_value_heads=config.get("enc_num_key_value_heads", 2),
            sliding_window=config.get("enc_sliding_window", 64),
            use_sliding_window=config.get("enc_use_sliding_window", True),
            extra_downsample_rate=config.get("enc_extra_downsample_rate", 2),
            downsample_num_hidden_layers=config.get("enc_downsample_num_hidden_layers", 4),
            attn_implementation=attn_implementation,
        )
        self.decoder = RedAEAudioDecoder(
            in_dim=config.get("bottleneck_dim", 64),
            upsample_rate=config.get("enc_extra_downsample_rate", 2),
            audio_patch_size=config.get("audio_patch_size", 480),
            audio_sample_rate=config.get("audio_sample_rate", 24000),
            hidden_size=config.get("dec_hidden_size", 896),
            intermediate_size=config.get("dec_intermediate_size", 3584),
            num_hidden_layers=config.get("dec_num_hidden_layers", 18),
            max_position_embeddings=config.get("dec_max_position_embeddings", 32768),
            max_window_layers=config.get("dec_max_window_layers", 0),
            num_attention_heads=config.get("dec_num_attention_heads", 14),
            num_key_value_heads=config.get("dec_num_key_value_heads", 2),
            sliding_window=config.get("dec_sliding_window", 64),
            use_sliding_window=config.get("dec_use_sliding_window", True),
            # The decoder always runs fp32 (same as upstream); FlashAttention only
            # accepts fp16/bf16, so the decoder keeps sdpa regardless of choice.
            attn_implementation="sdpa",
        )
        self.autocast_bf16 = True

    @property
    def downsample_rate(self):
        return self.encoder.downsample_rate

    @property
    def sample_rate(self):
        return self.decoder.audio_sample_rate

    @staticmethod
    def pad_to_multiple_of(audio, multiple_of):
        target = math.ceil(audio.shape[-1] / multiple_of) * multiple_of
        pad_len = target - audio.shape[-1]
        if pad_len > 0:
            audio = F.pad(audio, (pad_len, 0))  # left pad, same as upstream
        return audio

    @torch.no_grad()
    def encode(self, audio, audio_sr):
        device = self.out_device
        with _autocast_bf16(device, self.autocast_bf16):
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            audio = audio[:1]
            audio = torchaudio.functional.resample(audio, audio_sr, self.sample_rate)
            audio = self.pad_to_multiple_of(audio, self.downsample_rate)
            self._enforce_input_limit(audio)
            return self.encoder(audio.to(device))

    @torch.no_grad()
    def decode(self, latents):
        audio = self.decoder(latents.to(self.out_device))
        return audio, self.sample_rate

    @property
    def out_device(self):
        return torch.device(getattr(self, "_firered_runtime_device", next(self.parameters()).device))

    @property
    def max_input_seconds(self) -> float:
        return self.encoder.qwen3_config.max_position_embeddings / (
            self.sample_rate / self.encoder.audio_patch_size
        )

    def _enforce_input_limit(self, audio: torch.Tensor) -> None:
        frames = audio.shape[-1] // self.encoder.audio_patch_size
        max_positions = self.encoder.qwen3_config.max_position_embeddings
        if frames > max_positions:
            raise RuntimeError(
                f"Input audio is {audio.shape[-1] / self.sample_rate:.0f}s, but the RedAE encoder supports at most "
                f"{self.max_input_seconds:.0f}s ({max_positions} positions at "
                f"{self.sample_rate // self.encoder.audio_patch_size} Hz). Trim the clip; a clean 5-20s "
                f"reference clones best."
            )
        seconds = audio.shape[-1] / self.sample_rate
        if seconds > 120:
            logger.warning(
                "Input audio is %.0fs; long references raise VRAM use and can reduce cloning quality (5-20s is ideal).",
                seconds,
            )


# --------------------------------------------------------------------------- #
# CAM++ speaker encoder (upstream campp/, Apache-2.0, from 3D-Speaker)
# --------------------------------------------------------------------------- #
def _get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module ({name}).")
    return nonlinear


class StatsPool(nn.Module):
    def forward(self, x):
        stats = torch.cat([x.mean(dim=-1), x.std(dim=-1, unbiased=True)], dim=-1)
        return stats


class TDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bias=False, config_str="batchnorm-relu"):
        super().__init__()
        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, dilation=dilation, bias=bias)
        self.nonlinear = _get_nonlinear(config_str, out_channels)

    def forward(self, x):
        return self.nonlinear(self.linear(x))


class CAMLayer(nn.Module):
    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2):
        super().__init__()
        self.linear_local = nn.Conv1d(bn_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def seg_pooling(self, x, seg_len=100):
        seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        return seg[..., : x.shape[-1]]

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        return y * self.sigmoid(self.linear2(context))


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bn_channels, kernel_size, stride=1, dilation=1,
                 bias=False, config_str="batchnorm-relu"):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.nonlinear1 = _get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = _get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.linear1(self.nonlinear1(x))
        return self.cam_layer(self.nonlinear2(x))


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self, num_layers, in_channels, out_channels, bn_channels, kernel_size, stride=1,
                 dilation=1, bias=False, config_str="batchnorm-relu"):
        super().__init__()
        for i in range(num_layers):
            self.add_module(
                "tdnnd%d" % (i + 1),
                CAMDenseTDNNLayer(in_channels=in_channels + i * out_channels, out_channels=out_channels,
                                  bn_channels=bn_channels, kernel_size=kernel_size, stride=stride,
                                  dilation=dilation, bias=bias, config_str=config_str),
            )

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"):
        super().__init__()
        self.nonlinear = _get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        return self.linear(self.nonlinear(x))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = _get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if x.dim() == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        return self.nonlinear(x)


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class FCM(nn.Module):
    def __init__(self, block=BasicResBlock, num_blocks=(2, 2), m_channels=32, feat_dim=80):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)
        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        shape = out.shape
        return out.reshape(shape[0], shape[1] * shape[2], shape[3])


class CAMPPlus(nn.Module):
    def __init__(self, feat_dim=80, embedding_size=512, growth_rate=32, bn_size=4, init_channels=128,
                 config_str="batchnorm-relu"):
        super().__init__()
        from collections import OrderedDict

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.xvector = nn.Sequential(OrderedDict([
            ("tdnn", TDNNLayer(channels, init_channels, 5, stride=2, dilation=1, padding=-1, config_str=config_str)),
        ]))
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            self.xvector.add_module("block%d" % (i + 1), CAMDenseTDNNBlock(
                num_layers=num_layers, in_channels=channels, out_channels=growth_rate,
                bn_channels=bn_size * growth_rate, kernel_size=kernel_size, dilation=dilation,
                config_str=config_str))
            channels = channels + num_layers * growth_rate
            self.xvector.add_module("transit%d" % (i + 1), TransitLayer(channels, channels // 2, bias=False, config_str=config_str))
            channels //= 2
        self.xvector.add_module("out_nonlinear", _get_nonlinear(config_str, channels))
        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module("dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_"))

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        return self.xvector(self.head(x))


class CamppEmbedding(nn.Module):
    """Speaker embedding extractor; wraps CAMPPlus and kaldi fbank features."""

    def __init__(self):
        super().__init__()
        self.model = CAMPPlus(feat_dim=80, embedding_size=512)

    def forward(self, audio, audio_sr):
        import torchaudio.compliance.kaldi as kaldi

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        audio = audio[:1]
        if audio_sr != 16000:
            audio = torchaudio.functional.resample(audio, audio_sr, 16000)
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = (feat - feat.mean(dim=0, keepdim=True)).unsqueeze(0)
        device = torch.device(getattr(self, "_firered_runtime_device", next(self.parameters()).device))
        with torch.no_grad():
            return self.model(feat.to(device)).cpu()


# --------------------------------------------------------------------------- #
# TTS cores (upstream llm/fireredtts3_base.py + fireredtts3_instruct.py)
# --------------------------------------------------------------------------- #
# Hoisted flow-step constants. Timestep embeds, per-block AdaLN modulations and
# rotary frequencies are identical for every patch of a generation; computing
# them once and slicing per step removes ~500 kernel launches per patch.
# --------------------------------------------------------------------------- #
def _dit_flow_constants(core, t_span):
    steps = t_span[:-1].reshape(-1)
    t_embs = core.dit.t_embedder(steps)
    adas = torch.stack([block.adaLN_modulation(t_embs) for block in core.dit.blocks])
    final_ada = core.dit.final_layer.adaLN_modulation(t_embs)
    rope = core.dit.rotary_embed.forward_from_seq_len(core.history_length + core.patch_size)
    return adas, final_ada, rope


def _dit_step_impl(core, x_in, adas_step, final_step, rope):
    x = core.dit.in_proj(x_in)
    for index, block in enumerate(core.dit.blocks):
        x = block(x, None, None, rope, ada=adas_step[index].chunk(9, dim=-1))
    return core.dit.final_layer(x, None, ada=final_step.chunk(2, dim=-1))


def _backbone_config(attn_implementation: str):
    from transformers import Qwen3Config

    cfg = dict(QWEN3_1_7B_CONFIG)
    cfg.pop("attn_implementation", None)
    config = Qwen3Config.from_dict(cfg)
    # from_dict drops attn_implementation; the setter is what reaches the
    # runtime attention dispatch (verified: kernels fire only via this path).
    config._attn_implementation = attn_implementation
    return config


class FireRedTTS3BaseCore(nn.Module):
    def __init__(self, config: dict[str, Any], attn_implementation: str = "sdpa"):
        super().__init__()
        from transformers import Qwen3Model

        self.backbone_llm_config = _backbone_config(attn_implementation)
        self.backbone_llm = Qwen3Model(self.backbone_llm_config)
        hidden = self.backbone_llm_config.hidden_size
        redae_dim = config.get("redae_dim", 64)
        spk_in_dim = config.get("spk_in_dim", 512)
        self.spk_proj_llm = nn.Linear(spk_in_dim, hidden)
        self.spk_proj_dit = nn.Linear(spk_in_dim, spk_in_dim)
        self.patch_encoder = PatchEncoder(
            in_dim=redae_dim,
            out_dim=hidden,
            patch_size=config.get("patch_size", 4),
            hidden_size=config.get("patch_encoder_hidden_size", 1024),
            mlp_ratio=config.get("patch_encoder_mlp_ratio", 4),
            depth=config.get("patch_encoder_depth", 8),
            num_heads=config.get("patch_encoder_num_heads", 16),
        )
        dit_hidden = config.get("dit_hidden_size", 1024)
        self.dit_head = nn.Linear(hidden, dit_hidden)
        self.dit = DiT(
            in_channels=redae_dim + spk_in_dim + dit_hidden,
            out_channels=redae_dim,
            mlp_ratio=config.get("dit_mlp_ratio", 3),
            depth=config.get("dit_depth", 11),
            num_heads=config.get("dit_num_heads", 16),
            hidden_size=dit_hidden,
        )
        self.stop_head = nn.Linear(hidden, 1)
        self.redae_dim = redae_dim
        self.patch_size = self.patch_encoder.patch_size
        self.history_patches = config.get("num_history_patches", 2)
        self.history_length = self.history_patches * self.patch_size
        self.autocast_bf16 = True

    def _backbone_one_step(self, input_embeds, cache=None):
        device = input_embeds.device
        with _autocast_bf16(device, self.autocast_bf16):
            outs = self.backbone_llm(inputs_embeds=input_embeds, use_cache=True, past_key_values=cache)
        # .float() matches upstream numerics (fp32 RMSNorm promotion under autocast)
        return outs.last_hidden_state.float(), outs.past_key_values

    def _flow_constants(self, t_span):
        return _dit_flow_constants(self, t_span)

    def _dit_step(self, x_in, adas_step, final_step, rope):
        return _dit_step_impl(self, x_in, adas_step, final_step, rope)

    def _flow_one_step(self, hist_latents, backbone_cond, spk_cond, t_span, inference_cfg, flow_consts):
        adas, final_ada, rope = flow_consts
        x0 = torch.randn(1, self.patch_size, self.redae_dim, device=hist_latents.device)
        xt = torch.cat([hist_latents, x0], dim=1)
        cond = torch.cat([
            backbone_cond.repeat_interleave(self.patch_size, dim=1),
            spk_cond.unsqueeze(1).repeat(1, self.history_length + self.patch_size, 1),
        ], dim=-1)
        zeros_cond = torch.zeros_like(cond) if inference_cfg > 0 else None
        for ti in range(len(t_span) - 1):
            dt = t_span[ti + 1] - t_span[ti]
            x_in = torch.cat([xt, cond], dim=2)
            if inference_cfg > 0:
                x_in = torch.cat([x_in, torch.cat([xt, zeros_cond], dim=2)], dim=0)
            vt = self._dit_step(x_in, adas[:, ti], final_ada[ti], rope)
            if inference_cfg > 0:
                vt_cond, vt_cfg = vt.chunk(2, dim=0)
                vt = (1.0 + inference_cfg) * vt_cond - inference_cfg * vt_cfg
            xt[:, -self.patch_size:] = xt[:, -self.patch_size:] + dt.view(-1, 1, 1) * vt[:, -self.patch_size:]
        return xt[:, -self.patch_size:]

    @torch.no_grad()
    def generate(self, spk_emb, text_tokens, prompt_latents, n_timesteps=10, inference_cfg=2.0,
                 stop_threshold=0.5, min_gen_steps=6, max_gen_steps=None, progress_callback=None):
        device = text_tokens.device
        input_embeds = self.backbone_llm.embed_tokens(text_tokens).float()
        patch_prompt_latents = self.patch_encoder(prompt_latents)
        spk_embs_llm = self.spk_proj_llm(spk_emb.float())
        input_embeds = torch.cat([spk_embs_llm.unsqueeze(1), input_embeds, patch_prompt_latents], dim=1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        flow_consts = self._flow_constants(t_span)
        dit_spk_cond = self.spk_proj_dit(spk_emb.float())

        max_gen_steps = 400 if max_gen_steps is None else max_gen_steps
        n_prompt = prompt_latents.shape[1]
        latents_buf = prompt_latents.new_zeros(
            1, self.history_length + n_prompt + max_gen_steps * self.patch_size, self.redae_dim)
        latents_buf[:, self.history_length:self.history_length + n_prompt] = prompt_latents
        latents_filled = self.history_length + n_prompt
        n_prompt_patches = patch_prompt_latents.shape[1]
        cond_buf = input_embeds.new_zeros(
            1, self.history_patches + n_prompt_patches + max_gen_steps, input_embeds.shape[-1])
        cond_filled = self.history_patches
        backbone_cache = None
        terminal_progress = None
        if tqdm is not None:
            terminal_progress = tqdm(total=max_gen_steps, desc="FireRedTTS3 patches", unit="patch",
                                     ascii=False, dynamic_ncols=True, leave=True)
        try:
            for step_index in range(max_gen_steps):
                _check_interrupted()
                backbone_out, backbone_cache = self._backbone_one_step(input_embeds, cache=backbone_cache)
                stop_score = torch.sigmoid(self.stop_head(backbone_out[:, -1]).squeeze(-1)).item()
                if stop_score >= stop_threshold and (min_gen_steps is None or step_index >= min_gen_steps):
                    break
                if step_index == 0:
                    one_backbone_out = backbone_out[:, -n_prompt_patches:]
                else:
                    one_backbone_out = backbone_out[:, -1:]
                n_new = one_backbone_out.shape[1]
                cond_buf[:, cond_filled:cond_filled + n_new] = one_backbone_out
                cond_filled += n_new
                one_latents = self._flow_one_step(
                    hist_latents=latents_buf[:, latents_filled - self.history_length:latents_filled],
                    backbone_cond=self.dit_head(cond_buf[:, cond_filled - (self.history_patches + 1):cond_filled]),
                    spk_cond=dit_spk_cond,
                    t_span=t_span,
                    inference_cfg=inference_cfg,
                    flow_consts=flow_consts,
                )
                input_embeds = self.patch_encoder(one_latents)
                latents_buf[:, latents_filled:latents_filled + self.patch_size] = one_latents
                latents_filled += self.patch_size
                if progress_callback is not None:
                    progress_callback(step_index + 1, max_gen_steps)
                if terminal_progress is not None:
                    terminal_progress.update(1)
        finally:
            if terminal_progress is not None:
                if terminal_progress.n < terminal_progress.total:
                    terminal_progress.total = terminal_progress.n
                terminal_progress.close()
        return latents_buf[:, self.history_length:latents_filled]


class FireRedTTS3InstructCore(nn.Module):
    def __init__(self, config: dict[str, Any], attn_implementation: str = "sdpa"):
        super().__init__()
        from transformers import Qwen3ForCausalLM

        self.backbone_llm_config = _backbone_config(attn_implementation)
        self.backbone_llm = Qwen3ForCausalLM(self.backbone_llm_config)
        hidden = self.backbone_llm_config.hidden_size
        redae_dim = config.get("redae_dim", 64)
        self.patch_encoder = PatchEncoder(
            in_dim=redae_dim,
            out_dim=hidden,
            patch_size=config.get("patch_size", 4),
            hidden_size=config.get("patch_encoder_hidden_size", 1024),
            mlp_ratio=config.get("patch_encoder_mlp_ratio", 4),
            depth=config.get("patch_encoder_depth", 8),
            num_heads=config.get("patch_encoder_num_heads", 16),
        )
        dit_hidden = config.get("dit_hidden_size", 1024)
        self.dit_head = nn.Linear(hidden, dit_hidden)
        self.dit = DiT(
            in_channels=redae_dim + dit_hidden,
            out_channels=redae_dim,
            mlp_ratio=config.get("dit_mlp_ratio", 3),
            depth=config.get("dit_depth", 11),
            num_heads=config.get("dit_num_heads", 16),
            hidden_size=dit_hidden,
        )
        self.stop_head = nn.Linear(hidden, 1)
        self.redae_dim = redae_dim
        self.patch_size = self.patch_encoder.patch_size
        self.history_patches = config.get("num_history_patches", 2)
        self.history_length = self.history_patches * self.patch_size
        self.autocast_bf16 = True

    def _backbone_one_step(self, input_embeds, cache=None):
        device = input_embeds.device
        with _autocast_bf16(device, self.autocast_bf16):
            outs = self.backbone_llm.model(inputs_embeds=input_embeds, use_cache=True, past_key_values=cache)
        return outs.last_hidden_state.float(), outs.past_key_values

    def _flow_constants(self, t_span):
        return _dit_flow_constants(self, t_span)

    def _dit_step(self, x_in, adas_step, final_step, rope):
        return _dit_step_impl(self, x_in, adas_step, final_step, rope)

    def _flow_one_step(self, hist_latents, backbone_cond, t_span, inference_cfg, flow_consts):
        adas, final_ada, rope = flow_consts
        x0 = torch.randn(1, self.patch_size, self.redae_dim, device=hist_latents.device)
        xt = torch.cat([hist_latents, x0], dim=1)
        cond = backbone_cond.repeat_interleave(self.patch_size, dim=1)
        zeros_cond = torch.zeros_like(cond) if inference_cfg > 0 else None
        for ti in range(len(t_span) - 1):
            dt = t_span[ti + 1] - t_span[ti]
            x_in = torch.cat([xt, cond], dim=2)
            if inference_cfg > 0:
                x_in = torch.cat([x_in, torch.cat([xt, zeros_cond], dim=2)], dim=0)
            vt = self._dit_step(x_in, adas[:, ti], final_ada[ti], rope)
            if inference_cfg > 0:
                vt_cond, vt_cfg = vt.chunk(2, dim=0)
                vt = (1.0 + inference_cfg) * vt_cond - inference_cfg * vt_cfg
            xt[:, -self.patch_size:] = xt[:, -self.patch_size:] + dt.view(-1, 1, 1) * vt[:, -self.patch_size:]
        return xt[:, -self.patch_size:]

    def _embed_tokens(self, tokens):
        return self.backbone_llm.model.embed_tokens(tokens).float()

    @torch.no_grad()
    def generate(self, text_tokens, latents_in=None, latents_in_mask=None, latents_out=None,
                 latents_out_mask=None, infer_text=False, text_repetition_penalty=None,
                 text_do_sample=True, text_temperature=None, text_top_p=None, text_top_k=None,
                 n_timesteps=10, inference_cfg=2.0, stop_threshold=0.5, min_gen_steps=6,
                 max_gen_steps=None, progress_callback=None):
        from transformers.generation.logits_process import (
            LogitsProcessorList,
            RepetitionPenaltyLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        device = text_tokens.device
        input_embeds = self._embed_tokens(text_tokens)
        if latents_in is not None:
            latents_patch_in = self.patch_encoder(latents_in)
            input_embeds = input_embeds.masked_scatter(
                latents_in_mask.unsqueeze(-1), latents_patch_in.reshape(-1).to(input_embeds)
            )
        if latents_out is not None:
            latents_patch_out = self.patch_encoder(latents_out)
            input_embeds = input_embeds.masked_scatter(
                latents_out_mask.unsqueeze(-1), latents_patch_out.reshape(-1).to(input_embeds)
            )
        else:
            latents_out = torch.zeros(1, 0, self.redae_dim, device=device)
            latents_patch_out = None

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        flow_consts = self._flow_constants(t_span)
        backbone_cache = None

        if infer_text:
            processors = LogitsProcessorList()
            if text_repetition_penalty is not None and text_repetition_penalty != 1.0:
                processors.append(RepetitionPenaltyLogitsProcessor(penalty=text_repetition_penalty))
            if text_do_sample:
                if text_temperature is not None and text_temperature != 1.0:
                    processors.append(TemperatureLogitsWarper(temperature=float(text_temperature)))
                if text_top_k is not None and text_top_k > 0:
                    processors.append(TopKLogitsWarper(top_k=text_top_k))
                if text_top_p is not None and text_top_p < 1.0:
                    processors.append(TopPLogitsWarper(top_p=text_top_p))
            text_gen_ids = torch.empty((1, 0), dtype=torch.long, device=device)
            next_token = None
            for _text_step in range(200):
                _check_interrupted()
                backbone_out, backbone_cache = self._backbone_one_step(input_embeds, cache=backbone_cache)
                logits = self.backbone_llm.lm_head(backbone_out[:, -1, :])
                scores = processors(text_gen_ids, logits)
                if text_do_sample:
                    probs = torch.softmax(scores.float(), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)[:, 0]
                else:
                    next_token = scores.argmax(dim=-1)
                input_embeds = self._embed_tokens(next_token.unsqueeze(0))
                if next_token.item() == TEXT_EOT_ID:
                    break
                text_gen_ids = torch.cat([text_gen_ids, next_token.unsqueeze(0)], dim=1)
            if next_token is None:
                raise RuntimeError("FireRedTTS3 instruct text planning produced no tokens.")
            _, backbone_cache = self._backbone_one_step(input_embeds, cache=backbone_cache)
            next_token = next_token * 0 + AUDIO_SOS_ID
            input_embeds = self._embed_tokens(next_token.unsqueeze(0))
            _, backbone_cache = self._backbone_one_step(input_embeds, cache=backbone_cache)

        max_gen_steps = 400 if max_gen_steps is None else max_gen_steps
        n_prompt = latents_out.shape[1]
        latents_buf = latents_out.new_zeros(
            1, self.history_length + n_prompt + max_gen_steps * self.patch_size, self.redae_dim)
        if n_prompt:
            latents_buf[:, self.history_length:self.history_length + n_prompt] = latents_out
        latents_filled = self.history_length + n_prompt
        n_prompt_patches = latents_patch_out.shape[1] if latents_patch_out is not None else 0
        cond_buf = input_embeds.new_zeros(
            1, self.history_patches + n_prompt_patches + max_gen_steps, input_embeds.shape[-1])
        cond_filled = self.history_patches
        terminal_progress = None
        if tqdm is not None:
            terminal_progress = tqdm(total=max_gen_steps, desc="FireRedTTS3 patches", unit="patch",
                                     ascii=False, dynamic_ncols=True, leave=True)
        try:
            for step_index in range(max_gen_steps):
                _check_interrupted()
                backbone_out, backbone_cache = self._backbone_one_step(input_embeds, cache=backbone_cache)
                stop_score = torch.sigmoid(self.stop_head(backbone_out[:, -1]).squeeze(-1)).item()
                if stop_score >= stop_threshold and (min_gen_steps is None or step_index >= min_gen_steps):
                    break
                if step_index == 0 and latents_patch_out is not None:
                    one_backbone_out = backbone_out[:, -n_prompt_patches:]
                else:
                    one_backbone_out = backbone_out[:, -1:]
                n_new = one_backbone_out.shape[1]
                cond_buf[:, cond_filled:cond_filled + n_new] = one_backbone_out
                cond_filled += n_new
                one_latents = self._flow_one_step(
                    hist_latents=latents_buf[:, latents_filled - self.history_length:latents_filled],
                    backbone_cond=self.dit_head(cond_buf[:, cond_filled - (self.history_patches + 1):cond_filled]),
                    t_span=t_span,
                    inference_cfg=inference_cfg,
                    flow_consts=flow_consts,
                )
                input_embeds = self.patch_encoder(one_latents)
                latents_buf[:, latents_filled:latents_filled + self.patch_size] = one_latents
                latents_filled += self.patch_size
                if progress_callback is not None:
                    progress_callback(step_index + 1, max_gen_steps)
                if terminal_progress is not None:
                    terminal_progress.update(1)
        finally:
            if terminal_progress is not None:
                if terminal_progress.n < terminal_progress.total:
                    terminal_progress.total = terminal_progress.n
                terminal_progress.close()
        latents_gen = latents_buf[:, self.history_length:latents_filled]
        if infer_text:
            return latents_gen, text_gen_ids
        return latents_gen


# --------------------------------------------------------------------------- #
# ComfyUI castable module conversion (same approach as Higgs v3 node pack)
# --------------------------------------------------------------------------- #
class _ComfyLinear(nn.Linear):
    _combined_linear = None
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def forward(self, x):
        if not hasattr(self, "_v") and self.weight.device == x.device:
            return F.linear(x, self.weight, self.bias)
        weight, bias, stream = cast_bias_weight(self, x, offloadable=True)
        try:
            return F.linear(x, weight, bias)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


class _ComfyEmbedding(nn.Embedding):
    _combined_linear = None
    comfy_cast_weights = True
    weight_function = []
    bias_function = []
    bias = None

    def _weight_dtype(self):
        return getattr(self, "weight_comfy_model_dtype", None) or self.weight.dtype

    def forward(self, input):
        if not hasattr(self, "_v") and self.weight.device == input.device:
            return F.embedding(input, self.weight, self.padding_idx, self.max_norm,
                               self.norm_type, self.scale_grad_by_freq, self.sparse)
        weight, bias, stream = cast_bias_weight(self, dtype=self._weight_dtype(), device=input.device, offloadable=True)
        try:
            return F.embedding(input, weight, self.padding_idx, self.max_norm,
                               self.norm_type, self.scale_grad_by_freq, self.sparse)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


class _ComfyConv1d(nn.Conv1d):
    _combined_linear = None
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def forward(self, input):
        if (not hasattr(self, "_v") and self.weight.device == input.device
                and (self.bias is None or self.bias.device == input.device)):
            return self._conv_forward(input, self.weight, self.bias)
        weight, bias, stream = cast_bias_weight(self, input, offloadable=True)
        try:
            return self._conv_forward(input, weight, bias)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


class _ComfyConv2d(nn.Conv2d):
    _combined_linear = None
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def forward(self, input):
        if (not hasattr(self, "_v") and self.weight.device == input.device
                and (self.bias is None or self.bias.device == input.device)):
            return self._conv_forward(input, self.weight, self.bias)
        weight, bias, stream = cast_bias_weight(self, input, offloadable=True)
        try:
            return self._conv_forward(input, weight, bias)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


def _comfy_rmsnorm_forward(self, hidden_states):
    if not hasattr(self, "_v") and self.weight.device == hidden_states.device:
        weight = self.weight
        stream = None
        bias = None
    else:
        weight, bias, stream = cast_bias_weight(self, hidden_states, offloadable=True)
    try:
        return F.rms_norm(hidden_states, (hidden_states.shape[-1],), weight=weight, eps=self.eps)
    finally:
        uncast_bias_weight(self, weight, bias, stream)


def _patch_rmsnorm(module: nn.Module) -> None:
    if getattr(module, "_firered_comfy_cast_rmsnorm", False):
        return
    module.bias = None
    module.comfy_cast_weights = True
    module.weight_function = []
    module.bias_function = []
    module.forward = _comfy_rmsnorm_forward.__get__(module, module.__class__)
    module._firered_comfy_cast_rmsnorm = True


def _comfy_qwen3_rmsnorm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    if not hasattr(self, "_v") and self.weight.device == hidden_states.device:
        weight = self.weight
        stream = None
        bias = None
    else:
        weight, bias, stream = cast_bias_weight(self, hidden_states, offloadable=True)
    try:
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return weight * hidden_states.to(input_dtype)
    finally:
        uncast_bias_weight(self, weight, bias, stream)


def _patch_qwen3_rmsnorm(module: nn.Module) -> None:
    if getattr(module, "_firered_comfy_cast_rmsnorm", False):
        return
    module.bias = None
    module.comfy_cast_weights = True
    module.weight_function = []
    module.bias_function = []
    module.forward = _comfy_qwen3_rmsnorm_forward.__get__(module, module.__class__)
    module._firered_comfy_cast_rmsnorm = True


def convert_modules_for_comfy(model: nn.Module) -> None:
    """Patch castable modules in-place so DynamicVRAM can page their weights."""
    for module in model.modules():
        if isinstance(module, (_ComfyLinear, _ComfyEmbedding, _ComfyConv1d, _ComfyConv2d)):
            continue
        if isinstance(module, nn.Linear):
            module.__class__ = _ComfyLinear
        elif type(module) is nn.Embedding:
            module.__class__ = _ComfyEmbedding
        elif isinstance(module, nn.Conv1d) and not hasattr(module, "parametrizations"):
            module.__class__ = _ComfyConv1d
        elif isinstance(module, nn.Conv2d) and not hasattr(module, "parametrizations"):
            module.__class__ = _ComfyConv2d
        elif module.__class__.__name__ == "Qwen3RMSNorm" and hasattr(module, "variance_epsilon"):
            _patch_qwen3_rmsnorm(module)
        elif type(module) is RMSNorm:
            _patch_rmsnorm(module)


def set_runtime_dtype(module: nn.Module, dtype: torch.dtype) -> None:
    """Tag floating tensors with the dtype Comfy/AIMDO should materialize.

    INT8 ConvRot weights are never tagged (not floating), and per-row weight
    scales stay fp32 so the quantized kernels receive exact scales.
    """
    for sub in module.modules():
        for name, value in sub.named_parameters(recurse=False):
            if value is not None and value.is_floating_point() and not name.endswith("inv_freq") and not name.endswith("weight_scale"):
                setattr(sub, f"{name}_comfy_model_dtype", dtype)
        for name, value in sub.named_buffers(recurse=False):
            if value is not None and value.is_floating_point() and not name.endswith("inv_freq") and not name.endswith("weight_scale"):
                setattr(sub, f"{name}_comfy_model_dtype", dtype)


# --------------------------------------------------------------------------- #
# Weight loading
# --------------------------------------------------------------------------- #
def read_config(model_dir: Path) -> dict[str, Any]:
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {model_dir}.")
    return json.loads(config_path.read_text(encoding="utf-8"))


def iter_safetensor_items(model_dir: Path) -> Iterable[tuple[str, torch.Tensor]]:
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]
    for shard in shards:
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def _set_tensor(module: nn.Module, name: str, tensor: torch.Tensor, dtype: torch.dtype | None) -> None:
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype=dtype)
    try:
        from accelerate.utils.modeling import set_module_tensor_to_device

        set_module_tensor_to_device(module, name, device="cpu", value=tensor.contiguous())
        return
    except ImportError:
        pass
    target = dict(module.named_parameters(remove_duplicate=False)).get(name)
    if target is None:
        target = dict(module.named_buffers(remove_duplicate=False)).get(name)
    if target is None:
        raise KeyError(name)
    if target.shape != tensor.shape:
        raise ValueError(f"Shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}")
    target.data = tensor.contiguous()


def load_safetensors_into(model: nn.Module, model_dir: Path,
                          dtype_policy=None, ignore_missing: tuple[str, ...] = ()) -> None:
    """Load every tensor in model_dir into model, casting floats per dtype_policy(name).

    Keys ending in .comfy_quant are quantization metadata consumed by the int8
    runtime (see int8.py), never module tensors; they are skipped here.
    """
    param_names = set(dict(model.named_parameters(remove_duplicate=False)))
    buffer_names = set(dict(model.named_buffers(remove_duplicate=False)))
    loaded: set[str] = set()
    unexpected: list[str] = []
    quant_meta = 0
    for name, tensor in iter_safetensor_items(model_dir):
        if name.endswith(".comfy_quant"):
            quant_meta += 1
            continue
        if name not in param_names and name not in buffer_names:
            unexpected.append(name)
            continue
        target_dtype = dtype_policy(name) if dtype_policy is not None else None
        _set_tensor(model, name, tensor, target_dtype)
        loaded.add(name)
    missing = [
        name for name in param_names
        if name not in loaded and not any(pat in name for pat in ignore_missing)
    ]
    if missing:
        raise RuntimeError(f"Weights missing from {model_dir}: {len(missing)} tensor(s), first: {missing[:8]}")
    if unexpected:
        logger.debug("Ignored %d unexpected tensor(s) from %s, first: %s", len(unexpected), model_dir, unexpected[:8])
    if quant_meta:
        logger.debug("Consumed %d comfy_quant metadata entries from %s.", quant_meta, model_dir)
    _materialize_buffers(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _materialize_buffers(model: nn.Module) -> None:
    """Recompute deterministic buffers that are absent from the checkpoint."""
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.rope_init()
        elif isinstance(module, ISTFT):
            module.window = torch.hann_window(module.win_length)
        elif hasattr(module, "compute_default_rope_parameters") and hasattr(module, "config"):
            # transformers Qwen3RotaryEmbedding: inv_freq/original_inv_freq are non-persistent
            try:
                inv_freq, scaling = module.compute_default_rope_parameters(module.config, torch.device("cpu"))
                module.inv_freq = inv_freq
                if hasattr(module, "original_inv_freq"):
                    module.original_inv_freq = inv_freq.clone()
                module.attention_scaling = scaling
            except Exception:
                pass
        elif hasattr(module, "rope_init") and callable(module.rope_init):
            try:
                module.rope_init()
            except Exception:
                pass
    for name, buf in model.named_buffers(remove_duplicate=False):
        if buf is not None and buf.is_meta:
            raise RuntimeError(f"Buffer {name} is still on the meta device after weight loading.")


def tie_core_weights(core: nn.Module) -> None:
    backbone = getattr(core, "backbone_llm", None)
    if backbone is not None and hasattr(backbone, "tie_weights"):
        backbone.tie_weights()


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def comfy_audio_to_tensor(audio: dict) -> tuple[torch.Tensor, int]:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    wav = waveform[0].detach().float().cpu()
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    elif wav.ndim == 2:
        wav = wav.squeeze(0)
    return wav.contiguous(), sample_rate


def tensor_audio_to_comfy(audio: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> dict:
    audio = audio.detach().float().cpu().clamp(-1.0, 1.0)
    return {"waveform": audio.view(1, 1, -1).contiguous(), "sample_rate": int(sample_rate)}


def cross_fade(seg_a: torch.Tensor, seg_b: torch.Tensor, fade_len: int) -> torch.Tensor:
    if fade_len <= 0:
        return torch.cat([seg_a, seg_b], dim=1)
    fade_len = int(min(fade_len, seg_a.shape[1], seg_b.shape[1]))
    if fade_len <= 0:
        return torch.cat([seg_a, seg_b], dim=1)
    ramp = torch.linspace(0.0, 1.0, fade_len, device=seg_a.device, dtype=seg_a.dtype).view(1, -1)
    overlap = seg_a[:, -fade_len:] * (1.0 - ramp) + seg_b[:, :fade_len] * ramp
    return torch.cat([seg_a[:, :-fade_len], overlap, seg_b[:, fade_len:]], dim=1)


@contextlib.contextmanager
def attention_runtime(attention: str):
    """Temporarily route F.scaled_dot_product_attention through sageattention."""
    if attention != "sageattention":
        yield
        return
    from sageattention import sageattn

    original_sdpa = F.scaled_dot_product_attention

    def sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if (attn_mask is not None or dropout_p not in (0, 0.0) or query.device.type != "cuda"
                or query.dtype not in (torch.float16, torch.bfloat16)):
            return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                 is_causal=is_causal, scale=scale, **kwargs)
        try:
            output = sageattn(query, key, value, tensor_layout="HND", is_causal=is_causal, sm_scale=scale)
            return output[0] if isinstance(output, tuple) else output
        except Exception:
            return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                 is_causal=is_causal, scale=scale, **kwargs)

    F.scaled_dot_product_attention = sage_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa


# --------------------------------------------------------------------------- #
# Runtime helpers: tokenize, prompt preparation, per-sentence task calls
# --------------------------------------------------------------------------- #
def tokenize_text(bundle: Any, text: str) -> torch.Tensor:
    tokens = bundle.tokenizer(text, truncation=False, padding=False, add_special_tokens=False)["input_ids"]
    return torch.tensor([tokens], dtype=torch.long, device=bundle.device)


def measure_tokens(bundle: Any, text: str) -> int:
    tokens = bundle.tokenizer(text, truncation=False, padding=False, add_special_tokens=False)["input_ids"]
    return len(tokens)


def tokenize_prompt_audio(bundle: Any, audio: torch.Tensor, audio_sr: int) -> tuple[torch.Tensor, int]:
    """Resample/pad/encode the prompt clip. Returns (latents fp32, padded samples)."""
    from .tokenizer import REDAE_PROMPT_SCALE

    redae = bundle.redae
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    audio = audio[:1]
    audio = torchaudio.functional.resample(audio, audio_sr, redae.sample_rate)
    audio = redae.pad_to_multiple_of(audio, redae.downsample_rate * bundle.core.patch_size)
    audio = audio.to(bundle.device)
    latents = redae.encode(audio, redae.sample_rate)
    latents = (latents * REDAE_PROMPT_SCALE[bundle.variant]).to(torch.float32)
    return latents, audio.shape[1]


def speaker_embedding(bundle: Any, audio: torch.Tensor, audio_sr: int) -> torch.Tensor:
    if bundle.campp is None:
        raise RuntimeError("Speaker encoder (CAM++) is only loaded for the base model.")
    emb = bundle.campp(audio, audio_sr)
    return emb.to(bundle.device)


def base_clone_one(bundle: Any, *, text: str, language: str, prompt_text: str,
                   prompt_latents: torch.Tensor, prompt_audio_len: int, spk_emb: torch.Tensor,
                   stop_threshold: float, n_timesteps: int, inference_cfg: float, seed: int,
                   max_gen_steps: int | None = None, progress_callback=None) -> tuple[torch.Tensor, int]:
    from .tokenizer import MULTI_DIALECT_TAGS, MULTI_LANG_TAGS

    lang_tag = f"<|{language}|>"
    if lang_tag not in MULTI_LANG_TAGS + MULTI_DIALECT_TAGS:
        raise ValueError(f"Invalid language for FireRedTTS3 base: {language}")
    text_tokens = tokenize_text(bundle, f"{lang_tag}<|sot|>{prompt_text}{text}<|eot|>")
    if seed:
        fix_seed(int(seed))
    with torch.inference_mode(), attention_runtime(bundle.attention):
        gen_latents = bundle.core.generate(
            spk_emb=spk_emb,
            text_tokens=text_tokens,
            prompt_latents=prompt_latents,
            n_timesteps=int(n_timesteps),
            inference_cfg=float(inference_cfg),
            stop_threshold=float(stop_threshold),
            min_gen_steps=6,
            max_gen_steps=max_gen_steps,
            progress_callback=progress_callback,
        )
        gen_audio, gen_audio_sr = bundle.redae.decode(gen_latents)
    return gen_audio[:, prompt_audio_len:], gen_audio_sr


def instruct_clone_one(bundle: Any, *, text: str, prompt_text: str, prompt_latents: torch.Tensor,
                       stop_threshold: float, n_timesteps: int, inference_cfg: float, seed: int,
                       max_gen_steps: int | None = None, progress_callback=None) -> tuple[torch.Tensor, int]:
    from .tokenizer import CHATML_LATENT_OUT_PAD_ID, compose_generate_input_tts

    text_in = compose_generate_input_tts(prompt_latents.shape[1] // bundle.core.patch_size, prompt_text, text)
    text_tokens = tokenize_text(bundle, text_in)
    if seed:
        fix_seed(int(seed))
    with torch.inference_mode(), attention_runtime(bundle.attention):
        gen_latents = bundle.core.generate(
            text_tokens=text_tokens,
            latents_out=prompt_latents,
            latents_out_mask=(text_tokens == CHATML_LATENT_OUT_PAD_ID),
            infer_text=False,
            n_timesteps=int(n_timesteps),
            inference_cfg=float(inference_cfg),
            stop_threshold=float(stop_threshold),
            min_gen_steps=6,
            max_gen_steps=max_gen_steps,
            progress_callback=progress_callback,
        )
        gen_audio, gen_audio_sr = bundle.redae.decode(gen_latents / REDAE_SCALE)
    prompt_samples = bundle.redae.downsample_rate * prompt_latents.shape[1]
    return gen_audio[:, prompt_samples:], gen_audio_sr


def voice_design_one(bundle: Any, *, instruction: str, text: str, n_timesteps: int, inference_cfg: float,
                     seed: int, text_temperature: float, text_top_p: float, text_top_k: int,
                     text_repetition_penalty: float, max_gen_steps: int | None = None,
                     progress_callback=None) -> tuple[torch.Tensor, int, str]:
    from .tokenizer import compose_generate_input_voice_design

    text_tokens = tokenize_text(bundle, compose_generate_input_voice_design(instruction, text))
    if seed:
        fix_seed(int(seed))
    with torch.inference_mode(), attention_runtime(bundle.attention):
        gen_latents, gen_text_ids = bundle.core.generate(
            text_tokens=text_tokens,
            infer_text=True,
            text_repetition_penalty=float(text_repetition_penalty),
            text_do_sample=True,
            text_temperature=float(text_temperature),
            text_top_p=float(text_top_p),
            text_top_k=int(text_top_k),
            n_timesteps=int(n_timesteps),
            inference_cfg=float(inference_cfg),
            stop_threshold=0.5,
            min_gen_steps=6,
            max_gen_steps=max_gen_steps,
            progress_callback=progress_callback,
        )
        gen_audio, gen_audio_sr = bundle.redae.decode(gen_latents / REDAE_SCALE)
    gen_text = bundle.tokenizer.decode(gen_text_ids.squeeze(0).cpu())
    return gen_audio, gen_audio_sr, gen_text


def semantic_edit_one(bundle: Any, *, instruction: str, latents_in: torch.Tensor, n_timesteps: int,
                      inference_cfg: float, seed: int, stop_threshold: float = 0.5, max_gen_steps: int | None = None,
                      progress_callback=None) -> tuple[torch.Tensor, int, str]:
    from .tokenizer import CHATML_LATENT_IN_PAD_ID, compose_generate_input_semantic_edit

    text_in = compose_generate_input_semantic_edit(instruction, latents_in.shape[1] // bundle.core.patch_size)
    text_tokens = tokenize_text(bundle, text_in)
    if seed:
        fix_seed(int(seed))
    with torch.inference_mode(), attention_runtime(bundle.attention):
        gen_latents, gen_text_ids = bundle.core.generate(
            text_tokens=text_tokens,
            latents_in=latents_in,
            latents_in_mask=(text_tokens == CHATML_LATENT_IN_PAD_ID),
            infer_text=True,
            text_repetition_penalty=1.0,
            text_do_sample=False,
            n_timesteps=int(n_timesteps),
            inference_cfg=float(inference_cfg),
            stop_threshold=float(stop_threshold),
            min_gen_steps=6,
            max_gen_steps=max_gen_steps,
            progress_callback=progress_callback,
        )
        gen_audio, gen_audio_sr = bundle.redae.decode(gen_latents / REDAE_SCALE)
    gen_text = bundle.tokenizer.decode(gen_text_ids.squeeze(0).cpu())
    return gen_audio, gen_audio_sr, gen_text


def acoustic_edit_one(bundle: Any, *, instruction: str, latents_in: torch.Tensor, n_timesteps: int,
                      inference_cfg: float, seed: int, stop_threshold: float = 0.5, max_gen_steps: int | None = None,
                      progress_callback=None) -> tuple[torch.Tensor, int]:
    from .tokenizer import CHATML_LATENT_IN_PAD_ID, compose_generate_input_acoustic_edit

    text_in = compose_generate_input_acoustic_edit(instruction, latents_in.shape[1] // bundle.core.patch_size)
    text_tokens = tokenize_text(bundle, text_in)
    if seed:
        fix_seed(int(seed))
    with torch.inference_mode(), attention_runtime(bundle.attention):
        gen_latents = bundle.core.generate(
            text_tokens=text_tokens,
            latents_in=latents_in,
            latents_in_mask=(text_tokens == CHATML_LATENT_IN_PAD_ID),
            infer_text=False,
            n_timesteps=int(n_timesteps),
            inference_cfg=float(inference_cfg),
            stop_threshold=float(stop_threshold),
            min_gen_steps=6,
            max_gen_steps=max_gen_steps,
            progress_callback=progress_callback,
        )
        gen_audio, gen_audio_sr = bundle.redae.decode(gen_latents / REDAE_SCALE)
    return gen_audio, gen_audio_sr
