"""Flow-hoist optimization tests: rope cache, AdaLN precompute equivalence,
and the _dit_step path matching DiT.forward.

Runs without checkpoints on CPU (CUDA when available). Execute with pytest or
`python tests/test_flow_opt.py`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util  # noqa: E402

_pack_dir = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "fireredtts3_pack", _pack_dir / "__init__.py", submodule_search_locations=[str(_pack_dir)])
_pkg = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("fireredtts3_pack", _pkg)
_spec.loader.exec_module(_pkg)

from fireredtts3_pack import native  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _small_dit():
    torch.manual_seed(0)
    return native.DiT(in_channels=64 + 512 + 64, out_channels=64, mlp_ratio=2,
                      depth=3, num_heads=4, hidden_size=64).to(DEV).eval()


def _stub_core(dit, history_length=8, patch_size=4):
    return types.SimpleNamespace(dit=dit, history_length=history_length, patch_size=patch_size)


def test_rope_seq_len_cache():
    rope = native.RotaryEmbedding(16).to(DEV)
    a = rope.forward_from_seq_len(12)
    b = rope.forward_from_seq_len(12)
    assert a[0] is b[0], "cache should return the identical tensor"
    ref = rope(torch.arange(12, device=DEV))
    assert torch.equal(a[0], ref[0]) and a[1] == ref[1]
    c = rope.forward_from_seq_len(5)
    assert c[0].shape[1] == 5 and c[0] is not a[0]


def test_flow_block_precomputed_ada_matches():
    torch.manual_seed(0)
    block = native.FlowDiTBlock(hidden_size=64, num_heads=4, mlp_ratio=2).to(DEV).eval()
    x = torch.randn(2, 12, 64, device=DEV)
    c = torch.randn(2, 1, 64, device=DEV)
    with torch.no_grad():
        ref = block(x, c, None, None)
        ada = block.adaLN_modulation(c).chunk(9, dim=-1)
        out = block(x, None, None, None, ada=ada)
    assert torch.allclose(ref, out, atol=0, rtol=0), "precomputed ada must be bit-identical"


def test_final_layer_precomputed_ada_matches():
    torch.manual_seed(0)
    layer = native.AdaLNFinalLayer(hidden_size=64, out_channels=32).to(DEV).eval()
    x = torch.randn(2, 12, 64, device=DEV)
    c = torch.randn(2, 1, 64, device=DEV)
    with torch.no_grad():
        ref = layer(x, c)
        ada = layer.adaLN_modulation(c).chunk(2, dim=-1)
        out = layer(x, None, ada=ada)
    assert torch.allclose(ref, out, atol=0, rtol=0)


def test_dit_step_matches_dit_forward():
    dit = _small_dit()
    core = _stub_core(dit)
    x = torch.randn(2, 12, dit.in_channels, device=DEV)
    n_timesteps = 6
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=DEV)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    with torch.no_grad():
        adas, final_ada, rope = native._dit_flow_constants(core, t_span)
        max_diff = 0.0
        for ti in range(n_timesteps):
            t_in = t_span[ti].view(-1, 1, 1).expand(2, -1, -1)
            ref = dit(x=x, t=t_in)
            out = native._dit_step_impl(core, x, adas[:, ti], final_ada[ti], rope)
            max_diff = max(max_diff, (ref - out).abs().max().item())
    assert max_diff < 2e-5, f"hoisted path drifted: {max_diff}"


if __name__ == "__main__":
    test_rope_seq_len_cache()
    test_flow_block_precomputed_ada_matches()
    test_final_layer_precomputed_ada_matches()
    test_dit_step_matches_dit_forward()
    print("test_flow_opt: all passed on", DEV)
