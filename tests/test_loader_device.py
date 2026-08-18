"""Loader device resolution tests (GitHub issue #1: bare 'cuda' must get an index)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, r"C:\Users\drbaph\Documents\ComfyUI")

import importlib.util  # noqa: E402

_pack_dir = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "fireredtts3_pack", _pack_dir / "__init__.py", submodule_search_locations=[str(_pack_dir)])
_pkg = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("fireredtts3_pack", _pkg)
_spec.loader.exec_module(_pkg)

from fireredtts3_pack.loader import resolve_device  # noqa: E402


def test_explicit_cuda_has_index():
    if not torch.cuda.is_available():
        print("cuda unavailable; skipped")
        return
    device = resolve_device("cuda")
    assert device.type == "cuda"
    assert device.index is not None, "bare 'cuda' must be normalized to an indexed device"
    int(device.index)  # must not raise (comfy_aimdo calls int(index))


def test_auto_cuda_has_index():
    if not torch.cuda.is_available():
        print("cuda unavailable; skipped")
        return
    device = resolve_device("auto")
    assert device.type == "cuda"
    assert device.index is not None


def test_indexed_cuda_preserved():
    if not torch.cuda.is_available():
        print("cuda unavailable; skipped")
        return
    device = resolve_device("cuda:0")
    assert device.index == 0


def test_cpu():
    assert resolve_device("cpu").type == "cpu"


if __name__ == "__main__":
    test_explicit_cuda_has_index()
    test_auto_cuda_has_index()
    test_indexed_cuda_preserved()
    test_cpu()
    print("test_loader_device: all passed")
