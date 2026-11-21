"""
Checks for utils/device.py. Run from the Popcorn root:
    python -m pytest tests/test_device.py      (or)      python tests/test_device.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import device as dev  # noqa: E402


def _with_env(value):
    old = os.environ.get("POPCORN_DEVICE")
    if value is None:
        os.environ.pop("POPCORN_DEVICE", None)
    else:
        os.environ["POPCORN_DEVICE"] = value
    dev.reset_device()
    return old


def _restore(old):
    _with_env(old)


def test_auto_order():
    old = _with_env(None)
    try:
        if torch.cuda.is_available():
            expected = "cuda"
        elif torch.backends.mps.is_available():
            expected = "mps"
        else:
            expected = "cpu"
        assert dev.get_device().type == expected
        assert dev.is_cuda() == (expected == "cuda")
        assert dev.is_mps() == (expected == "mps")
    finally:
        _restore(old)


def test_override_cpu_and_cached():
    old = _with_env("cpu")
    try:
        assert dev.get_device() == torch.device("cpu")
        os.environ["POPCORN_DEVICE"] = "mps"  # cached until reset
        assert dev.get_device() == torch.device("cpu")
        # CUDA-only helpers must be harmless off-CUDA
        dev.empty_cache()
        dev.synchronize()
        dev.seed_device(0)
        assert dev.memory_used_gb() != dev.memory_used_gb()  # NaN on CPU
    finally:
        _restore(old)


def test_override_unavailable_raises():
    if torch.cuda.is_available():
        return
    old = _with_env("cuda")
    try:
        try:
            dev.get_device()
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError for unavailable CUDA")
    finally:
        _restore(old)


def test_tensor_roundtrip():
    old = _with_env(None)
    try:
        d = dev.get_device()
        x = torch.ones(3, 3).to(d)
        assert x.device.type == d.type
        dev.empty_cache()
        dev.synchronize()
        assert (x * 2).sum().item() == 18.0
    finally:
        _restore(old)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
