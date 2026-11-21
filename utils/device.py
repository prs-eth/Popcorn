"""
Device helper: one place that decides where tensors and models live.

Order: CUDA, then Apple MPS, then CPU. Set POPCORN_DEVICE (e.g. "cpu", "mps",
"cuda", "cuda:1") to override the automatic choice.

All CUDA-only calls (cache, memory stats, seeding) go through here and are
skipped when the active device is not CUDA.
"""

import os

import torch

_DEVICE = None


def _select_device() -> torch.device:
    override = os.environ.get("POPCORN_DEVICE", "").strip().lower()
    if override:
        device = torch.device(override)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("POPCORN_DEVICE={} but CUDA is not available".format(override))
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("POPCORN_DEVICE={} but MPS is not available".format(override))
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device() -> torch.device:
    """The device Popcorn runs on (chosen once per process)."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = _select_device()
    return _DEVICE


def reset_device() -> None:
    """Forget the cached choice, so the next get_device() re-reads POPCORN_DEVICE (tests)."""
    global _DEVICE
    _DEVICE = None


def is_cuda() -> bool:
    return get_device().type == "cuda"


def is_mps() -> bool:
    return get_device().type == "mps"


def empty_cache() -> None:
    """Release cached allocator memory on the active device (no-op on CPU)."""
    device = get_device()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def synchronize() -> None:
    """Wait for queued kernels on the active device (for honest timings)."""
    device = get_device()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def seed_device(seed: int) -> None:
    """Seed accelerator RNGs. torch.manual_seed already covers CPU and MPS; CUDA only if present."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def memory_used_gb() -> float:
    """Memory in use on the active device, in GB (NaN on CPU).

    CUDA: whole-GPU usage from NVML when available (as the original code logged), else the
    PyTorch allocator. MPS: memory the Metal driver holds for this process.
    """
    device = get_device()
    if device.type == "cuda":
        try:
            import nvidia_smi

            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device.index or 0)
            return nvidia_smi.nvmlDeviceGetMemoryInfo(handle).used / 1e9
        except Exception:
            return torch.cuda.memory_reserved(device) / 1e9
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return float("nan")
