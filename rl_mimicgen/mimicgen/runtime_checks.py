from __future__ import annotations

import os


def check_torch_cuda_compatibility() -> None:
    requested_device = os.environ.get("MIMICGEN_TRAIN_DEVICE", "").strip().lower()
    if requested_device == "cpu":
        return

    try:
        import torch
    except Exception:
        return

    compiled_arches = set(getattr(torch.cuda, "get_arch_list", lambda: [])())
    if not compiled_arches:
        return

    try:
        if not torch.cuda.is_available():
            return
        major, minor = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0)
    except Exception:
        return

    device_arch = f"sm_{major}{minor}"
    if device_arch in compiled_arches:
        return

    raise RuntimeError(
        "\n".join(
            [
                f"Detected GPU '{device_name}' with CUDA capability {device_arch}, "
                f"but the installed PyTorch build only includes: {', '.join(sorted(compiled_arches))}.",
                "",
                "This environment cannot run GPU training with the current torch wheel.",
                "For NVIDIA Blackwell GPUs such as RTX 5090, install a CUDA 12.8 PyTorch build first.",
                "To run on CPU instead, set MIMICGEN_TRAIN_DEVICE=cpu.",
            ]
        )
    )
