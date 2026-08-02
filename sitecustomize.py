"""Apply Safetensors Rig compatibility hooks in every vLLM worker process."""

from __future__ import annotations

import os


if os.environ.get("SAFETENSORS_VLLM_MUSICFLAMINGO_COMPAT") == "1":
    try:
        from vllm_launcher import _patch_musicflamingo

        _patch_musicflamingo()
    except ModuleNotFoundError:
        # This is harmless for non-vLLM helper interpreters. The actual vLLM
        # server has its full dependency set, so the hook is applied there.
        pass
