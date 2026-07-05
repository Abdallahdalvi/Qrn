from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        return "not found"
    except subprocess.TimeoutExpired:
        return "timed out"

    output = (completed.stdout or completed.stderr or "").strip()
    return output if output else f"exit={completed.returncode}"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    print("=== NVIDIA ===")
    print(_run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]))

    print("\n=== ONNX Runtime package ===")
    import onnxruntime as ort  # noqa: PLC0415

    print(json.dumps({
        "version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "QURAN_ASR_PROVIDER": os.environ.get("QURAN_ASR_PROVIDER", "cpu"),
        "QURAN_ASR_DECODER": os.environ.get("QURAN_ASR_DECODER", "greedy"),
    }, indent=2))

    print("\n=== Backend session ===")
    import backend  # noqa: PLC0415

    print(json.dumps({
        "active_providers": backend._onnx_session.get_providers(),
        "decoder": backend.ASR_DECODER,
        "model": str(backend.ONNX_MODEL_PATH),
        "model_exists": backend.ONNX_MODEL_PATH.exists(),
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
