"""Interpreter lock — refuse to run training on the wrong Python.

The baseline was accidentally trained on the CPU-only *system* Python 3.13
instead of the project's CUDA-enabled venv (audit: B-3 / run log showed
'CUDA is not available' warnings). This guard makes that regression loud and
fatal instead of silent.

Usage (top of any training entry point):

    from utils.require_venv import require_project_venv
    require_project_venv(require_cuda=True)

Escape hatches (for genuinely CPU-only machines / CI):
    ALLOW_SYSTEM_PYTHON=1   skip the venv-path check
    ALLOW_CPU=1             downgrade the CUDA requirement to a warning
"""
import os
import sys


def _project_root() -> str:
    # utils/require_venv.py -> repo root is one level up from utils/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def require_project_venv(require_cuda: bool = False) -> None:
    root      = _project_root()
    venv_dir  = os.path.join(root, "venv")
    exe       = os.path.abspath(sys.executable)
    on_venv   = exe.lower().startswith(venv_dir.lower())

    if not on_venv and os.environ.get("ALLOW_SYSTEM_PYTHON") != "1":
        raise RuntimeError(
            "Wrong Python interpreter.\n"
            f"  running: {exe}\n"
            f"  expected under: {venv_dir}\n"
            "This project must run on its CUDA-enabled venv. Use:\n"
            "  .\\run.ps1 python <script>            (PowerShell)\n"
            "  run.bat python <script>              (cmd)\n"
            "or call venv\\Scripts\\python.exe directly.\n"
            "Set ALLOW_SYSTEM_PYTHON=1 only if you really mean to."
        )

    # Device banner — always visible, so a CPU run can never pass unnoticed.
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev  = torch.cuda.get_device_name(0) if cuda else "CPU"
        print(f"[env] python={exe}")
        print(f"[env] torch={torch.__version__}  cuda_available={cuda}  device={dev}")
    except Exception as e:  # torch missing/broken shouldn't crash the guard itself
        print(f"[env] python={exe}")
        print(f"[env] WARNING: could not import torch to report device: {e}")
        cuda = False

    if require_cuda and not cuda:
        if os.environ.get("ALLOW_CPU") == "1":
            print("[env] WARNING: CUDA not available; proceeding on CPU (ALLOW_CPU=1). "
                  "Training will be very slow.")
        else:
            raise RuntimeError(
                "CUDA required but not available on this interpreter.\n"
                "Expected the CUDA-enabled venv (torch==2.5.1+cu121).\n"
                "Set ALLOW_CPU=1 to train on CPU anyway (very slow)."
            )
