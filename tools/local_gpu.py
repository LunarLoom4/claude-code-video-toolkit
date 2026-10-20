#!/usr/bin/env python3
"""
Local GPU provider: run cloud GPU tools on your own NVIDIA card, for free.

`--cloud local` makes a tool run the same job it would send to Modal/RunPod in
a local worker process instead. Workers take the cloud endpoint's JSON payload
and return its response shape, so the calling tool parses either the same way.

The workers live in an isolated environment at ~/.video-toolkit/local-gpu
(Python 3.12 + CUDA PyTorch), not in the toolkit's uv env: torch is several GB,
and most users never need it locally.

Usage:
    # One-time install (downloads ~3GB of PyTorch; model weights on first use)
    uv run tools/local_gpu.py --install qwen3_tts upscale
    uv run tools/local_gpu.py --install --all

    # What's installed, and can it see the GPU?
    uv run tools/local_gpu.py --status

    # Then any supported tool:
    uv run tools/qwen3_tts.py --text "Hello" --cloud local --output hello.mp3
    uv run tools/upscale.py --input in.png --output out.png --cloud local

Supported tools: qwen3_tts (and voiceover.py --provider qwen3), upscale.
The whisper feature has no --cloud flag: captions.py --engine whisper uses it
automatically when the uv env's torch can't see a CUDA GPU.

Hardware:
    NVIDIA GPU with 8GB+ VRAM. PyTorch is the CUDA 12.8 build, which RTX 50-series
    (Blackwell) cards require; older builds fail on them with "no kernel image".
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from win_encoding import ensure_utf8_output, utf8_env

LOCAL_GPU_HOME = Path(os.getenv("LOCAL_GPU_HOME", Path.home() / ".video-toolkit" / "local-gpu"))
WORKERS_DIR = Path(__file__).parent / "local_workers"
PYTHON_VERSION = "3.12"

# CUDA 12.8 wheels: the oldest CUDA with Blackwell (sm_120) kernels. torch 2.8 is
# the last release whose torchaudio still does its own audio I/O (2.9 moved it to
# torchcodec), which keeps qwen-tts working unmodified.
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
TORCH_PACKAGES = ["torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0"]

# Per-tool packages, installed on top of torch. Pins match the cloud images where
# the cloud image uses the same library.
FEATURES = {
    "qwen3_tts": {
        "description": "Qwen3-TTS speech (built-in speakers, clone, voice design)",
        "packages": ["qwen-tts==0.1.1", "soundfile"],
        "import": "qwen_tts",
        "worker": "qwen3_tts_worker.py",
    },
    "upscale": {
        # spandrel loads the same RealESRGAN weights as the cloud image's
        # basicsr/realesrgan, which no longer import on torchvision >= 0.17.
        "description": "RealESRGAN upscaling (general, anime, photo models)",
        "packages": ["spandrel==0.4.2", "opencv-python-headless", "numpy"],
        "import": "spandrel",
        "worker": "upscale_worker.py",
    },
    "whisper": {
        # No worker: captions.py reruns itself under this env's python when the
        # uv env's torch is CPU-only (the default uv wheel on Windows), so
        # whisper gets turbo on the GPU instead of base on the CPU.
        "description": "openai-whisper on CUDA for captions.py (turbo model)",
        "packages": ["openai-whisper>=20240930", "python-dotenv"],
        "import": "whisper",
        "worker": None,
    },
}


def venv_dir() -> Path:
    return LOCAL_GPU_HOME / ".venv"


def venv_python(venv: Path | None = None) -> Path:
    """Path to a venv's interpreter (Scripts\\python.exe on Windows)."""
    venv = venv or venv_dir()
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def feature_python(feature: str) -> Path | None:
    """The local env's interpreter if `feature` is installed in it, else None."""
    if quick_status()["features"].get(feature):
        return venv_python()
    return None


def _log(msg: str, verbose: bool = True):
    if verbose:
        print(msg, file=sys.stderr)


def quick_status() -> dict:
    """Which features are installed, from the filesystem alone (no torch import).

    check_status() is the real test, but importing torch takes 5-50s.
    """
    py = venv_python()
    site = next(iter(venv_dir().glob("[Ll]ib/**/site-packages")), None) if py.exists() else None
    features = {
        name: bool(site and (site / f["import"]).is_dir())
        for name, f in FEATURES.items()
    }
    return {"env": py.exists(), "path": str(LOCAL_GPU_HOME), "features": features}


def check_status() -> dict:
    """Report the local environment: interpreter, torch, CUDA, installed features."""
    py = venv_python()
    status = {
        "installed": False,
        "path": str(LOCAL_GPU_HOME),
        "python": str(py) if py.exists() else None,
        "torch": None,
        "cuda": None,
        "gpu": None,
        "vram_gb": None,
        "features": {name: False for name in FEATURES},
        "error": None,
    }
    if not py.exists():
        status["error"] = "not installed (run: uv run tools/local_gpu.py --install --all)"
        return status

    probe = (
        "import json, importlib.util as u\n"
        "out = {'features': {}}\n"
        f"for name, mod in {json.dumps({n: f['import'] for n, f in FEATURES.items()})}.items():\n"
        "    out['features'][name] = u.find_spec(mod) is not None\n"
        "try:\n"
        "    import torch\n"
        "    out['torch'] = torch.__version__\n"
        "    out['cuda'] = torch.version.cuda\n"
        "    if torch.cuda.is_available():\n"
        "        p = torch.cuda.get_device_properties(0)\n"
        "        out['gpu'] = p.name\n"
        "        out['vram_gb'] = round(p.total_memory / 1024**3, 1)\n"
        "        torch.zeros(1, device='cuda').add_(1)  # fails here on a CUDA build without this GPU's kernels\n"
        "except Exception as e:\n"
        "    out['error'] = f'{type(e).__name__}: {e}'\n"
        "print(json.dumps(out))\n"
    )
    try:
        result = subprocess.run(
            [str(py), "-c", probe], capture_output=True, text=True,
            encoding="utf-8", env=utf8_env(), timeout=120,
        )
        data = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as e:
        status["error"] = f"probe failed: {e}"
        return status

    status.update({k: data.get(k) for k in ("torch", "cuda", "gpu", "vram_gb")})
    status["features"].update(data.get("features", {}))
    status["error"] = data.get("error")
    if not status["error"] and not status["gpu"]:
        status["error"] = "PyTorch cannot see a CUDA GPU (CPU-only build, or driver too old)"
    status["installed"] = status["gpu"] is not None and not status["error"]
    return status


def install(features: list[str], verbose: bool = True) -> bool:
    """Create the local env and install torch + the given features' packages."""
    unknown = [f for f in features if f not in FEATURES]
    if unknown:
        print(f"Error: unknown feature(s): {', '.join(unknown)}. "
              f"Choose from: {', '.join(FEATURES)}", file=sys.stderr)
        return False

    uv = shutil.which("uv")
    if not uv:
        print("Error: uv not found. Install it from https://docs.astral.sh/uv/", file=sys.stderr)
        return False

    capture = not verbose
    venv = venv_dir()
    py = venv_python(venv)

    _log("=" * 60, verbose)
    _log("Local GPU environment", verbose)
    _log("=" * 60, verbose)
    _log(f"Path: {LOCAL_GPU_HOME}", verbose)
    _log(f"Features: {', '.join(features)}", verbose)
    _log("", verbose)

    _log(f"[1/3] Creating Python {PYTHON_VERSION} environment...", verbose)
    if not py.exists():
        LOCAL_GPU_HOME.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([uv, "venv", "--python", PYTHON_VERSION, str(venv)],
                                capture_output=capture)
        if result.returncode != 0:
            print("Error: failed to create the environment", file=sys.stderr)
            return False
    else:
        _log("  Exists, reusing", verbose)

    _log("[2/3] Installing PyTorch (CUDA 12.8 build, ~3GB)...", verbose)
    torch_cmd = [uv, "pip", "install", "--python", str(py), *TORCH_PACKAGES]
    if sys.platform != "darwin":
        torch_cmd += ["--index-url", TORCH_INDEX_URL]
    if subprocess.run(torch_cmd, capture_output=capture).returncode != 0:
        print("Error: failed to install PyTorch", file=sys.stderr)
        return False

    _log("[3/3] Installing tool packages...", verbose)
    packages = [p for f in features for p in FEATURES[f]["packages"]]
    # Constrain torch to what step 2 installed, so no package can swap the CUDA
    # build for PyPI's CPU-only one on Windows.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(TORCH_PACKAGES))
        constraints = f.name
    try:
        result = subprocess.run(
            [uv, "pip", "install", "--python", str(py), "--constraint", constraints, *packages],
            capture_output=capture,
        )
    finally:
        os.unlink(constraints)
    if result.returncode != 0:
        print("Error: failed to install tool packages", file=sys.stderr)
        return False

    status = check_status()
    _log("", verbose)
    _log("=" * 60, verbose)
    if status["installed"]:
        _log(f"Ready: {status['gpu']} ({status['vram_gb']}GB), "
             f"torch {status['torch']} / CUDA {status['cuda']}", verbose)
        _log("Model weights download on first use of each tool.", verbose)
    else:
        _log(f"Installed, but GPU check failed: {status['error']}", verbose)
    _log("=" * 60, verbose)
    return status["installed"]


def call_local(
    payload: dict,
    tool_name: str,
    progress_label: str = "Processing",
    progress=None,
) -> tuple[dict, float]:
    """Run a tool's job on the local GPU. Same contract as call_cloud_endpoint().

    No timeout: the first run of a tool downloads its model weights, which can
    take longer than any cloud timeout. The worker can be interrupted with Ctrl-C.
    """
    def _emit(stage, msg, level="dim", pct=None):
        if progress:
            progress.event(stage, msg, pct=pct, level=level)

    feature = FEATURES.get(tool_name)
    if not feature or not feature["worker"]:
        supported = ", ".join(n for n, f in FEATURES.items() if f["worker"])
        return {"error": f"{tool_name} has no local GPU mode. Supported: {supported}"}, 0

    py = venv_python()
    if not py.exists():
        return {"error": f"Local GPU environment not installed. Run:\n"
                         f"  uv run tools/local_gpu.py --install {tool_name}"}, 0

    request = payload.get("input", payload) if isinstance(payload, dict) else payload
    work_dir = Path(tempfile.mkdtemp(prefix=f"local_{tool_name}_"))
    request_path = work_dir / "request.json"
    response_path = work_dir / "response.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    env = utf8_env()
    env["PYTHONUNBUFFERED"] = "1"  # stream worker logs live, not at exit
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    start = time.time()
    _emit("submit", f"{progress_label} on local GPU...", level="info")
    heartbeat_ctx = (progress.heartbeat(
        "waiting", f"{progress_label} on local GPU... ({{elapsed:.0f}}s)"
    ) if progress else contextmanager(lambda: (yield))())

    try:
        with heartbeat_ctx:
            # Worker logs go to our stderr so --json output on stdout stays clean.
            result = subprocess.run(
                [str(py), str(WORKERS_DIR / feature["worker"]), str(request_path), str(response_path)],
                stdout=sys.stderr, env=env,
            )
        elapsed = time.time() - start

        if not response_path.exists():
            msg = f"Local worker exited with code {result.returncode} and no response (see log above)"
            _emit("error", msg, level="error")
            return {"error": msg}, elapsed

        output = json.loads(response_path.read_text(encoding="utf-8"))
        if output.get("error"):
            _emit("error", output["error"], level="error")
        else:
            _emit("complete", f"Completed in {elapsed:.1f}s", pct=100, level="success")
        return output, elapsed
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    ensure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Install and check the local GPU provider (--cloud local)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[1] if __doc__ else None,
    )
    parser.add_argument("--install", nargs="*", metavar="FEATURE",
                        help=f"Install features: {', '.join(FEATURES)}")
    parser.add_argument("--all", action="store_true", help="With --install: every feature")
    parser.add_argument("--status", action="store_true", help="Show what's installed")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.install is not None:
        features = list(FEATURES) if args.all else args.install
        if not features:
            parser.error(f"--install needs feature names or --all ({', '.join(FEATURES)})")
        ok = install(features, verbose=not args.json)
        if args.json:
            print(json.dumps(check_status(), indent=2))
        sys.exit(0 if ok else 1)

    # --status, and the default with no flags
    status = check_status()
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(f"Local GPU environment: {status['path']}")
        if status["python"]:
            print(f"  torch:  {status['torch']} (CUDA {status['cuda']})")
            print(f"  GPU:    {status['gpu'] or '-'}"
                  + (f" ({status['vram_gb']}GB)" if status["vram_gb"] else ""))
            for name, ok in status["features"].items():
                print(f"  {name:<10} {'installed' if ok else 'not installed'}"
                      f"  - {FEATURES[name]['description']}")
        if status["error"]:
            print(f"  Problem: {status['error']}")
    sys.exit(0 if status["installed"] else 1)


if __name__ == "__main__":
    main()
