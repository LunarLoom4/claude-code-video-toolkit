"""
Local GPU worker for RealESRGAN upscaling (run by tools/local_gpu.py, not directly).

Same models, weights and output as docker/modal-upscale/app.py, loaded with
spandrel instead of basicsr/realesrgan (which no longer import on the torchvision
that Blackwell GPUs need). Differences from the cloud endpoint:

- Reads `image_path` and writes `output_path` directly; no upload, no base64.
- Tiled inference (512px tiles) so large inputs fit in 8GB of VRAM.
- No face enhancement (GFPGAN depends on basicsr too).

Usage: python upscale_worker.py request.json response.json
"""

import json
import sys
import time
import traceback
import urllib.request
from pathlib import Path

WEIGHT_URLS = {
    "general": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "photo": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
}
TILE = 512
TILE_PAD = 10


def weights_dir() -> Path:
    # Next to the local env (sys.prefix is ~/.video-toolkit/local-gpu/.venv)
    d = Path(sys.prefix).parent / "weights" / "realesrgan"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_model(model: str):
    import torch
    from spandrel import ModelLoader

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA GPU. Run: uv run tools/local_gpu.py --status")
    path = weights_dir() / f"{model}.pth"
    if not path.exists():
        print(f"Downloading {model} weights...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(WEIGHT_URLS[model], tmp)
        tmp.rename(path)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading model: {model}")
    descriptor = ModelLoader().load_from_file(str(path))
    descriptor.cuda().eval()
    if descriptor.supports_half:
        descriptor.half()
    return descriptor


def upscale_rgb(descriptor, rgb):
    """Upscale an HxWx3 float32 [0,1] array by the model's scale, tile by tile."""
    import numpy as np
    import torch

    scale = descriptor.scale
    dtype = torch.float16 if descriptor.supports_half else torch.float32
    h, w = rgb.shape[:2]
    out = np.zeros((h * scale, w * scale, 3), dtype=np.float32)

    with torch.inference_mode():
        for y in range(0, h, TILE):
            for x in range(0, w, TILE):
                # Padded input tile; the padding is cropped from the output to hide seams
                y0, y1 = max(y - TILE_PAD, 0), min(y + TILE + TILE_PAD, h)
                x0, x1 = max(x - TILE_PAD, 0), min(x + TILE + TILE_PAD, w)
                tile = torch.from_numpy(np.ascontiguousarray(rgb[y0:y1, x0:x1].transpose(2, 0, 1)))
                tile = tile.unsqueeze(0).to("cuda", dtype)
                result = descriptor(tile).squeeze(0).float().clamp_(0, 1).cpu().numpy().transpose(1, 2, 0)

                ty1, tx1 = min(y + TILE, h), min(x + TILE, w)
                oy, ox = (y - y0) * scale, (x - x0) * scale
                out[y * scale:ty1 * scale, x * scale:tx1 * scale] = \
                    result[oy:oy + (ty1 - y) * scale, ox:ox + (tx1 - x) * scale]
    return out


def process(request: dict) -> dict:
    import cv2
    import numpy as np

    start_time = time.time()
    image_path = request.get("image_path")
    output_path = request.get("output_path")
    if not image_path or not Path(image_path).exists():
        return {"error": f"image_path not found: {image_path}"}
    if not output_path:
        return {"error": "Missing required 'output_path'"}

    scale = request.get("scale", 4)
    model = request.get("model", "general")
    output_format = request.get("output_format", "png").lower()
    if request.get("face_enhance"):
        return {"error": "--face-enhance is not available with --cloud local (GFPGAN needs basicsr). "
                         "Drop the flag, or use --cloud modal/runpod for it."}
    if scale not in [2, 4]:
        return {"error": f"Invalid scale: {scale}. Must be 2 or 4"}
    if model not in WEIGHT_URLS:
        return {"error": f"Invalid model: {model}"}

    # imdecode/imencode rather than imread/imwrite: those can't open non-ASCII paths on Windows
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"error": "Failed to read image file"}
    max_value = 65535.0 if img.dtype == np.uint16 else 255.0
    in_h, in_w = img.shape[:2]
    print(f"Input: {in_w}x{in_h}")

    alpha = None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3]
        img = img[:, :, :3]

    descriptor = load_model(model)
    print("Upscaling...")
    inference_start = time.time()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / max_value
    out = cv2.cvtColor(upscale_rgb(descriptor, rgb), cv2.COLOR_RGB2BGR)

    # Like RealESRGANer's outscale: run at the model's native 4x, then resize
    out_w, out_h = in_w * scale, in_h * scale
    if out.shape[1] != out_w:
        out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    out = (out.clip(0, 1) * max_value).round().astype(np.uint16 if max_value > 255 else np.uint8)
    if alpha is not None:
        out = np.dstack([out, cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_LINEAR)])
    inference_time = time.time() - inference_start
    print(f"Output: {out_w}x{out_h} ({inference_time:.2f}s)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if output_format in ["jpg", "jpeg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif output_format == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 95]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    ok, buf = cv2.imencode(".jpg" if output_format == "jpeg" else f".{output_format}", out, params)
    if not ok:
        return {"error": f"Failed to encode output as {output_format}"}
    Path(output_path).write_bytes(buf.tobytes())

    return {
        "success": True,
        "output_path": output_path,
        "input_dimensions": f"{in_w}x{in_h}",
        "output_dimensions": f"{out_w}x{out_h}",
        "scale": scale,
        "model_used": model,
        "face_enhance": False,
        "output_format": output_format,
        "inference_time_seconds": round(inference_time, 2),
        "processing_time_seconds": round(time.time() - start_time, 2),
    }


def main():
    request_path, response_path = sys.argv[1], sys.argv[2]
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    try:
        result = process(request)
    except Exception as e:
        print(traceback.format_exc())
        result = {"error": f"Local upscale failed: {type(e).__name__}: {e}"}
    Path(response_path).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
