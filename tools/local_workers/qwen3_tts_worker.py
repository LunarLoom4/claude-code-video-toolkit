"""
Local GPU worker for Qwen3-TTS (run by tools/local_gpu.py, not directly).

Mirrors docker/modal-qwen3-tts/app.py: same three modes (custom_voice, clone,
voice_design), same batch protocol, same response shape. Two differences:

- Loads only the model the request's mode needs. The cloud container keeps
  CustomVoice + Base resident (~7GB), which leaves no headroom on an 8GB card.
- Clone mode reads the reference from `ref_audio_path`; there is no upload.

Usage: python qwen3_tts_worker.py request.json response.json
"""

import base64
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

MODELS = {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


def _as_list(value, length: int):
    if isinstance(value, list):
        if len(value) != length:
            raise ValueError(f"list length mismatch: got {len(value)}, need {length}")
        return value
    return [value] * length


def resolve_model(repo: str) -> str:
    """Local snapshot path if cached, else the repo id (which downloads it).

    A path matters even when cached: given a repo id, transformers 4.57.3 asks the
    Hub about it on every load (tokenizer `_patch_mistral_regex`), ~30s on a slow
    link, and fails outright offline.
    """
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        print(f"Downloading {repo} (~4GB, first run only)...")
        return repo


def load_model(mode: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    # Initialise CUDA before from_pretrained: loading straight onto an
    # uninitialised device crashed in torch_cpu.dll (0xC0000005) on Windows.
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA GPU. Run: uv run tools/local_gpu.py --status")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading {MODELS[mode]}...")
    return Qwen3TTSModel.from_pretrained(
        resolve_model(MODELS[mode]),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def generate(request: dict) -> dict:
    import soundfile as sf

    start_time = time.time()

    text_in = request.get("text")
    if not text_in:
        return {"error": "Missing required field: text"}

    is_batch = isinstance(text_in, list)
    texts = text_in if is_batch else [text_in]
    if not texts or any(not t for t in texts):
        return {"error": "text list must contain non-empty strings"}

    mode = request.get("mode", "custom_voice")
    if mode not in MODELS:
        return {"error": f"Unknown mode: {mode}"}
    output_format = request.get("output_format", "mp3")

    try:
        languages = _as_list(request.get("language", "Auto"), len(texts))
    except ValueError as e:
        return {"error": str(e)}

    gen_kwargs = {}
    if "temperature" in request:
        gen_kwargs["temperature"] = float(request["temperature"])
    if "top_p" in request:
        gen_kwargs["top_p"] = float(request["top_p"])

    # Validate before the slow model load
    if mode == "clone":
        ref_audio_path = request.get("ref_audio_path")
        ref_text = request.get("ref_text")
        if not ref_text:
            return {"error": "ref_text is required for clone mode"}
        if not ref_audio_path or not Path(ref_audio_path).exists():
            return {"error": f"ref_audio_path not found: {ref_audio_path}"}
    elif mode == "voice_design":
        if is_batch:
            return {"error": "voice_design mode expects a single seed text, not a list"}
        if not request.get("instruct"):
            return {"error": "voice_design mode requires 'instruct'"}

    model = load_model(mode)

    # One text at a time throughout: batching via text=[...] hits a qwen-tts
    # tensor-aliasing bug (see the Modal app).
    wavs = []
    sr = None
    if mode == "clone":
        print("Extracting voice_clone_prompt from reference...")
        prompt = model.create_voice_clone_prompt(ref_audio=ref_audio_path, ref_text=ref_text)
        print(f"Generating {len(texts)} utterance(s) with shared clone prompt...")
        for i, (t, lang) in enumerate(zip(texts, languages)):
            single_wavs, sr = model.generate_voice_clone(
                text=t, language=lang, voice_clone_prompt=prompt, **gen_kwargs,
            )
            wavs.append(single_wavs[0] if isinstance(single_wavs, list) else single_wavs)
            print(f"  [{i + 1}/{len(texts)}] generated")

    elif mode == "voice_design":
        instruct = request["instruct"]
        print(f"Designing voice (instruct={instruct[:60]}...)")
        out, sr = model.generate_voice_design(
            text=texts[0], language=languages[0], instruct=instruct, **gen_kwargs,
        )
        wavs = list(out) if isinstance(out, list) else [out]

    else:
        speaker = request.get("speaker", "Ryan")
        instruct = request.get("instruct", "")
        print(f"Generating {len(texts)} utterance(s) with speaker {speaker}...")
        for i, (t, lang) in enumerate(zip(texts, languages)):
            cv_kwargs = {"text": t, "language": lang, "speaker": speaker}
            if instruct:
                cv_kwargs["instruct"] = instruct
            cv_kwargs.update(gen_kwargs)
            single_wavs, sr = model.generate_custom_voice(**cv_kwargs)
            wavs.append(single_wavs[0] if isinstance(single_wavs, list) else single_wavs)
            print(f"  [{i + 1}/{len(texts)}] generated")

    outputs = []
    with tempfile.TemporaryDirectory(prefix="qwen3tts_local_") as tmp:
        work_dir = Path(tmp)
        for i, wav_data in enumerate(wavs):
            wav_path = work_dir / f"output_{i}.wav"
            sf.write(str(wav_path), wav_data, sr)
            out_path = wav_path
            if output_format == "mp3":
                out_path = work_dir / f"output_{i}.mp3"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame",
                     "-b:a", "192k", str(out_path)],
                    capture_output=True, timeout=120, check=True,
                )
            outputs.append({
                "duration_seconds": round(len(wav_data) / sr, 2),
                "audio_base64": base64.b64encode(out_path.read_bytes()).decode("utf-8"),
            })

    result = {
        "success": True,
        "mode": mode,
        "processing_time_seconds": round(time.time() - start_time, 2),
        "outputs": outputs,
    }
    if not is_batch and outputs:
        result.update(outputs[0])
    return result


def main():
    request_path, response_path = sys.argv[1], sys.argv[2]
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    try:
        result = generate(request)
    except Exception as e:
        print(traceback.format_exc())
        result = {"error": f"Local Qwen3-TTS failed: {type(e).__name__}: {e}"}
    Path(response_path).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
