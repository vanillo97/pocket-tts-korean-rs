#!/usr/bin/env python3
"""synthesize.py — 한/영 합성 최소 코드.

    pip install -r python/requirements.txt
    python python/synthesize.py --lang ko --voice voice.wav \
        --text "안녕하세요. 한국어 음성 합성 모델입니다." --out out.wav
    python python/synthesize.py --lang en --voice alba \
        --text "Hello world!" --out out_en.wav
"""

import argparse
import re
import time
from pathlib import Path

KOREAN_CONFIG = "hf://seastar105/pocket-tts-korean-300m/korean.yaml"
ROOT = Path(__file__).resolve().parent.parent  # 리포 루트 (python/ 의 상위)
# CWD 가 아니라 리포 루트 기준. python/ 안에서 실행해도 동작한다.
MODEL_DIR_DEFAULT = str(ROOT / "models")
SAMPLE_RATE = 24000
# 신모델 per-language 임베딩 (pip 3.1.0 get_predefined_voice와 동일).
ENGLISH_VOICE_REPO = "kyutai/pocket-tts-without-voice-cloning"
ENGLISH_VOICE_PREFIX = "languages/english/embeddings"
ENGLISH_VOICE_REVISION = "e81d79e8194ad4c7ce879c87a4258ef20cbf2487"
ENGLISH_VOICES = ["alba", "marius", "javert", "jean",
                  "fantine", "cosette", "eponine", "azelma"]
DEFAULT_TEXT = {
    "ko": "안녕하세요. 한국어 음성 합성 모델입니다.",
    "en": "Hello world! I am Pocket TTS, running in Python.",
}


LANG_VARIANT = {"ko": "korean", "en": "english"}


def _local_bundle_complete(yaml_path: Path) -> bool:
    """local yaml이 있고, 가리키는 가중치·토크나이저가 모두 있으면 True."""
    if not yaml_path.is_file():
        return False
    try:
        txt = yaml_path.read_text()
    except OSError:
        return False
    for key in ("weights_path:", "tokenizer_path:"):
        m = re.search(rf"^\s*{key}\s*['\"]?(\S+?)['\"]?\s*$", txt, re.M)
        if not m:
            return False
        p = m.group(1)
        if p.startswith("hf://") or not Path(p).exists():
            return False
    return True


def resolve_model(lang: str, model_dir: str, explicit: str | None = None):
    """(config, source) 반환. source ∈ explicit/local/download.
    download(None config)은 호출자가 HF 기본값으로 로드 = 자동 다운로드."""
    if explicit:
        return explicit, "explicit"
    local = Path(model_dir) / f"{LANG_VARIANT[lang]}.local.yaml"
    if _local_bundle_complete(local):
        return str(local), "local"
    return None, "download"


def resolve_voice(lang: str, voice: str | None, model_dir: str = MODEL_DIR_DEFAULT) -> str:
    """stock 이름·로컬 경로를 get_state_for_audio_prompt용 지정자로 변환.
    models/{이름}.safetensors가 있으면 로컬 우선."""
    if voice:
        v = voice.strip()
        if lang == "en" and v in ENGLISH_VOICES:
            local = Path(model_dir) / f"{v}.safetensors"
            if local.is_file():
                return str(local)
            return (f"hf://{ENGLISH_VOICE_REPO}/{ENGLISH_VOICE_PREFIX}/{v}.safetensors"
                    f"@{ENGLISH_VOICE_REVISION}")
        return v
    if lang == "en":
        local = Path(model_dir) / "alba.safetensors"
        if local.is_file():
            return str(local)
        return (f"hf://{ENGLISH_VOICE_REPO}/{ENGLISH_VOICE_PREFIX}/alba.safetensors"
                f"@{ENGLISH_VOICE_REVISION}")
    return "voice.wav"


def apply_speed(audio, speed: float = 1.0, sample_rate: int = SAMPLE_RATE):
    """재생 속도 조절 (피치 보존 WSOLA). Rust `apply_speed`와 동일 알고리즘.
    audio: 1D numpy 배열. 출력 길이 = round(입력 길이 / speed). 범위 0.5~2.0."""
    import numpy as np

    a = np.asarray(audio, dtype=np.float32).ravel()
    if abs(speed - 1.0) < 1e-6:
        return a
    if not 0.5 <= speed <= 2.0:
        raise ValueError(f"speed must be in 0.5..=2.0, got {speed}")
    n = a.size
    target = max(1, round(n / speed))

    def _linear():  # 짧은 입력 폴백 (피치 변하지만 길이가 무시 가능)
        pos = np.arange(target) * speed
        i0 = np.floor(pos).astype(int).clip(0, n - 1)
        i1 = np.minimum(i0 + 1, n - 1)
        return (a[i0] * (1 - (pos - i0)) + a[i1] * (pos - i0)).astype(np.float32)

    if n < 2048:
        return _linear()
    sr = max(1000, int(sample_rate))
    win = max(64, sr * 30 // 1000)   # 분석창 30ms
    hop = win // 2                    # 출력 홉 (50% 겹침)
    tol = max(8, sr * 10 // 1000)     # 정합 탐색 ±10ms
    if n < win + 2 * tol + hop:
        return _linear()

    w = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)).astype(np.float64)
    out = np.zeros(target + win, dtype=np.float64)
    wsum = np.zeros(target + win, dtype=np.float64)
    # 창 에너지를 누적합으로 O(1) 조회 (정규화 상관의 분모)
    csum = np.concatenate([[0.0], np.cumsum(a.astype(np.float64) ** 2)])
    prev_next = 0
    k = 0
    while True:
        out_pos = k * hop
        nominal = int(out_pos * speed)
        # 공칭 위치는 항상 speed 배로만 전진한다. 정합 위치를 되먹이면 드리프트가
        # 누적되어 길이가 틀어진다.
        if out_pos + win > out.size or nominal + win + tol + hop >= n:
            break
        if k == 0:
            src = nominal
        else:
            tmpl = a[prev_next:prev_next + win].astype(np.float64)
            lo, hi = max(0, nominal - tol), min(n - win, nominal + tol)
            cand = np.lib.stride_tricks.sliding_window_view(a[lo:hi + win], win)[:hi - lo + 1]
            e = np.sqrt(csum[lo + win:hi + win + 1] - csum[lo:hi + 1])
            src = lo + int(np.argmax(cand @ tmpl / (e + 1e-9)))
        out[out_pos:out_pos + win] += a[src:src + win] * w
        wsum[out_pos:out_pos + win] += w
        prev_next = src + hop
        k += 1

    nz = wsum > 1e-3
    out[nz] /= wsum[nz]
    out = out[:target]
    if out.size < target:
        out = np.pad(out, (0, target - out.size))
    return out.astype(np.float32)


def synthesize(lang: str, text: str, voice: str, out: str, threads: int = 8,
               quantize: bool = True, seed: int = 0, config: str | None = None,
               speed: float = 1.0) -> dict:
    import torch
    import scipy.io.wavfile
    from pocket_tts import TTSModel

    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)

    if config:
        model = TTSModel.load_model(config=config, quantize=quantize)
    elif lang == "ko":
        model = TTSModel.load_model(config=KOREAN_CONFIG, quantize=quantize)
    else:
        # 영어 신모델. gated면 HF_TOKEN + 이용약관 동의 필요.
        model = TTSModel.load_model(language="english", quantize=quantize)
    model.eval()
    vs = model.get_state_for_audio_prompt(voice)
    sr = model.sample_rate

    torch.manual_seed(seed)
    t0 = time.perf_counter()
    audio = model.generate_audio(vs, text, copy_state=True)
    gen = time.perf_counter() - t0

    wav = audio.detach().cpu().float().numpy().ravel()
    wav = apply_speed(wav, speed, sr)
    secs = wav.size / sr
    scipy.io.wavfile.write(out, sr, wav)
    return {"out": out, "audio_s": round(secs, 2), "gen_s": round(gen, 3),
            "speed_x": round(secs / gen, 2), "rtf": round(gen / secs, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Pocket-TTS Korean/English minimal synthesis")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--text", default=None)
    ap.add_argument("--voice", default=None,
                    help="ko: 화자 wav 경로 / en: stock 이름(alba 등) 또는 wav 경로")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-quant", action="store_true", help="INT8-dyn 대신 FP32 사용")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None,
                    help="커스텀 yaml (로컬 경로 또는 hf:// URL). 지정 시 자동 선택 대신 사용")
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT,
                    help="로컬 모델 폴더 (default: models). 없으면 HF 자동 다운로드")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="재생 속도 배율 (1.0 원본, >1 빠름, <1 느림; 피치 보존, 0.5~2.0)")
    a = ap.parse_args()

    config, source = resolve_model(a.lang, a.model_dir, a.config)
    print(f"model source: {source}" + (f" ({config})" if config else ""))
    voice = resolve_voice(a.lang, a.voice, a.model_dir)
    if not voice.startswith("hf://") and not Path(voice).exists():
        raise SystemExit(f"voice not found: {voice}")
    r = synthesize(a.lang, a.text or DEFAULT_TEXT[a.lang], voice, a.out,
                   threads=a.threads, quantize=not a.no_quant, seed=a.seed,
                   config=config, speed=a.speed)
    print(f"saved {r['out']} ({r['audio_s']}s audio, gen {r['gen_s']}s, "
          f"{r['speed_x']}x, RTF {r['rtf']})")


if __name__ == "__main__":
    main()
