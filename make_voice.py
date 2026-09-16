#!/usr/bin/env python3
"""make_voice.py — 화자 wav를 --voice용 임베딩(.safetensors)으로 변환한다.

Rust(`--voice xxx.safetensors`)가 읽는 `audio_prompt` 잠재 형식을 저장한다.
10초 기준 약 16KB로, wav(24kHz mono int16 10초 ≈ 480KB)보다 ~30배 작다.
Mimi 인코딩을 미리 해 두므로 합성 시작도 빨라진다.

주의: 임베딩은 인코더 가중치에 종속되므로 만든 모델(한/영)에서만 쓴다.

    # 1) 원본 휴지 패턴 그대로
    python make_voice.py --in voice_raw.wav --out voice.keep.safetensors --mode keep
    # 2) 앞뒤 무음 제거 + 긴 내부 휴지 압축 (권장)
    python make_voice.py --in voice_raw.wav --out voice.trim.safetensors --mode trim
    # 3) 길이 상한까지 (5~10초 권장). 단어 중간이 아니라 휴지에서 끊는다
    python make_voice.py --in voice_raw.wav --out voice.10s.safetensors --max-seconds 10

    cargo run --release -- --lang ko --voice voice.trim.safetensors \
      --text "안녕하세요." --out out.wav
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synthesize import KOREAN_CONFIG, resolve_model  # noqa: E402

SAMPLE_RATE = 24000


def read_mono_24k(path: str):
    """wav → mono 24kHz float32. 스테레오는 평균, 리샘플은 polyphase."""
    import numpy as np
    import scipy.io.wavfile
    from scipy.signal import resample_poly

    sr, d = scipy.io.wavfile.read(path)
    a = np.asarray(d, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=1)
    if sr != SAMPLE_RATE:
        import math
        g = math.gcd(sr, SAMPLE_RATE)
        a = resample_poly(a, SAMPLE_RATE // g, sr // g).astype(np.float64)
    peak = np.abs(a).max()
    if peak > 0:
        a = a / peak * 0.89
    return a.astype(np.float32)


def split_silence(a, thresh_db: float = -40.0, frame_ms: float = 10.0):
    """10ms 프레임 RMS가 임계값 미만이면 무음으로 판정. 마스크 배열 반환."""
    import numpy as np

    frame = max(1, int(SAMPLE_RATE * frame_ms / 1000))
    n = len(a) // frame
    rms = np.sqrt((a[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    thresh = 10.0 ** (thresh_db / 20.0)
    mask = np.repeat(rms >= thresh, frame)
    return np.pad(mask, (0, len(a) - len(mask)), constant_values=False)


def trim_edges(a, thresh_db: float = -40.0):
    voiced = split_silence(a, thresh_db)
    idx = voiced.nonzero()[0]
    if len(idx) == 0:
        raise SystemExit("no speech detected: threshold too high or file silent")
    return a[idx[0]: idx[-1] + 1]


def cap_pauses(a, max_pause_s: float = 0.4, thresh_db: float = -40.0):
    """내부 무음 구간을 각각 최대 max_pause_s까지만 남긴다. (제거량, 결과) 반환."""
    import numpy as np

    voiced = split_silence(a, thresh_db)
    out = [a[0:1]]
    removed = 0
    i = 1
    n = len(a)
    while i < n:
        if voiced[i]:
            out.append(a[i: i + 1])
            i += 1
        else:
            j = i
            while j < n and not voiced[j]:
                j += 1
            keep = min(j - i, int(max_pause_s * SAMPLE_RATE))
            out.append(a[i: i + keep])
            removed += (j - i) - keep
            i = j
    return removed / SAMPLE_RATE, np.concatenate(out)


def cap_length(a, max_s: float, thresh_db: float = -40.0):
    """앞에서부터 max_s 초까지만 남긴다.

    단어 중간이 잘리면 프롬프트 끝에 파열음 조각이 남으므로, 목표 직전의
    마지막 '발화→무음' 전환점에서 끊는다. 다만 그 지점이 목표의 70%에도
    못 미치면 버리는 양이 너무 커서, 그 때는 목표 지점에서 그냥 자른다.
    """
    import numpy as np

    limit = int(max_s * SAMPLE_RATE)
    if len(a) <= limit:
        return a
    v = split_silence(a, thresh_db)[:limit]
    ends = np.nonzero(v[:-1] & ~v[1:])[0] + 1  # 발화가 끝나는 지점들
    cut = int(ends[-1]) if len(ends) and ends[-1] >= limit * 0.7 else limit
    return a[:cut]


def encode_prompt(audio, lang: str, model_dir: str, config: str | None):
    """Mimi 인코딩 + speaker 투영 → [1, T, 1024] audio_prompt 텐서.

    Rust `get_voice_state_from_prompt_tensor`는 투영을 하지 않고 그대로
    FlowLM에 넣으므로(pip `_encode_audio`와 동일한) 투영 후 형식을 저장한다.
    BOS도 저장하지 않는다: Rust가 prepend한다.
    """
    import torch
    from pocket_tts import TTSModel

    cfg, source = resolve_model(lang, model_dir, config)
    print(f"model source: {source}" + (f" ({cfg})" if cfg else ""))
    torch.set_grad_enabled(False)
    # cfg=None을 그대로 넘기면 load_model이 English로 기본 동작해 --lang이 무시된다.
    # 그러면 gated English 대신 without-voice-cloning 가중치로 떨어지고, 그 체크포인트는
    # Mimi 인코더가 없어 임베딩이 전부 0이 된다. synthesize.py와 같은 분기를 쓴다.
    if cfg:
        model = TTSModel.load_model(config=cfg, quantize=False)
    elif lang == "ko":
        model = TTSModel.load_model(config=KOREAN_CONFIG, quantize=False)
    else:
        model = TTSModel.load_model(language="english", quantize=False)
    model.eval()
    enc_max = max(
        float(p.float().abs().max()) for n, p in model.mimi.named_parameters() if "encoder" in n
    )
    if enc_max == 0.0:
        raise SystemExit(
            "Mimi 인코더 가중치가 비어 있어 임베딩이 0이 된다. "
            "voice cloning이 가능한 체크포인트인지 확인할 것 "
            "(without-voice-cloning 가중치에는 인코더가 없다)."
        )
    with torch.no_grad():
        prompt = model._encode_audio(
            torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(model.device)
        )
    return prompt.detach().cpu().float().contiguous()


def main() -> None:
    ap = argparse.ArgumentParser(description="wav → voice embedding (.safetensors)")
    ap.add_argument("--in", dest="inp", required=True, help="입력 화자 wav")
    ap.add_argument("--out", dest="out", required=True, help="출력 .safetensors")
    ap.add_argument("--mode", choices=["keep", "trim"], default="trim",
                    help="keep: 휴지 그대로 / trim: 앞뒤 무음 제거+긴 휴지 압축")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--max-pause", type=float, default=0.4,
                    help="trim 모드에서 내부 무음 상한(초)")
    ap.add_argument("--thresh-db", type=float, default=-40.0, help="무음 판정 임계(dBFS)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="프롬프트 길이 상한(초). 생략 시 제한 없음. 5~10초 권장: "
                         "그 이상 늘려도 화자 유사도는 포화되고 합성만 느려진다")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-dir", default="models")
    a = ap.parse_args()
    if a.max_seconds is not None and a.max_seconds <= 0:
        raise SystemExit(f"--max-seconds must be positive (got {a.max_seconds})")

    audio = read_mono_24k(a.inp)
    print(f"input: {len(audio) / SAMPLE_RATE:.2f}s @ {a.inp}")
    if a.mode == "trim":
        audio = trim_edges(audio, a.thresh_db)
        removed, audio = cap_pauses(audio, a.max_pause, a.thresh_db)
        print(f"trim: edges cut, internal pauses shortened by {removed:.2f}s"
              f" → {len(audio) / SAMPLE_RATE:.2f}s")
    else:
        print(f"keep: pauses preserved → {len(audio) / SAMPLE_RATE:.2f}s")

    if a.max_seconds is not None:
        before = len(audio) / SAMPLE_RATE
        audio = cap_length(audio, a.max_seconds, a.thresh_db)
        after = len(audio) / SAMPLE_RATE
        if after < before:
            print(f"cap: {before:.2f}s → {after:.2f}s (limit {a.max_seconds:.2f}s)")
        else:
            print(f"cap: already within {a.max_seconds:.2f}s, unchanged")

    prompt = encode_prompt(audio, a.lang, a.model_dir, a.config)
    print(f"audio_prompt: {tuple(prompt.shape)} (~{prompt.numel() * 4 / 1024:.0f}KB)")

    from safetensors.torch import save_file
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({"audio_prompt": prompt}, str(out))
    print(f"saved {out} ({out.stat().st_size / 1024:.0f}KB)")


if __name__ == "__main__":
    main()
