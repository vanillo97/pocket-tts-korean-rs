# pocket-tts-essential — 한/영 합성 (독립실행)

> Forked from [babybirdprd/pocket-tts](https://github.com/babybirdprd/pocket-tts)
> (itself a Rust/Candle port of [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)).
> 한국어(24L teacher) 지원을 추가한 변형이다. MIT License — `LICENSE` 참조.

이 폴더만 복사해도 빌드·실행된다. 원본(`babybirdprd/pocket-tts` 포크) 바깥을
참조하는 path 의존·상대경로가 없다. 추론 코어는 `libs/pocket-tts`에 벤더링했고,
서버·Web UI·WASM·벤치·ONNX/OV 하네스는 걷어냈다.

- 한국어: `seastar105/pocket-tts-korean-300m` (bundled `korean` 24L, temp 0.3, 토큰 불필요)
- 영어: `kyutai/pocket-tts` 신모델 (bundled `english` 6L, temp 0.3,
  pip `language="english"`과 동일 가중치. gated라 `HF_TOKEN` + HF 이용약관 동의 필요)
- 설정: `libs/pocket-tts/config/*.yaml` (컴파일 시점에 번들 탐색)
- 가중치·토크나이저·stock voice: 첫 실행 시 HF에서 자동 다운로드 후 캐시

## 성능 요약 (원본 포크에서 동일 조건 실측, 한국어 기준)

공통: `"안녕하세요. 한국어 음성 합성 모델입니다."` + 화자 wav, e2e 타이밍.

| 경로 | 조건 | gen | 속도 | 비고 |
|---|---|---|---|---|
| Python torch | CPU 8T, INT8-dyn | 0.81s | 5.44x | 전체 최속, 제품 1순위 |
| Rust candle (`tts`, MKL 빌드) | CPU 8T, FP32 | 1.81s | 2.44x | Python 없이 최고속 |

권장 스레드: **8T** (24T는 오버서브스크립션으로 더 느림). NPU·1024ctx·GPU 단독은 비권장.

## 구성

```
pocket-tts-essential/
  src/lib.rs            # load_korean/load_english, resolve_voice, synthesize(_to_wav)
  src/main.rs           # CLI: tts --lang ko|en
  Cargo.toml            # pocket-tts = { path = "libs/pocket-tts" } (내부 경로만)
  libs/pocket-tts/      # 추론 코어 벤더링 (src + config + standalone Cargo.toml)
  models/               # 로컬 모델 번들 (HF 캐시 하드링크, 추가 디스크 0)
    korean.safetensors / korean.tokenizer.model / korean.local.yaml
    english.safetensors / english.tokenizer.model / english.local.yaml
    alba.safetensors    # 영어 stock voice 임베딩
  synthesize.py         # Python 최소 합성 (--lang ko|en, INT8 기본)
  fetch_models.py       # models/ 고정 스크립트 (캐시 있으면 하드링크, 없으면 다운로드)
  requirements.txt      # pocket-tts==3.1.0, torch, scipy
```

## 모델 자동 선택 (로컬 우선, 없으면 다운로드)

`--config`를 따로 주지 않으면 `models/{korean,english}.local.yaml`이
완전하면(가중치·토크나이저 실존) 로컬을 쓰고, 없으면 HF에서 자동 다운로드한다.
stock voice도 `models/{이름}.safetensors` 우선. 실행 시 `model source:`로 표시.

```bash
python synthesize.py --lang ko --voice voice.wav --text "안녕하세요." --out out.wav
# model source: local (models/korean.local.yaml)   ← 로컬 있음
# model source: download                            ← 로컬 없음, HF 자동 다운로드

# 새 머신에서 models/ 미리 받기 (이 폴더에서 실행)
python fetch_models.py            # 한+영 전체
python fetch_models.py --lang ko  # 한국어만
```

## 로컬 모델로 오프라인 실행
`models/*.local.yaml`은 가중치·토크나이저를 전부 로컬 경로로 가리키므로
네트워크 없이 동작한다. 이 폴더에서 실행할 것 (상대경로 기준).

```bash
# Python (HF_HUB_OFFLINE=1: 네트워크 접근 시도시 에러 → 오프라인 보증)
HF_HUB_OFFLINE=1 python synthesize.py --lang ko --config models/korean.local.yaml \
  --voice voice.wav --text "안녕하세요." --out out.wav
HF_HUB_OFFLINE=1 python synthesize.py --lang en --config models/english.local.yaml \
  --voice models/alba.safetensors --text "Hello world!" --out out_en.wav

# Rust (로컬 경로는 다운로드를 타지 않음 — 소스상 확정)
cargo run --release -- --lang ko --config models/korean.local.yaml \
  --voice voice.wav --text "안녕하세요." --out out.wav
```

다른 머신으로 옮길 땐 `models/` 폴더만 통째로 복사하면 된다.
캐시에서 다시 묶으려면 각 파일을 `models/`에 복사한 뒤
`*.local.yaml`의 `weights_path`·`tokenizer_path`만 로컬 경로로 고치면 된다.

## 사용법

화자 음성 `voice.wav` (10초 이내, 본인 동의 음성만)를 이 폴더에 준비한다.
앞뒤 무음을 자르고 6~10초 낭독 구간만 쓰는 것을 권장한다.
28초 같은 긴 원본을 그대로 쓰면 원본의 휴지 패턴을 따라가
문장 중간에 멈추는(false EOS) 경우가 있다. 영어는 stock voice 이름(`alba`, `marius`, `javert`, `jean`, `fantine`,
`cosette`, `eponine`, `azelma`)도 바로 쓸 수 있다. stock 임베딩은 신모델용
`languages/english/embeddings/*.safetensors`를 쓴다.

### 화자 임베딩 미리 만들기 (`make_voice.py`)

wav를 `--voice`용 `.safetensors`(`audio_prompt` 잠재)로 변환한다.
28초 스테레오 원본 10.9MB → keep 1.4MB / trim 1.3MB 수준으로 줄고,
합성 시작 시 Mimi 인코딩을 생략하므로 로딩도 빨라진다.
임베딩은 만든 모델(한/영)에서만 쓴다.

```bash
# 1) 원본 휴지 패턴 그대로
python make_voice.py --in voice_raw.wav --out voice.keep.safetensors --mode keep
# 2) 앞뒤 무음 제거 + 긴 내부 휴지 압축 (권장, --max-pause 초 단위)
python make_voice.py --in voice_raw.wav --out voice.trim.safetensors --mode trim
cargo run --release -- --lang ko --voice voice.trim.safetensors \
  --text "안녕하세요." --out out.wav
```

### A. Python (가장 빠름, 권장)

```bash
pip install -r requirements.txt
# 한국어
python synthesize.py --lang ko --voice voice.wav \
  --text "안녕하세요. 한국어 음성 합성 모델입니다." --out out.wav
# 영어 (gated면 export HF_TOKEN 필요)
python synthesize.py --lang en --voice alba \
  --text "Hello world!" --out out_en.wav
# FP32가 필요하면: --no-quant
```

### B. Rust (단일 바이너리, Python 없이 배포)

```bash
# 한국어
cargo run --release -- --lang ko --voice voice.wav \
  --text "안녕하세요. 한국어 음성 합성 모델입니다." --out out.wav
# 영어 (export HF_TOKEN="hf_..." 필수)
cargo run --release -- --lang en --voice alba \
  --text "Hello world!" --out out_en.wav
# 최고속 MKL 빌드: cargo build --release --features pocket-tts/mkl
```

라이브러리 임베드:

```rust
pocket_tts_essential::set_threads(8);
let model = pocket_tts_essential::load_english(None)?; // 또는 load_korean(None)
let secs = pocket_tts_essential::synthesize_to_wav(&model, "alba", "Hello world!", "out.wav")?;
```

## 제품 적용 시 제약
- 입력 10초 분량 초과 시 `ScatterElementsUpdate` 실패 → 문장 단위로 나눠서 호출.
- `ctx 256 / mimi 128 latents` 한도.
- `--voice`: stock 이름(영어 8종), `.wav`(Mimi 인코딩), `.safetensors`(임베딩) 지원.
- 중간에 끊기면 `--eos-debounce 3` → `--extra-frames 20` → `--seed` 변경 순으로 시도.
  그래도 안 되면 화자 wav 구간을 바꿀 것 (휴지가 긴 구간은 false EOS 유발).
- 본인 동의 음성만 사용. 출력 24kHz wav.

## 원본에서 걷어낸 것

`serve`/Web UI·`wasm`/데모, `test-assets`, Python 바인딩,
core의 benches/tests/examples. `quantized`/`metal`/`mkl` 피처 선언은 유지.
