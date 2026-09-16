//! pocket-tts-essential: 한/영 합성 최소 라이브러리 (독립실행).
//!
//! 추론 코어(`libs/pocket-tts`, 원본 `pocket-tts`의 core 크레이트 벤더링)를
//! 내부에 포함하므로 이 폴더만 복사해도 빌드·실행된다. 외부 path 의존 없음.
//! - 한국어: `seastar105/pocket-tts-korean-300m` (bundled `korean` 24L, temp 0.3, 토큰 불필요)
//! - 영어: `kyutai/pocket-tts` 신모델 (bundled `english` 6L, temp 0.3,
//!   pip `language="english"`과 동일. gated라 `HF_TOKEN` + 이용약관 동의 필요)
//! - 설정: `libs/pocket-tts/config/*.yaml` (컴파일 시점에 번들 탐색)
//! - 가중치·토크나이저: 첫 실행 시 HF에서 자동 다운로드 후 캐시
//! - 음성: `.wav` 보이스 클로닝 + 영어 stock voice (`alba` 등 8종)
//! - 제외: serve/web-ui, wasm, bench, onnx-bench, synth-rs(OV IR)
//!
//! ```no_run
//! let model = pocket_tts_essential::load_korean(None)?;
//! let vs = pocket_tts_essential::resolve_voice(&model, "voice.wav")?;
//! let audio = pocket_tts_essential::synthesize(&model, &vs, "안녕하세요.")?;
//! pocket_tts_essential::save_wav("out.wav", &audio, model.sample_rate as u32)?;
//! # anyhow::Ok(())
//! ```

use anyhow::Result;
use candle_core::Tensor;
use pocket_tts::config::load_config;
use pocket_tts::weights::download_if_necessary;
use pocket_tts::{ModelState, TTSModel};

/// 한국어 권장 생성 파라미터 (벤치 기준값).
pub const KOREAN_LSD_STEPS: usize = 1;
pub const KOREAN_EOS_THRESHOLD: f32 = -4.0;
/// HF 공개 설정. 토큰 불필요.
pub const KOREAN_CONFIG_HF: &str = "hf://seastar105/pocket-tts-korean-300m/korean.yaml";

/// 로컬 모델 폴더 기본값 (이 폴더에서 실행 기준).
pub const MODEL_DIR_DEFAULT: &str = "models";

/// 영어 variant (신모델 6L, 한국어 teacher와 동일 계열: insert_bos + inner/outer dim).
/// pip `TTSModel.load_model(language="english")`와 같은 가중치.
pub const ENGLISH_VARIANT: &str = "english";
pub const ENGLISH_LSD_STEPS: usize = 1;
pub const ENGLISH_EOS_THRESHOLD: f32 = -4.0;

/// 영어 stock voice 8종 (신모델용 per-language 임베딩 사용).
pub const PREDEFINED_ENGLISH_VOICES: &[&str] = &[
    "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
];
/// stock voice 임베딩 저장소. revision은 pip 3.1.0 `get_predefined_voice`와 동일.
const STOCK_VOICE_REPO: &str = "kyutai/pocket-tts-without-voice-cloning";
const STOCK_VOICE_PATH_PREFIX: &str = "languages/english/embeddings";
const STOCK_VOICE_REVISION: &str = "e81d79e8194ad4c7ce879c87a4258ef20cbf2487";

/// RAYON/MKL/OMP 스레드 고정. candle gemm가 RAYON을 참조하므로 셋 다 설정해야
/// `--threads`가 실제로 반영된다 (BENCHMARK_REPORT_RS §재측정 변경점 ②).
pub fn set_threads(n: usize) {
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", n.to_string());
        std::env::set_var("OMP_NUM_THREADS", n.to_string());
        std::env::set_var("MKL_NUM_THREADS", n.to_string());
    }
}

/// 한국어 모델 로드. `config=None`이면 바이너리에 번들된 `korean` variant 사용.
/// `temp=None`이면 yaml의 `default_temperature`(0.3) 사용.
pub fn load_korean(config: Option<&str>) -> Result<TTSModel> {
    load_korean_with_params(config, None, KOREAN_EOS_THRESHOLD)
}

/// 한국어 모델 로드 (파라미터 오버라이드).
/// - `temp=None`이면 yaml의 `default_temperature` 사용
/// - `eos_threshold`가 낮을수록(더 음수) 길게 생성 (기본 -4.0, 잘리면 -6~-10 시도)
pub fn load_korean_with_params(
    config: Option<&str>,
    temp: Option<f32>,
    eos_threshold: f32,
) -> Result<TTSModel> {
    let cfg = if let Some(c) = config {
        let p = download_if_necessary(c)?;
        load_config(&p)?
    } else {
        let p = TTSModel::config_path_for_variant("korean")?;
        load_config(&p)?
    };
    TTSModel::load_from_config(
        cfg,
        temp,
        KOREAN_LSD_STEPS,
        eos_threshold,
        None,
        &candle_core::Device::Cpu,
    )
}

/// 영어 모델 로드. `config=None`이면 번들된 `english` variant 사용.
/// yaml의 `default_temperature`(0.3)가 적용된다.
/// gated 가중치라 `HF_TOKEN` 환경변수 + HF 이용약관 동의가 필요하다.
pub fn load_english(config: Option<&str>) -> Result<TTSModel> {
    load_english_with_params(config, None, ENGLISH_EOS_THRESHOLD)
}

/// 영어 모델 로드 (파라미터 오버라이드).
pub fn load_english_with_params(
    config: Option<&str>,
    temp: Option<f32>,
    eos_threshold: f32,
) -> Result<TTSModel> {
    let cfg = if let Some(c) = config {
        let p = download_if_necessary(c)?;
        load_config(&p)?
    } else {
        let p = TTSModel::config_path_for_variant(ENGLISH_VARIANT)?;
        load_config(&p)?
    };
    TTSModel::load_from_config(
        cfg,
        temp,
        ENGLISH_LSD_STEPS,
        eos_threshold,
        None,
        &candle_core::Device::Cpu,
    )
}

/// voice 지정자를 `ModelState`로 변환.
/// - 영어 stock voice 이름 (`alba` 등 8종): `{model_dir}/{이름}.safetensors`가
///   있으면 로컬 우선, 없으면 HF 임베딩 다운로드 후 로드
/// - `.wav` 경로: Mimi로 인코딩 (한/영 공통 보이스 클로닝)
/// - `.safetensors` 경로: 사전 계산된 임베딩 로드
pub fn resolve_voice(model: &TTSModel, spec: &str) -> Result<ModelState> {
    resolve_voice_in(model, spec, None)
}

/// `model_dir`를 함께 보는 `resolve_voice`.
pub fn resolve_voice_in(
    model: &TTSModel,
    spec: &str,
    model_dir: Option<&str>,
) -> Result<ModelState> {
    let spec = spec.trim();
    if PREDEFINED_ENGLISH_VOICES.contains(&spec) {
        if let Some(dir) = model_dir {
            let local = std::path::Path::new(dir).join(format!("{spec}.safetensors"));
            if local.is_file() {
                return model.get_voice_state_from_prompt_file(&local);
            }
        }
        let hf_path = format!(
            "hf://{}/{}/{}.safetensors@{}",
            STOCK_VOICE_REPO, STOCK_VOICE_PATH_PREFIX, spec, STOCK_VOICE_REVISION
        );
        let local = download_if_necessary(&hf_path)?;
        return model.get_voice_state_from_prompt_file(&local);
    }
    let ext = std::path::Path::new(spec)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "safetensors" => model.get_voice_state_from_prompt_file(spec),
        "wav" | "wave" => model.get_voice_state(spec),
        _ => anyhow::bail!(
            "voice '{}' not recognized. Use a stock voice ({}), a .wav file, or a .safetensors file",
            spec,
            PREDEFINED_ENGLISH_VOICES.join(", ")
        ),
    }
}

/// `{model_dir}/{lang}.local.yaml`이 있고, 가리키는 가중치·토크나이저가
/// 모두 로컬에 있으면 `Some(경로)`. 없으면 `None` (호출자가 HF 기본값 사용).
pub fn find_local_config(lang: &str, model_dir: &str) -> Option<std::path::PathBuf> {
    let yaml = std::path::Path::new(model_dir).join(format!("{lang}.local.yaml"));
    if !yaml.is_file() {
        return None;
    }
    let cfg = load_config(&yaml).ok()?;
    let weights = cfg.weights_path.as_deref()?;
    let tokenizer = cfg.flow_lm.lookup_table.tokenizer_path.as_str();
    for p in [weights, tokenizer] {
        if p.starts_with("hf://") || !std::path::Path::new(p).exists() {
            return None;
        }
    }
    Some(yaml)
}

/// 텍스트 한 건 합성. 긴 입력(문장 다수)은 내부에서 청킹 처리된다.
/// 반환: `[1, C, T]` 오디오 텐서. 재현이 필요하면 호출 전 `model.seed = Some(n)`.
///
/// 속도 조절이 필요하면 결과에 `apply_speed`를 걸면 된다 (모델 재추론 없음).
pub fn synthesize(model: &TTSModel, voice: &ModelState, text: &str) -> Result<Tensor> {
    let mut chunks = Vec::new();
    for r in model.generate_stream_long(text, voice) {
        chunks.push(r?);
    }
    if chunks.is_empty() {
        anyhow::bail!("no audio generated (text too short or invalid)");
    }
    Ok(Tensor::cat(&chunks, 2)?.squeeze(0)?)
}

/// 재생 속도 조절 (WSOLA, 무의존성).
/// - `speed == 1.0`: 그대로 반환 (clone)
/// - `speed > 1`: 짧아짐, `speed < 1`: 길어짐. 출력 길이 = `round(입력 길이 / speed)` (정확).
/// - 30ms 창을 50% 겹쳐 쌓되, 직전 프레임의 자연스러운 연속과 가장 닮은 지점을
///   ±10ms 안에서 찾아 붙이므로 파형 위상이 이어진다 → 피치 보존.
/// - 분석 위치(`nominal`)는 항상 speed 배로만 전진한다. 정합 위치를 되먹이면
///   드리프트가 누적되어 길이가 틀어진다.
/// - 모노 믹스로 정합 지점을 구해 전 채널에 동일 적용 (채널 간 위상 보존).
// ponytail: 탐색 구간 전수 상관 O(tol*win). 6.5초 오디오 ~40ms. 느리면 4x 다운샘플 탐색.
pub fn apply_speed(audio: &Tensor, speed: f32, sample_rate: usize) -> Result<Tensor> {
    if !(0.5..=2.0).contains(&speed) {
        anyhow::bail!("speed must be in 0.5..=2.0, got {speed}");
    }
    if (speed - 1.0).abs() < 1e-6 {
        return Ok(audio.clone());
    }
    let shape = audio.dims().to_vec();
    // [C, T] 또는 [T] 정규화 → 채널별 벡터 + 정합용 모노 믹스
    let (chans, mono): (Vec<Vec<f32>>, Vec<f32>) = match shape.as_slice() {
        [_, _] => {
            let v = audio.to_vec2::<f32>()?;
            if v[0].len() < 2048 {
                return resample_naive(audio, speed);
            }
            let n = v[0].len();
            let mut mix = vec![0.0f32; n];
            for ch in &v {
                for (i, s) in ch.iter().enumerate() {
                    mix[i] += *s;
                }
            }
            let inv = 1.0 / v.len() as f32;
            for s in mix.iter_mut() {
                *s *= inv;
            }
            (v, mix)
        }
        [t] => {
            let v = audio.to_vec1::<f32>()?;
            if *t < 2048 {
                return resample_naive(audio, speed);
            }
            (vec![v.clone()], v)
        }
        _ => anyhow::bail!("expected audio tensor [C, T] or [T], got {shape:?}"),
    };

    let n = mono.len();
    let sr = sample_rate.max(1000);
    let win = (sr * 30 / 1000).max(64); // 분석창 30ms
    let hop = win / 2; // 출력 홉 (50% 겹침)
    let tol = (sr * 10 / 1000).max(8); // 정합 탐색 ±10ms
    let target = ((n as f32) / speed).round().max(1.0) as usize;
    if n < win + 2 * tol + hop {
        return resample_naive(audio, speed);
    }
    let w: Vec<f32> = (0..win)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / win as f32).cos())
        .collect();

    let mut outs: Vec<Vec<f32>> = chans.iter().map(|_| vec![0.0f32; target + win]).collect();
    let mut wsum = vec![0.0f32; target + win];
    let mut prev_next = 0usize; // 직전 프레임의 자연스러운 다음 위치
    let mut k = 0usize;
    loop {
        let out_pos = k * hop;
        let nominal = (out_pos as f32 * speed) as usize;
        if out_pos + win > outs[0].len() || nominal + win + tol + hop >= n {
            break;
        }
        let src = if k == 0 {
            nominal
        } else {
            let tmpl = &mono[prev_next..prev_next + win];
            let lo = nominal.saturating_sub(tol);
            let hi = (nominal + tol).min(n - win);
            let mut best = nominal.min(hi);
            let mut best_c = f32::NEG_INFINITY;
            for d in lo..=hi {
                let mut dot = 0.0f32;
                let mut e = 0.0f32;
                for (a, b) in mono[d..d + win].iter().zip(tmpl) {
                    dot += a * b;
                    e += a * a;
                }
                // 정규화 상관: 정규화하지 않으면 에너지 큰 구간으로 끌려가 위상이 어긋난다.
                let c = dot / (e.sqrt() + 1e-9);
                if c > best_c {
                    best_c = c;
                    best = d;
                }
            }
            best
        };
        for i in 0..win {
            let oi = out_pos + i;
            for (ci, ch) in chans.iter().enumerate() {
                outs[ci][oi] += ch[src + i] * w[i];
            }
            wsum[oi] += w[i];
        }
        prev_next = src + hop;
        k += 1;
    }

    // 누적 윈도우 합으로 나눠 COLA 오차를 보정 (가장자리 페이드도 함께 복원).
    for out in outs.iter_mut() {
        for (o, s) in out.iter_mut().zip(&wsum) {
            if *s > 1e-3 {
                *o /= *s;
            }
        }
        out.truncate(target);
        if out.len() < target {
            out.resize(target, 0.0);
        }
    }

    let device = audio.device();
    match shape.as_slice() {
        [c, _] => {
            let flat: Vec<f32> = outs.into_iter().flatten().collect();
            Ok(Tensor::from_vec(flat, (*c, target), device)?)
        }
        [_] => Ok(Tensor::from_vec(std::mem::take(&mut outs[0]), target, device)?),
        _ => unreachable!(),
    }
}

/// 짧은 오디오(<2048 샘플)용 선형 리샘플 폴백.
fn resample_naive(audio: &Tensor, speed: f32) -> Result<Tensor> {
    let shape = audio.dims().to_vec();
    let device = audio.device().clone();
    match shape.as_slice() {
        [c, t] => {
            let v = audio.to_vec2::<f32>()?;
            let out_len = ((*t as f32) / speed).round().max(1.0) as usize;
            let mut out = Vec::with_capacity(c * out_len);
            for ch in &v {
                for i in 0..out_len {
                    let pos = i as f32 * speed;
                    let i0 = pos.floor() as usize;
                    let frac = pos - i0 as f32;
                    let s0 = ch[i0.min(*t - 1)];
                    let s1 = ch[(i0 + 1).min(*t - 1)];
                    out.push(s0 * (1.0 - frac) + s1 * frac);
                }
            }
            Ok(Tensor::from_vec(out, (*c, out_len), &device)?)
        }
        [t] => {
            let v = audio.to_vec1::<f32>()?;
            let out_len = ((*t as f32) / speed).round().max(1.0) as usize;
            let mut out = Vec::with_capacity(out_len);
            for i in 0..out_len {
                let pos = i as f32 * speed;
                let i0 = pos.floor() as usize;
                let frac = pos - i0 as f32;
                out.push(v[i0.min(*t - 1)] * (1.0 - frac) + v[(i0 + 1).min(*t - 1)] * frac);
            }
            Ok(Tensor::from_vec(out, out_len, &device)?)
        }
        _ => anyhow::bail!("expected audio tensor [C, T] or [T], got {shape:?}"),
    }
}

/// 합성 + wav 저장까지 한 번에. 제품에서 가장 많이 쓰는 진입점.
/// `voice_spec`: stock voice 이름, `.wav` 또는 `.safetensors` 경로.
pub fn synthesize_to_wav(
    model: &TTSModel,
    voice_spec: &str,
    text: &str,
    out_wav: &str,
) -> Result<f32> {
    synthesize_to_wav_in(model, voice_spec, text, out_wav, None, 1.0)
}

/// `model_dir`(stock voice 로컬 우선)와 `speed`를 함께 받는 전체 진입점.
/// `speed`: 1.0 = 원본 속도, 허용 범위 `0.5..=2.0` (자세한 건 `apply_speed` 참조).
pub fn synthesize_to_wav_in(
    model: &TTSModel,
    voice_spec: &str,
    text: &str,
    out_wav: &str,
    model_dir: Option<&str>,
    speed: f32,
) -> Result<f32> {
    let voice = resolve_voice_in(model, voice_spec, model_dir)?;
    let audio = apply_speed(&synthesize(model, &voice, text)?, speed, model.sample_rate)?;
    let secs = audio_len_secs(&audio, model.sample_rate);
    pocket_tts::audio::write_wav(out_wav, &audio, model.sample_rate as u32)?;
    Ok(secs)
}

/// `[C, T]` 또는 `[T]` 텐서의 재생 길이(초).
pub fn audio_len_secs(audio: &Tensor, sample_rate: usize) -> f32 {
    let n = match audio.dims() {
        [_, t] => *t,
        [t] => *t,
        _ => 0,
    };
    n as f32 / sample_rate as f32
}

/// hound 직접 저장이 필요할 때 (f32 mono).
pub fn save_wav(path: &str, audio: &Tensor, sample_rate: u32) -> Result<()> {
    pocket_tts::audio::write_wav(path, audio, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_config_missing_dir() {
        assert!(find_local_config("korean", "/nonexistent_dir_xyz").is_none());
        assert!(find_local_config("korean", "/tmp/empty_models_xyz").is_none());
    }

    #[test]
    fn local_config_bundle() {
        // models/ 번들 무결성 검사도 겸함 (CWD = 패키지 루트 기준).
        for lang in ["korean", "english"] {
            let p = find_local_config(lang, MODEL_DIR_DEFAULT)
                .expect("models/{lang}.local.yaml bundle");
            assert!(p.ends_with(format!("{lang}.local.yaml")));
        }
    }

    #[test]
    fn speed_keeps_pitch() {
        // 440Hz 순음. 제로크로싱 수로 피치 보존을 확인한다.
        // 1초 440Hz = 880 크로싱 → speed배 출력에서도 초당 880이어야 한다.
        // 리샘플이면 speed배로 변한다 (2.0x에서 1760).
        use candle_core::Device;
        let sr = 24000usize;
        let sine: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let t = Tensor::from_vec(sine, (1, sr), &Device::Cpu).unwrap();
        for speed in [0.5f32, 1.5, 2.0] {
            let y = apply_speed(&t, speed, sr)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap()
                .remove(0);
            let want_len = (sr as f32 / speed).round() as usize;
            assert_eq!(y.len(), want_len, "speed {speed}: len");
            let zc = y.windows(2).filter(|p| (p[0] < 0.0) != (p[1] < 0.0)).count();
            let want = 880.0 * y.len() as f32 / sr as f32; // 피치 보존 시 기대 크로싱
            assert!(
                (zc as f32 - want).abs() < want * 0.12,
                "speed {speed}: 제로크로싱 {zc}, 피치보존 기대 ~{want:.0} \
                 (리샘플이면 ~{:.0})",
                want * speed * speed
            );
        }
    }

    #[test]
    fn speed_scales_duration() {
        use candle_core::Device;
        use std::f32::consts::PI;
        let device = Device::Cpu;
        // 2초 음성 유사 신호 24kHz: 5ms마다 바뀌는 F0(유성음) + 무성음 잡음 버스트 +
        // 음절 게이팅. 프레임 단위로 예측 불가해야 WSOLA가 nominal 홉을 추종한다.
        // (구간 정상 톤은 NCC≈1 위상고정에 빠져 길이가 1.0으로 수렴하는 병리 케이스)
        let n = 48000;
        let sr = 24000.0f32;
        let micro = 120usize; // 5ms
        let hash = |x: usize| ((x.wrapping_mul(2654435761) >> 8) % 1000) as f32 / 1000.0;
        let mut lcg: u32 = 0x12345678;
        let mut phase = 0.0f32;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr;
            let mb = i / micro;
            let gate = (2.0 * PI * 3.7 * t).sin().max(0.0).powf(1.2);
            lcg = lcg.wrapping_mul(1103515245).wrapping_add(12345);
            let nz = ((lcg >> 16) & 0x7fff) as f32 / 32768.0 - 0.5;
            let x = if mb % 9 == 8 {
                nz * 0.5 // 무성음 버스트
            } else {
                let f0 = 90.0 + hash(mb) * 160.0;
                phase += 2.0 * PI * f0 / sr;
                phase.sin() * 0.5 + (2.0 * phase).sin() * 0.25 + nz * 0.15
            };
            data.push(gate * x * (0.6 + 0.4 * hash(mb / 7)));
        }
        let t = Tensor::from_vec(data, (1, n), &device).unwrap();
        for (speed, expect_ratio) in [(2.0, 0.5), (0.5, 2.0), (1.5, 1.0 / 1.5), (1.0, 1.0)] {
            let out = apply_speed(&t, speed, 24000).unwrap();
            let got = out.dims()[1] as f32 / n as f32;
            assert!(
                (got - expect_ratio).abs() < 0.02,
                "speed {speed}: ratio {got}, want ~{expect_ratio}"
            );
        }
        assert!(apply_speed(&t, 0.1, 24000).is_err());
        assert!(apply_speed(&t, 3.0, 24000).is_err());
    }
}
