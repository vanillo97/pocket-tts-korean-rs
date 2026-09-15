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
/// `speed`는 재생 속도 배율 (1.0 = 원본, >1 빠름, <1 느림).
/// 피치 보존 시간-늘림(`apply_speed`)으로 후처리하므로 모델 재추론 없이 동작한다.
/// 허용 범위 `0.5..=2.0` 밖이면 에러.
pub fn synthesize(model: &TTSModel, voice: &ModelState, text: &str) -> Result<Tensor> {
    synthesize_with_speed(model, voice, text, 1.0)
}

/// `speed` 지정 합성 (`synthesize`의 속도 조절판).
pub fn synthesize_with_speed(
    model: &TTSModel,
    voice: &ModelState,
    text: &str,
    speed: f32,
) -> Result<Tensor> {
    let mut chunks = Vec::new();
    for r in model.generate_stream_long(text, voice) {
        chunks.push(r?);
    }
    if chunks.is_empty() {
        anyhow::bail!("no audio generated (text too short or invalid)");
    }
    let audio = Tensor::cat(&chunks, 2)?.squeeze(0)?;
    apply_speed(&audio, speed, model.sample_rate)
}

/// 재생 속도 조절 (TD-PSOLA-lite, 무의존성).
/// - `speed == 1.0`: 그대로 반환 (clone)
/// - `speed > 1`: 짧아짐, `speed < 1`: 길어짐. 출력 길이 = 입력 길이 / speed (정확).
/// - 피치 주기 단위로 복사/건너뛰므로 피치 보존. 무성음은 10ms 고정 프레임.
/// - 모노 믹스로 피치마크를 구해 전 채널에 동일 적용 (위상 보존).
// ponytail: 고정 분석창(30ms/10ms). 정교한 피치 추적 필요하면 외부 크레이트로 교체.
pub fn apply_speed(audio: &Tensor, speed: f32, sample_rate: usize) -> Result<Tensor> {
    if !(0.5..=2.0).contains(&speed) {
        anyhow::bail!("speed must be in 0.5..=2.0, got {speed}");
    }
    if (speed - 1.0).abs() < 1e-6 {
        return Ok(audio.clone());
    }
    let shape = audio.dims().to_vec();
    // [C, T] 또는 [T] 정규화 → Vec<Vec<f32>>
    let (chans, mono_mix): (Vec<Vec<f32>>, Vec<f32>) = match shape.as_slice() {
        [_, _] => {
            let v = audio.to_vec2::<f32>()?;
            let n = v[0].len();
            if n < 2048 {
                return resample_naive(audio, speed);
            }
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

    let sr = sample_rate.max(1000);
    let win = (sr * 30 / 1000).max(64); // 피치 분석창 30ms
    let hop = (sr * 10 / 1000).max(16); // 분석 홉 10ms
    let min_p = (sr / 500).max(8); // 500Hz
    let max_p = (sr / 50).max(min_p + 1); // 50Hz
    let uv_span = (sr * 10 / 1000).max(16); // 무성음 고정 프레임 10ms
    let n = mono_mix.len();

    // 1. 프레임별 주기 추정 (0 = 무성음). 낮은 lag부터 임계값(0.5) 초과 지점 채택.
    let mut centers: Vec<(usize, usize)> = Vec::new(); // (center, period)
    if n > win + 2 * max_p {
        let mut c = max_p + win / 2;
        let end = n - win / 2 - max_p;
        while c <= end {
            let w0 = c - win / 2;
            let mut e0 = 0.0f32;
            for i in 0..win {
                e0 += mono_mix[w0 + i] * mono_mix[w0 + i];
            }
            let mut period = 0usize;
            if e0 > 1e-12 {
                let mut lag = min_p;
                while lag <= max_p {
                    let mut dot = 0.0f32;
                    let mut e1 = 0.0f32;
                    for i in 0..win {
                        let b = mono_mix[w0 + i + lag];
                        dot += mono_mix[w0 + i] * b;
                        e1 += b * b;
                    }
                    if dot / (e0 * e1).sqrt() >= 0.5 {
                        period = lag;
                        break;
                    }
                    lag += 1;
                }
            }
            centers.push((c, period));
            c += hop;
        }
    }
    let period_at = |pos: usize| -> usize {
        if centers.is_empty() {
            return 0;
        }
        let mut bi = 0;
        let mut bd = usize::MAX;
        for (i, (cw, _)) in centers.iter().enumerate() {
            let d = cw.abs_diff(pos);
            if d < bd {
                bd = d;
                bi = i;
            }
        }
        centers[bi].1
    };

    // 2. 피치마크: 주기 간격 전진 + 성대 펄스(국소 최대)에 스냅.
    let mut marks: Vec<usize> = vec![0];
    for _ in 0..(n / 8 + 64) {
        let pos = *marks.last().unwrap();
        if pos >= n {
            break;
        }
        let p = period_at(pos);
        let next = if p == 0 {
            pos + uv_span
        } else {
            // 스냅은 이웃 마크 중간을 넘지 않게 (단조성 보장)
            let r = (p / 4).max(1);
            let prev = marks.len().checked_sub(2).map(|i| marks[i]).unwrap_or(0);
            let lo = (prev + pos) / 2 + 1;
            let hi = (pos + r).min(n.saturating_sub(1));
            let mut mi = pos;
            if hi > lo && hi > pos.saturating_sub(r) {
                let s0 = lo.max(pos.saturating_sub(r));
                let mut mv = f32::NEG_INFINITY;
                for i in s0..=hi {
                    let v = mono_mix[i].abs();
                    if v > mv {
                        mv = v;
                        mi = i;
                    }
                }
            }
            mi + p
        };
        if next <= pos || next >= n + max_p {
            break;
        }
        marks.push(next.min(n));
        if *marks.last().unwrap() >= n {
            break;
        }
    }
    if *marks.last().unwrap() < n {
        marks.push(n);
    }

    // 3. 합성: 입력 시간은 speed배速으로 걷고, 출력은 동일 주기로 쌓기.
    let target = ((n as f32) / speed).round().max(1.0) as usize;
    let spans: Vec<usize> = marks
        .windows(2)
        .map(|w| (w[1] - w[0]).max(8))
        .collect();
    let max_span = spans.iter().copied().max().unwrap_or(uv_span);
    let mut outs: Vec<Vec<f32>> = chans
        .iter()
        .map(|_| vec![0.0f32; target + 2 * max_span + 8])
        .collect();
    let mut synth = spans.first().copied().unwrap_or(uv_span); // 첫 마크 중심
    let mut input_t = 0.0f32;
    let mut j = 0usize;
    let mut filled = 0usize;
    for _ in 0..(marks.len() * 2 + 16) {
        if j >= spans.len() || input_t >= n as f32 {
            break;
        }
        while j + 1 < spans.len() && (marks[j + 1] as f32) <= input_t {
            j += 1;
        }
        let s = spans[j];
        let m = marks[j].min(n);
        // Hann(2s+1) 윈도우 overlap-add (50% COLA → 진폭 보존)
        for k in 0..=2 * s {
            let si = m as i64 - s as i64 + k as i64;
            let oi = synth as i64 - s as i64 + k as i64;
            if si < 0 || si >= n as i64 || oi < 0 || oi >= outs[0].len() as i64 {
                continue;
            }
            let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / (2 * s) as f32).cos();
            for (ch_idx, ch) in chans.iter().enumerate() {
                outs[ch_idx][oi as usize] += ch[si as usize] * w;
            }
        }
        filled = filled.max(synth + s);
        synth += s;
        input_t += s as f32 * speed;
    }

    // 4. 길이를 target에 정확히 맞추기 (128샘플 페이드아웃 후 자르기/0패딩).
    let fade = 128.min(target).min(filled);
    for out in outs.iter_mut() {
        let l = filled.min(target);
        if fade > 0 && l >= fade {
            for i in 0..fade {
                out[l - fade + i] *= (fade - i) as f32 / fade as f32;
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
            let t = outs[0].len();
            let mut flat = Vec::with_capacity(c * t);
            for ch in outs {
                flat.extend(ch);
            }
            Ok(Tensor::from_vec(flat, (*c, t), device)?)
        }
        [_] => {
            let t = outs[0].len();
            Ok(Tensor::from_vec(std::mem::take(&mut outs[0]), t, device)?)
        }
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
    synthesize_to_wav_in(model, voice_spec, text, out_wav, None)
}

/// `model_dir`를 함께 보는 `synthesize_to_wav` (stock voice 로컬 우선).
/// `speed`: 1.0 = 원본 속도 (자세한 건 `apply_speed` 참조).
pub fn synthesize_to_wav_in(
    model: &TTSModel,
    voice_spec: &str,
    text: &str,
    out_wav: &str,
    model_dir: Option<&str>,
) -> Result<f32> {
    synthesize_to_wav_in_with_speed(model, voice_spec, text, out_wav, model_dir, 1.0)
}

/// `synthesize_to_wav_in`의 속도 조절판.
pub fn synthesize_to_wav_in_with_speed(
    model: &TTSModel,
    voice_spec: &str,
    text: &str,
    out_wav: &str,
    model_dir: Option<&str>,
    speed: f32,
) -> Result<f32> {
    let voice = resolve_voice_in(model, voice_spec, model_dir)?;
    let audio = synthesize_with_speed(model, &voice, text, speed)?;
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
