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
pub fn synthesize_to_wav_in(
    model: &TTSModel,
    voice_spec: &str,
    text: &str,
    out_wav: &str,
    model_dir: Option<&str>,
) -> Result<f32> {
    let voice = resolve_voice_in(model, voice_spec, model_dir)?;
    let audio = synthesize(model, &voice, text)?;
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
}
