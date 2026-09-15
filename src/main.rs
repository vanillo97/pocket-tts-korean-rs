//! tts: 한/영 합성 최소 CLI (`generate`의 제품용 축소판).
//!
//! ```bash
//! # 한국어: voice.wav를 이 폴더에 준비 (10초 이내, 본인 동의 음성)
//! cargo run --release -- --lang ko --voice voice.wav \
//!   --text "안녕하세요. 한국어 음성 합성 모델입니다." --out out.wav
//! # 영어: stock voice 사용 (HF_TOKEN 필요)
//! cargo run --release -- --lang en --voice alba \
//!   --text "Hello world!" --out out_en.wav
//! ```

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use pocket_tts_essential::{
    find_local_config, load_english_with_params, load_korean_with_params, set_threads,
    synthesize_to_wav_in, ENGLISH_EOS_THRESHOLD, KOREAN_EOS_THRESHOLD, MODEL_DIR_DEFAULT,
};

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum, Debug)]
enum Lang {
    /// 한국어 (`korean` 24L teacher, temp 0.3, 토큰 불필요)
    Ko,
    /// 영어 (신모델 `english` 6L, temp 0.3, `HF_TOKEN` 필요)
    En,
}

#[derive(Parser, Debug)]
#[command(name = "tts", about = "Pocket-TTS Korean/English minimal synthesis")]
struct Args {
    /// 합성 언어 (10초 분량 초과 시 문장을 나눠서 호출)
    #[arg(long, value_enum, default_value_t = Lang::Ko)]
    lang: Lang,
    /// 합성할 텍스트 (생략 시 언어별 기본 문구)
    #[arg(long)]
    text: Option<String>,
    /// 화자 지정: ko는 `.wav` 경로, en은 stock 이름(alba 등) 또는 `.wav` 경로
    #[arg(long)]
    voice: Option<String>,
    /// 출력 wav 경로
    #[arg(long, default_value = "out.wav")]
    out: String,
    /// 번들 variant 대신 쓸 설정 (로컬 경로 또는 hf:// URL).
    /// 생략 시 models/{korean,english}.local.yaml 우선, 없으면 HF 자동 다운로드.
    #[arg(long)]
    config: Option<String>,
    /// 로컬 모델 폴더 (default: models)
    #[arg(long, default_value = MODEL_DIR_DEFAULT)]
    model_dir: String,
    /// CPU 스레드 수 (권장 8; MKL 빌드에서 1T→8T 약 2배)
    #[arg(long, default_value_t = 8)]
    threads: usize,
    /// 재현용 시드 (고정 시 동일 wav)
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// 샘플링 온도 (높을수록 다양, 생략 시 yaml 기본값 0.3)
    #[arg(long, allow_negative_numbers = true)]
    temperature: Option<f32>,
    /// EOS 임계값 (더 음수일수록 길게 생성, 잘리면 -6~-10 시도)
    #[arg(long, allow_negative_numbers = true)]
    eos_threshold: Option<f32>,
    /// EOS 이후 추가로 생성할 프레임 (조기종료 진단/복구용, 12.5프레임=1초)
    #[arg(long, default_value_t = 0)]
    extra_frames: usize,
    /// 연속 EOS 필요 횟수 (단발성 오검출 무시, 기본 1)
    #[arg(long, default_value_t = 1)]
    eos_debounce: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    set_threads(args.threads);

    let (default_text, default_voice) = match args.lang {
        Lang::Ko => ("안녕하세요. 한국어 음성 합성 모델입니다.", "voice.wav"),
        Lang::En => ("Hello world! I am Pocket TTS, running in Rust.", "alba"),
    };
    let text = args.text.as_deref().unwrap_or(default_text);
    let voice = args.voice.as_deref().unwrap_or(default_voice);

    let t0 = std::time::Instant::now();
    let lang_name = match args.lang {
        Lang::Ko => "korean",
        Lang::En => "english",
    };
    // 로컬 번들 우선, 없으면 HF 자동 다운로드.
    let local = if args.config.is_none() {
        find_local_config(lang_name, &args.model_dir)
    } else {
        None
    };
    if let Some(p) = &local {
        println!("model source: local ({})", p.display());
    } else if args.config.is_some() {
        println!(
            "model source: explicit ({})",
            args.config.as_deref().unwrap()
        );
    } else {
        println!("model source: download (HF)");
    }
    let cfg_override = args
        .config
        .as_deref()
        .or_else(|| local.as_ref().and_then(|p| p.to_str()));
    let mut model = match args.lang {
        Lang::Ko => load_korean_with_params(
            cfg_override,
            args.temperature,
            args.eos_threshold.unwrap_or(KOREAN_EOS_THRESHOLD),
        )
        .context("한국어 모델 로드 실패")?,
        Lang::En => load_english_with_params(
            cfg_override,
            args.temperature,
            args.eos_threshold.unwrap_or(ENGLISH_EOS_THRESHOLD),
        )
        .context(
            "영어 모델 로드 실패 (gated 가중치: HF_TOKEN 환경변수 + HF 이용약관 동의 확인)",
        )?,
    };
    model.seed = Some(args.seed);
    model.extra_frames = args.extra_frames;
    model.eos_debounce = args.eos_debounce;
    // voice 인코딩은 타이머 밖(웜 상태 가정). 벤치 프로토콜과 동일.
    let secs = synthesize_to_wav_in(&model, voice, text, &args.out, Some(&args.model_dir))
        .with_context(|| format!("합성 실패 (voice='{}')", voice))?;
    let gen = t0.elapsed().as_secs_f32();

    println!(
        "saved {} ({:.2}s audio, gen {:.2}s, {:.2}x, RTF {:.3})",
        args.out,
        secs,
        gen,
        secs / gen,
        gen / secs
    );
    Ok(())
}
