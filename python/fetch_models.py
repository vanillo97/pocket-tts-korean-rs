#!/usr/bin/env python3
"""fetch_models.py — HF에서 모델 파일을 받아 models/에 고정한다.

캐시에 있으면 하드링크(순간 완료, 추가 디스크 0), 없으면 다운로드 후 고정.
마지막에 models/{ko,en}.local.yaml을 (재)생성한다.

    python python/fetch_models.py
    python python/fetch_models.py --lang ko        # 한국어만
    python python/fetch_models.py --copy           # 하드링크 대신 복사 (다른 FS로 옮길 때)
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 리포 루트 (python/ 의 상위)
MODEL_DIR = ROOT / "models"

FILES = {
    "ko": [
        ("seastar105/pocket-tts-korean-300m", "model.safetensors",
         "df328c817a02866f20a6f74e5183e0a1fc6f6435", "korean.safetensors"),
        ("seastar105/pocket-tts-korean-300m", "tokenizer.model",
         "df328c817a02866f20a6f74e5183e0a1fc6f6435", "korean.tokenizer.model"),
    ],
    "en": [
        ("kyutai/pocket-tts-without-voice-cloning", "languages/english/model.safetensors",
         "d29db7978e464fb90cb3359ee0c69a273b9142cc", "english.safetensors"),
        ("kyutai/pocket-tts-without-voice-cloning", "languages/english/tokenizer.model",
         "d29db7978e464fb90cb3359ee0c69a273b9142cc", "english.tokenizer.model"),
        ("kyutai/pocket-tts-without-voice-cloning", "languages/english/embeddings/alba.safetensors",
         "e81d79e8194ad4c7ce879c87a4258ef20cbf2487", "alba.safetensors"),
    ],
}


def pin_file(src: Path, dest: Path, copy: bool) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        return "exists"
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    if copy:
        shutil.copyfile(src, tmp)
    else:
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copyfile(src, tmp)
    os.replace(tmp, dest)
    return "copied" if copy else "linked"


def write_local_yaml(lang: str) -> None:
    variant = lang_map(lang)
    base = (ROOT / "libs" / "pocket-tts" / "config" / f"{variant}.yaml").read_text()
    base = re.sub(r"^weights_path: .*$", f"weights_path: models/{variant}.safetensors",
                  base, flags=re.M)
    base = re.sub(r"^(\s*)tokenizer_path: .*$", rf"\1tokenizer_path: models/{variant}.tokenizer.model",
                  base, flags=re.M)
    (MODEL_DIR / f"{variant}.local.yaml").write_text(base)


def lang_map(lang: str) -> str:
    return {"ko": "korean", "en": "english"}[lang]


def main() -> None:
    ap = argparse.ArgumentParser(description="Pin HF models into models/")
    ap.add_argument("--lang", choices=["ko", "en"], default=None)
    ap.add_argument("--copy", action="store_true", help="하드링크 대신 복사")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download

    langs = [a.lang] if a.lang else ["ko", "en"]
    for lang in langs:
        for repo, filename, rev, destname in FILES[lang]:
            src = Path(hf_hub_download(repo_id=repo, filename=filename, revision=rev))
            how = pin_file(src, MODEL_DIR / destname, a.copy)
            print(f"[{lang}] {destname}: {how} ({src.stat().st_size / 1e6:.1f}MB)")
        write_local_yaml(lang)
        print(f"[{lang}] {lang_map(lang)}.local.yaml written")
    print("done.")


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        raise SystemExit("python 3.10+ required")
    main()
