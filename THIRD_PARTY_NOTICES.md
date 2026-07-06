# Third-Party Notices and Commercial Release Checklist

This file tracks datasets, audio sources, models, APIs, and code that may affect
commercial distribution of this Quran recitation companion. It is an engineering
checklist, not legal advice. Before selling hardware or bundled software, verify
each item with counsel and, where needed, direct permission from rights holders.

## Quranic Universal Ayahs

- Source: https://huggingface.co/datasets/hetchyy/quranic-universal-ayahs
- Current project status: candidate dataset for evaluation, timing features, and future training; not bundled in the app at this checkpoint.
- Hugging Face listed license: `cc-by-4.0`.
- Useful fields observed: ayah audio, `duration_ms`, `text_uthmani`, waqf-aware `segments`, `word_timestamps`, `letter_timestamps`, `source_url`, and `source_offset_ms`.
- Commercial note: CC BY 4.0 generally allows commercial use with attribution, but this dataset includes recitation audio derived from upstream URLs. Do not redistribute or preinstall the audio on commercial hardware until the upstream audio rights are verified independently.
- Required attribution if used: name the dataset, author/account, source URL, license, and describe any modifications or derived artifacts.

## TarteelAI Quranic Universal Library (QUL)

- Source: https://github.com/TarteelAI/quranic-universal-library
- Current project status: historical/planned source for Quran metadata, ayah text, and word/alignment data.
- Previously recorded license status: MIT for code and CC BY-SA-style terms for data. Reverify exact current license files before commercial release.
- Commercial note: if QUL data is bundled or used to generate derivative assets, keep attribution and any share-alike obligations visible in release docs.

## Offline Tarteel / Tilawa Recognition Work

- Source: https://github.com/yazinsai/offline-tarteel and successor/reference work https://github.com/yazinsai/tilawa
- Current project status: architecture/model provenance reference for FastConformer-style ONNX Quran recognition and regression discipline.
- Previously recorded license status: MIT. Reverify exact license and model asset provenance before commercial release.
- Commercial note: if `fastconformer_phoneme_q8.onnx` or `quran_phonemes.json` came from this lineage, keep the exact repository, commit, model checkpoint, export script, and quantization method in the release artifact.

## TarteelAI Quran Models

- Source: https://huggingface.co/tarteel-ai
- Current project status: research/model family reference.
- Commercial note: do not assume all TarteelAI-hosted models share one license. Verify each model card, dataset card, and checkpoint before training or distribution.

## Quranic Verse Recognition References

- Source: https://github.com/Abdelrahman47-code/Quranic-Verse-Recognition
- Current project status: reference for Levenshtein/transcript-to-verse confidence scoring ideas.
- Previously recorded license status: MIT. Reverify before copying or adapting code.

## EveryAyah Prompt Audio

- Source pattern used in app: `https://everyayah.com/data/<reciter>/<surah><ayah>.mp3`.
- Current project status: used at runtime for Luqmah prompt playback in `lib/ui/live_recitation_screen.dart`.
- Commercial note: treat this as an external recitation-audio dependency requiring permission/terms verification before commercial use, offline bundling, or redistribution.
- Release action: either obtain written permission, replace with a clearly licensed recitation corpus, or ship without bundled audio and require user-provided audio sources.

## Quran.com / QDC Audio and API

- Source pattern observed: `https://download.quranicaudio.com/qdc/...` in mock assets and Quran API usage in `lib/api/quran_api.dart`.
- Current project status: playback/highlighting support and mock data.
- Commercial note: verify API terms, rate limits, attribution rules, and audio redistribution rights before using in a paid product or hardware device.

## Current ASR Model and Quran Phoneme Data

- Model file: `assets/web/fastconformer_phoneme_q8.onnx`.
- Current project status: primary recognition model.
- Commercial note: original checkpoint, dataset, export chain, and model license must be identified before commercial release.
- Quran phoneme data: `assets/web/quran_phonemes.json` and `assets/models/quran_phonemes.json`.
- Commercial note: confirm source and license for Quran text normalization, transliteration/phoneme conversion, and any generated derivative data.

## Open-Source Code Dependencies

- Backend includes Python packages such as FastAPI, ONNX Runtime, NumPy, librosa, RapidFuzz/Levenshtein-related libraries, and Uvicorn.
- Flutter app includes Dart/Flutter packages and Android build dependencies.
- Web engine includes ONNX Runtime Web and JavaScript tooling.
- Release action: generate a full dependency license report for Python, Flutter/Gradle, and npm before public distribution.

## Commercial Release Rules

- Do not bundle third-party recitation audio until the specific reciter/source license permits commercial redistribution.
- Keep attribution visible in app settings/about screen and printed/device documentation when required.
- Keep a machine-readable copy of third-party notices in every shipped build.
- Track model provenance for every ONNX or quantized model, including original repository, checkpoint, license, training data, export script, and quantization method.
- Treat research-only/non-commercial models or datasets as evaluation-only unless written permission is obtained.
