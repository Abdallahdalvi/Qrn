# Third-Party Notices and Open Source Acknowledgements

This application utilizes components, datasets, and models from several open-source projects. We gratefully acknowledge the contributions of the open-source community.

## 1. TarteelAI Quranic Universal Library (QUL)
* **Author:** TarteelAI
* **Repository:** [https://github.com/TarteelAI/quranic-universal-library](https://github.com/TarteelAI/quranic-universal-library)
* **License:** MIT / CC BY-SA (Data)
* **Components Used:**
  - Quranic metadata (Surah names, Ayah counts)
  - Arabic Uthmani text datasets
  - Phoneme-to-Arabic alignment rules (Planned integration for next ayah helpers)

## 2. Offline Tarteel (Verse Recognition Engine)
* **Author:** yazinsai
* **Repository:** [https://github.com/yazinsai/offline-tarteel](https://github.com/yazinsai/offline-tarteel)
* **License:** MIT License
* **Components Used:**
  - `fastconformer_phoneme_q8.onnx` (Quantized Acoustic Model)
  - `quran_phonemes.json` (Target Phoneme Dictionary)
  - Mel-spectrogram feature extraction logic (adapted for backend pipeline)
  - Greedy phoneme decoding pipeline

## 3. Tarteel Whisper Quran Models
* **Author:** TarteelAI
* **Repository/Registry:** [https://huggingface.co/tarteel-ai](https://huggingface.co/tarteel-ai)
* **License:** Apache 2.0 / MIT
* **Components Used:**
  - Underlying acoustic research and dataset foundations that inspired the FastConformer quantization.

## 4. Quranic Verse Recognition Tool
* **Author:** Abdelrahman47-code
* **Repository:** [https://github.com/Abdelrahman47-code/Quranic-Verse-Recognition](https://github.com/Abdelrahman47-code/Quranic-Verse-Recognition)
* **License:** MIT License
* **Components Used:**
  - Reference for Levenshtein-based transcript-to-verse confidence scoring metrics.

---

## Integration Recommendations

**What can be directly reused:**
- **Datasets:** The QUL JSON/SQL exports should be used as the absolute source of truth for Surah names, Ayah text, and translation mappings.
- **Acoustic Models:** The ONNX quantized models (`fastconformer_phoneme_q8.onnx`) from `offline-tarteel` are perfectly suited for edge-device or local server inference and should be retained.
- **Word Alignment:** QUL's word-by-word timestamp data can be integrated with our phoneme tracker to offer precise word-highlighting.

**What should be rewritten/adapted:**
- **State Machine & Tracking:** Most open-source tools perform stateless global searches. Our "Taraweeh Mode", "Controlled Rewind", and "Pause Detection" state machines must be custom-written to fit the strict live-recitation requirements of Huffaz.
- **WebSocket Streaming:** The bridging between Flutter Audio Capture and Python ONNX Runtime via WebSockets is highly specific to our architecture and cannot be copy-pasted from existing repos.

**License Implications:**
All identified dependencies are available under highly permissive licenses (MIT, Apache 2.0, or CC). This permits commercial and private use, provided we maintain this `THIRD_PARTY_NOTICES.md` file and display an Acknowledgements page within the Flutter app settings.
