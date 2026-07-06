# ASR Regression Ground Truth

This file records the intended mistakes in manually recorded regression audio.
Use it when updating `tmp-local-audio-regression.json` or any shared manifest.

## Surah Qiyamah Wrong Recitation

Audio used locally:

`C:\Users\devev\Downloads\Test Quran\Surah Qiyamah Wrong Recitation.wav`

Start context:

- Taraweeh mode locked to Surah 75, Ayah 1.
- The reciter self-corrects most planted mistakes within roughly 3-5 seconds.

Expected correction cases:

- Around Ayah 75:4, the reciter intentionally reads something else. The system should issue Luqmah for 75:4 quickly, ideally within 3-5 seconds.
- After Ayah 75:10, the reciter jumps forward to Ayah 75:14. The system should detect a forward jump and prompt from the missing expected continuation, starting at 75:11.
- After Ayah 75:16, the reciter wrongly recites Ayah 75:18. The system should prompt the expected next ayah, 75:17. Current baseline may classify this as either `forward_jump` or `foreign_surah` depending on ASR matching ambiguity.

Expected non-correction case:

- Around Ayah 75:13, there is a small stutter. This should not be treated as a separate Luqmah unless it also coincides with a real jump or wrong-recitation correction.

Baseline observed on 2026-07-06:

- The after-75:16 wrong-recitation case is caught at about 93s as `foreign_recitation`, expected 75:17 and detected 78:17.
- The Ayah 75:4 wrong-recitation case is not yet caught.
- The jump from 75:10 to 75:14 is not yet caught.
