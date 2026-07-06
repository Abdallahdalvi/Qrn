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

## Surah Rahman Wrong Recitation

Audio used locally:

`C:\Users\devev\Downloads\Test Quran\Surah Rahman Wrong Recitation.wav`

Start context:

- Taraweeh mode locked to Surah 55, Ayah 1.
- Repeating or briefly going back is allowed when the recitation is correct.
- In this file the repeated sections are intentionally wrong, so the system should use the repeated audio as extra evidence and issue Luqmah instead of waiting forever.

Expected correction cases:

- After Ayah 55:15, the reciter starts reading some other random ayah and repeats the wrong ayah twice. The system should prompt the expected continuation, Ayah 55:16, ideally within 3-5 seconds of the wrong-recitation evidence.
- After Ayah 55:21, the reciter takes a long pause. The system should prompt Ayah 55:22, not an older ayah.
- At Ayah 55:24, the reciter starts reading some other random ayah and repeats the wrong ayah twice. The system should prompt/correct Ayah 55:24.
- After Ayah 55:25, the reciter jumps to Ayah 55:29 and continues. The system should detect a forward jump and prompt from the missing expected continuation, starting at Ayah 55:26.

Baseline observed on 2026-07-06:

- The system currently emits a pause Luqmah at about 82.5s, but it incorrectly prompts expected 55:6 instead of the real expected continuation 55:16.
- The system currently emits a pause Luqmah at about 117s, but it incorrectly prompts expected 55:8 instead of 55:22.
- The Ayah 55:24 wrong-recitation case is caught at about 138s as `same_ayah_tail_mismatch`.
- The jump from 55:25 to 55:29 is not yet caught.
