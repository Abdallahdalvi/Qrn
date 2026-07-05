# Live Test Notes - July 4, 2026

## Session

- Device: connected Android phone over `adb`
- Backend: local `127.0.0.1:8000`
- Test: `Surah Yaseen` recited live
- Goal: observe false Luqmah, cross-surah false hits, and Taraweeh forward-skip behavior

## Run: Yaseen 1 -> 30

### Startup issue still present

- `20:32:00` IST: matched `36:1` at `79%`
- `20:32:02` IST: matched `36:1` again at `87%`
- `20:32:08` IST: false `mistake_detected`
  - expected: `36:1`
  - detected: `0:0`
  - result: Luqmah for `36:1` played
- This is a startup/no-context stall issue, not a cross-surah false positive

### Tracking after startup recovered well

- `20:32:16` IST: `36:2`
- `20:32:21` IST: `36:3`
- `20:32:26` IST: `36:4`
- `20:32:32` IST: `36:5`
- `20:32:39` IST: `36:6`
- `20:32:50` IST: `36:7`
- `20:32:59` IST: `36:8`
- `20:33:10` IST: `36:9`
- `20:33:30` IST: `36:10`
- `20:33:40` IST: `36:11`
- `20:34:02` IST: `36:12`
- `20:34:25` IST: `36:13`
- `20:34:34` IST: `36:14`
- `20:34:44` IST: `36:15`
- `20:34:46` IST: `36:16`

### Ayah 11 false-positive regression check

- Earlier bug: `36:11` could trigger false Luqmah
- In this run:
  - no `84:20` cross-surah false hit
  - no Luqmah fired at `36:11`
  - tracker held on `36:11` across multiple partial fragments and advanced to `36:12`
- Conclusion: the `36:11 -> 84:20` bug appears fixed in this run

### Forward-jump protection behavior

- `20:32:55` IST: warning only
  - "Possible forward jump to Ayah 10 ... confirmation 1/4"
  - tracker did not commit the jump and later recovered into `36:8`
- `20:33:26` IST: warning only
  - "Possible repeated earlier ayah 4; holding current position."
- `20:34:55` IST: warning only
  - "Possible forward jump to Ayah 10 ... confirmation 1/4"
- `20:34:55` IST to `20:34:57` IST:
  - temporary rewind confirmation to `36:15`
  - then cleanly matched `36:15`
- Conclusion: forward-skip protection is working better; suspicious jumps were warned about but not committed immediately

### Soft warnings seen but not escalated into Luqmah

- `20:33:30` IST: word correction
  - ayah `36:10`
  - expected word: `"<am"`
  - heard: `"<"`
- These were informational and did not trigger audio correction

## Current assessment

- Fixed or strongly improved:
  - false Luqmah at long `36:11`
  - cross-surah partial-phrase false hit (`84:20`)
  - over-aggressive forward skips in Taraweeh
- Still broken:
  - startup from `36:1` can still trigger an early false Luqmah before the recitation has fully settled

## Run: Mulk 1 -> 30 - July 5, 2026

### Setup

- Device: connected Android phone over `adb`
- App: release APK installed before the run
- Backend: local `127.0.0.1:8000`
- Mode: Taraweeh, configured at `67:1`
- Source: YouTube recitation, so any Luqmah would be a tracker false positive
- Log files:
  - `backend-live-mulk-20260705-005005.out.log`
  - `backend-live-mulk-20260705-005005.err.log`
  - `adb-live-mulk-20260705-005101.log`

### Result

- Full Surah Mulk playback completed.
- `0` `mistake_detected` events.
- `0` Luqmah prompts.
- This is a clean user-facing result for Surah Mulk.

### Non-blocking anomalies observed

- App emitted 8 `word_correction` events, but the samples were mostly ASR artifacts:
  - expected word embedded inside a glued phrase, such as `xaasi<an` inside `<a<ilaykalbASAruxaasi<an`
  - tiny partial fragments, such as expected `rabbahum` but heard `raa`
  - very short word fragments, such as expected `man` but heard `a`
- Backend briefly confirmed a rewind from `67:17` back to `67:16`.
  - This appears to be ayah-boundary overlap, not an actual recitation error.
  - It did not produce Luqmah, but it showed internal tracker drift.
- After `67:30`, the app matched `68:1` and `68:2`.
  - This violates Taraweeh mode's intent: once locked to Surah Mulk, the tracker should not spill into the next surah.

### Fixes applied after this run

- Taraweeh search context now stays strictly inside the selected surah at the final ayah.
- End-of-surah global fallback hits from another surah are ignored in Taraweeh mode.
- Word correction scoring now treats embedded expected words inside glued ASR phrases as recognized.
- Word correction prompts are suppressed for tiny fragments and very short expected words.
- One-ayah rewind confirmations now have a transition grace period and require stronger confirmation.
- Frontend Taraweeh prompt timing now has safer floors:
  - 7 seconds minimum in Taraweeh mode
  - 12 seconds for startup/no-text/muqattaat openings such as `Yaseen` and `Ha Meem`
  - 10 seconds before early current-ayah correction prompts
- Frontend now defers `0:0`/no-context backend mistakes instead of playing instant Luqmah.
- Frontend now treats weak word-correction events as diagnostics unless confidence is severely low.

### Verification

- `python -m py_compile backend.py` passed.
- Focused backend regression checks passed for:
  - Mulk glued-word examples from the live run
  - short-fragment suppression
  - Taraweeh end-of-surah search staying within Surah 67
- `flutter test` passed.
- Release APK rebuilt and installed on device `DQHYWWMBPF4TNF4H`.
- Installed package:
  - `com.tarteel.alfatih_mobile`
  - `versionName=1.1.3`
  - `versionCode=5`
  - `lastUpdateTime=2026-07-05 01:23:11`

## Run: Rahman live test - July 5, 2026

### Setup

- Device: `DQHYWWMBPF4TNF4H`
- App: release APK `versionName=1.1.3`, installed at `2026-07-05 01:23:11`
- Backend: patched `backend.py` live on `127.0.0.1:8000`
- Source: YouTube recitation, so Luqmah should almost never fire
- ADB capture:
  - `adb-live-rahman-20260705-012831.log`
- Backend log marker:
  - `backend-live-after-mulk-20260705-010526.out.log`
  - marker bytes at start: `0`

### Live observations

- Taraweeh mode was configured for `55:1`.
- The app tracked through Surah Rahman to `55:78`.
- Sampled app logs showed:
  - `0` word corrections
  - `0` cross-surah verse matches during the surah
  - `0` Luqmah events in the sampled windows until the late-surah/end phase
- User reported a prompt around ayah `55:51`.
- Backend confirmed the false prompt:
  - `Assisted prompt pinned to 55:51`
  - `Assisted match successful. Locking to Ayah 51`
- Interpretation:
  - `55:51` is one of the repeated `fabi-ayyi alaa-i rabbikuma tukadhdhiban` refrains.
  - The app prompted while the correct repeated refrain was being recited or during a normal YouTube pause.
  - The backend immediately recovered because the recitation was correct.
- End-of-surah issue observed after `55:78`:
  - Backend repeatedly emitted uncertain `mistake_detected` for expected `56:1`.
  - Frontend correctly deferred these as `no_context_progress`, so no Luqmah audio played.
  - Still a backend bug: Taraweeh mode should not normalize `55:79` into `56:1`.

### Fixes applied after Rahman run

- Frontend now gives Rahman's repeated refrain a longer Taraweeh pause floor:
  - `15` seconds before auto-prompting `fabi-ayyi alaa-i rabbikuma tukadhdhiban`.
  - Continuous recitation keeps resetting the prompt timer, so a reciter repeating the phrase should not be interrupted.
  - If the reciter truly stops/stalls on that refrain, Luqmah can still happen after the longer grace period.
- Backend now clamps Taraweeh expected prompt positions at the selected surah's final ayah.
- Backend now suppresses assisted prompts that would cross from a Taraweeh surah into the next surah.
- Backend now sends `surah_complete` and leaves `next_ayah_text` empty at the selected surah end in Taraweeh mode.
- Frontend now uses `surah_complete` to stop end-of-surah countdown/prompt loops.

### Verification after Rahman fixes

- `python -m py_compile backend.py` passed.
- Focused backend regression passed:
  - `55:78` in Taraweeh stays `55:78`, not `56:1`.
  - Surah Rahman end search context stays inside Surah `55`.
- `flutter test` passed.
- Backend restarted with patched code:
  - `backend-live-after-rahman-fix-20260705-014818.err.log`
  - port `8000`, process `1464`
- Release APK rebuilt and installed on device `DQHYWWMBPF4TNF4H`.
- Installed package:
  - `com.tarteel.alfatih_mobile`
  - `versionName=1.1.3`
  - `versionCode=5`
  - `lastUpdateTime=2026-07-05 01:50:25`

## Run: Najm live test - July 5, 2026

### Setup

- Device: `DQHYWWMBPF4TNF4H`
- App: release APK `versionName=1.1.3`, installed at `2026-07-05 01:50:25`
- Backend: patched Rahman-fix `backend.py` live on `127.0.0.1:8000`
- Source: YouTube recitation, so Luqmah should almost never fire
- ADB capture:
  - `adb-live-najm-20260705-015638.log`
- Backend log marker:
  - `backend-live-after-rahman-fix-20260705-014818.out.log`
  - marker bytes at start: `0`

## Checkpoint B local audio regression - July 6, 2026

### Test corpus

- `C:\Users\devev\Downloads\Test Quran\Surah Rahman Correct Recitation.mp3`
- `C:\Users\devev\Downloads\Test Quran\Surah Qiyamah Correct Recitation.mp3`
- `C:\Users\devev\Downloads\Test Quran\Surah Rahman Wrong Recitation.wav`
- `C:\Users\devev\Downloads\Test Quran\Surah Qiyamah Wrong Recitation.wav`

### Current production path

- Provider: `CPUExecutionProvider`
- Decoder: `context_beam`
- Stateful Taraweeh replay: `4/4 passed`
- Correct recitations:
  - Rahman correct: final `55:78`, `0` corrections, avg latency `0.231s`
  - Qiyamah correct: final `75:40`, `0` corrections, avg latency `0.226s`
- Wrong recitations:
  - Rahman wrong: `3` corrections, including `55:24` tail mismatch (`fiaaalbaHrikaal<aElaami` expected vs `fiaaaljinniwal<insi` heard)
  - Qiyamah wrong: `1` foreign-recitation correction near expected `75:17`

### CUDA finding

- CUDA was tested on the same local files but is not recommended for checkpoint B.
- On this Q8 ONNX model it was slower and less stable than CPU.
- ONNX Runtime also reports hundreds of graph memcpy nodes for CUDA, which likely explains the poor GTX-class performance.

### External model spike

- `TheGreatQuran2026/fastconformer-quran-ar` ONNX downloaded to `tmp-models` for local-only evaluation.
- Model metadata from Hugging Face:
  - FastConformer Hybrid Large, `114.6M` parameters
  - Arabic BPE/CTC-style output, `1025` output classes
  - `88.4 MB` quantized ONNX
  - reported WER `0.14%` on `tarteel-ai/everyayah`, unverified
- Local Arabic-text verse tracking adapter: `tool/asr_text_model_regression.py`
- Same corpus verse-level result: `4/4 passed`
  - Rahman correct: final `55:78`, avg latency `0.419s`
  - Qiyamah correct: final `75:40`, avg latency `0.411s`
  - Rahman wrong: final `55:21`, avg latency `0.407s`
  - Qiyamah wrong: final `75:18`, avg latency `0.410s`
- Interpretation:
  - promising for future Arabic-text verse tracking
  - slower than current CPU phoneme model in this harness
  - does not yet replace phoneme-level Luqmah/tail mismatch detection

### Policy correction

- Repetition/backward recitation is valid in Taraweeh.
- Confirmed rewind/repetition is now treated as reciter tracking, not `mistake_detected`.
- Luqmah should come from wrong-word/tail mismatch, foreign-surah recitation, impossible fast forward jumps, or no-progress/stuck behavior, not from a Hafiz repeating earlier ayahs.

### Live observations

- Wrong other-surah recitation at `55:16` worked:
  - backend globally matched the wrong recitation to `53:56` with high confidence
  - app played Luqmah pinned to expected recovery target `55:17`
  - backend then locked successfully back onto `55:17`
- Long pause after `55:21` did not play the expected Luqmah for `55:22`.
  - backend emitted repeated `mistake_detected` events with `reason=no_context_progress`
  - frontend deferred those uncertain corrections and refreshed `_lastProgressionTime`
  - result: the local pause countdown kept getting reset before it could mature
- Wrong ending at `55:24` was detected but not prompted.
  - user intentionally recited `aajameen` instead of the tail of `55:24`
  - backend briefly saw a strong wrong-recitation candidate, globally matched to `96:1`
  - the earlier no-context path/cooldown and confirmation logic meant no Luqmah played
- Planned next regression:
  - repeat the same Rahman manual faults
  - add a forward jump from `55:25` to `55:31`

### Fixes applied after manual Rahman fault run

- Backend no longer sends Taraweeh no-context stalls as `mistake_detected`.
  - it now sends a non-audio `status` event with `tracking_mode=TARAWEEH_STALL`
  - it does not start `mistake_cooldown_until`
  - this lets the frontend pause timer continue and trigger the local Luqmah normally
- Backend now allows a strong, non-ambiguous immediate wrong-recitation candidate to prompt on the first hit.
  - repeated ambiguous signatures still require extra confirmation
  - weaker mid-ayah/cross-surah candidates still require confirmation

## Run: Rahman manual fault retest - July 5, 2026

### Setup

- Device: `DQHYWWMBPF4TNF4H`
- App: release APK `versionName=1.1.3`, installed at `2026-07-05 02:08:07`
- Backend: `backend-live-after-manual-fix-20260705-152600.out.log`
- ADB logcat: `adb-live-rahman-manual2-20260705-152900.log`
- Planned faults:
  - wrong other-surah recitation around `55:16`
  - long pause after `55:21` to trigger Luqmah for `55:22`
  - wrong tail at `55:24`, reciting `aajameen` instead of `fil bahri kal-a'lam`
  - forward jump from `55:25` to `55:31`

### Live observations

- Setup noise:
  - ADB entry accidentally started Taraweeh at `114:1`; stopped and restarted.
  - Clean user-controlled run then started at `55:1`.
- Tracking progressed through Rahman into the 20s.
- Planned wrong-surah recitation around `55:16` did not produce a visible `mistake_detected` in the sampled logs.
- Long pause after `55:21` behaved correctly after the backend fix:
  - backend logged `Assisted prompt pinned to 55:22`
  - then `Assisted match successful. Locking to Ayah 22`
  - no repeated `no_context_progress` cooldown loop occurred
- Wrong tail at `55:24` is still not fixed:
  - backend decoded variants like `faa faaalbar alaajamiin`, `faufalbar kalaajamiin`, and `f abari alaaamiin`
  - backend only rejected weak jump candidates such as `55:43`, `55:41`, `55:19`, `55:8`, `55:44`, and `55:1`
  - no Luqmah or actionable word correction was emitted for the local tail substitution
- Forward jump from `55:25` to `55:31` worked:
  - frontend first logged `Possible forward jump to Ayah 31 ... confirmation 1/4`
  - backend then logged `Immediate mistake near 55:25; globally matched 55:31 (Conf: 0.95)`
  - backend sent `mistake_detected`
  - frontend logged `[MISTAKE] Detected recitation of 55:31 instead of 55:26`
  - frontend queued and started Luqmah for `55:26`

### Remaining fix target

- Add a local same-ayah tail-mismatch detector so a wrong ending inside the expected ayah, such as `55:24` `aajameen` versus `fil bahri kal-a'lam`, can trigger a current-ayah/word Luqmah without requiring a global cross-surah match.

### Tail-mismatch fix applied - July 5, 2026

- Added a backend same-ayah final-word detector for Taraweeh mode.
- The detector compares the final recognized phoneme word against the expected final word with both raw phoneme ratio and a vowel-stripped skeleton ratio.
- It requires two consecutive confirmations before sending `mistake_detected` with reason `same_ayah_tail_mismatch`.
- Local validation against the live `55:24` decodes:
  - ignored noisy near-match `kal<aEqli`
  - flagged wrong endings `alaajamiin`, `kalaajamiin`, and `alaaamiin`
  - ignored correct split variants such as `kaal <aElaami`
- Backend restarted on `127.0.0.1:8000`; health check passed.
- Backend log files:
  - `backend-live-tail-fix-20260705-201534.out.log`
  - `backend-live-tail-fix-20260705-201534.err.log`

### Live observations

- Taraweeh mode was configured for `53:1`.
- The app tracked through Surah Najm to `53:62`.
- User-facing result:
  - `0` Luqmah prompts
  - `0` `mistake_detected`
  - `0` word corrections
- Notable non-blocking behavior:
  - A long hold around `53:26` and `53:32`, but no prompt fired.
  - Several forward-jump warnings were held instead of committed immediately, including candidates around `53:20`, `53:28`, `53:40`, `53:45`, `53:47`, `53:48`, and `53:50`.
  - End of surah behaved correctly:
    - matched `53:62`
    - then repeatedly logged `Surah 53 complete; holding Taraweeh lock`
    - no `54:1` spill
    - no end-of-surah Luqmah

### Post-run tuning

- User feedback: `15` seconds for Rahman's repeated refrain is too long.
- Frontend Rahman repeated-refrain floor changed from `15` seconds to `11` seconds.
- This still protects against normal YouTube/repeated-refrain pauses, but should help faster if a real reciter gets stuck.

### Verification after Najm run

- `flutter test` passed.
- Release APK rebuilt and installed on device `DQHYWWMBPF4TNF4H`.
- Installed package:
  - `com.tarteel.alfatih_mobile`
  - `versionName=1.1.3`
  - `versionCode=5`
  - `lastUpdateTime=2026-07-05 02:08:07`

## Run: Rahman YouTube false Luqmah retest - July 5, 2026

### Observation

- Device: `DQHYWWMBPF4TNF4H`
- Backend: `backend-adb-test-20260705-233529.out.log`
- ADB logcat: `adb-live-app-20260705-233730.log`
- Setup: Taraweeh mode locked to Surah Rahman `55:1`, app auto-recording from Taraweeh start.
- Tracking initially progressed correctly through Rahman into the teens.
- False Luqmah fired around `55:17`.
- Backend root cause:
  - local Taraweeh matching rejected non-local candidates correctly
  - the later restart fallback still searched the full `_verses` Quran corpus
  - it globally matched the audio to `53:57` with score about `0.79`
  - WebSocket confirmation then sent `mistake_detected`, causing Luqmah while the run was supposed to be Surah-locked

### Fix applied

- Taraweeh restart recovery now searches only `_by_surah[current_surah]`.
- Added a defensive suppression metric/log if a future matcher change ever produces a cross-surah Taraweeh recovery candidate.
- This preserves same-surah recovery/jump detection while enforcing the rule that Taraweeh mode must not compare against other surahs.

## Run: Rahman manual fault test - July 5, 2026

### Setup

- Device: `DQHYWWMBPF4TNF4H`
- App: release APK `versionName=1.1.3`, installed at `2026-07-05 02:08:07`
- Backend: patched Rahman-fix `backend.py` live on `127.0.0.1:8000`
- Planned recitation: Surah Rahman `55:1` through `55:25`
- Planned faults:
  - at `55:16`, recite a different ayah from another surah
  - after `55:21`, pause long enough to trigger Luqmah for `55:22`
  - at `55:24`, make a mistake in the second half of the ayah
- ADB capture:
  - `adb-live-rahman-manual-20260705-022505.log`
- Backend log marker:
  - `backend-live-after-rahman-fix-20260705-014818.out.log`
  - marker bytes at start: `0`
