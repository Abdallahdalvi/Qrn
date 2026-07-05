from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import onnxruntime as ort
from Levenshtein import ratio


SAMPLE_RATE = 16000
ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
ARABIC_SYMBOL_RE = re.compile(r"[^\u0621-\u064a ]+")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    cases = data.get("cases", data if isinstance(data, list) else [])
    if not isinstance(cases, list):
        raise ValueError("Manifest must be a list or an object with a 'cases' list.")
    return cases


def _audio_path(case: dict[str, Any], manifest_path: Path) -> Path:
    raw = Path(case["audio"])
    if raw.is_absolute():
        return raw
    candidate = (manifest_path.parent / raw).resolve()
    if candidate.exists():
        return candidate
    return (_repo_root() / raw).resolve()


def _load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _features(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    audio = audio - float(np.mean(audio))
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms > 1e-6:
        audio = np.clip(audio * min(4.0, max(0.5, 0.08 / rms)), -0.98, 0.98)

    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=80,
        fmax=8000,
        htk=True,
        norm="slaney",
    )
    mel = np.log(mel + 1e-5)
    mel = (mel - mel.mean(axis=1, keepdims=True)) / (mel.std(axis=1, keepdims=True) + 1e-10)
    return mel.astype(np.float32)[np.newaxis], np.array([mel.shape[1]], dtype=np.int64)


def _load_tokens(path: Path) -> dict[int, str]:
    id_to_token: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        token, raw_id = line.rsplit(" ", 1)
        id_to_token[int(raw_id)] = token
    return id_to_token


def _decode_ctc(logprobs: np.ndarray, id_to_token: dict[int, str]) -> str:
    if logprobs.ndim == 3:
        logprobs = logprobs[0]
    blank_id = logprobs.shape[-1] - 1
    ids = np.argmax(logprobs, axis=-1).astype(int).tolist()
    pieces: list[str] = []
    prev = None
    for token_id in ids:
        if token_id == blank_id or token_id == prev:
            prev = token_id
            continue
        piece = id_to_token.get(token_id, "")
        if piece and piece != "<unk>":
            pieces.append(piece)
        prev = token_id
    return "".join(pieces).replace("▁", " ").strip()


def _normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\ufeff", "")
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.replace("\u0640", "")
    replacements = {
        "ٱ": "ا",
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = ARABIC_SYMBOL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _semi_global_distance(query: str, ref: str) -> int:
    if not query:
        return 0
    if not ref:
        return len(query)
    m = len(query)
    prev = list(range(m + 1))
    best = prev[m]
    for ch in ref:
        curr = [0] * (m + 1)
        for i in range(1, m + 1):
            cost = 0 if query[i - 1] == ch else 1
            curr[i] = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
        best = min(best, curr[m])
        prev = curr
    return best


def _fragment_score(query: str, ref: str) -> float:
    if not query:
        return 1.0
    return max(0.0, 1.0 - _semi_global_distance(query, ref) / len(query))


def _load_quran() -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    raw = json.loads((_repo_root() / "assets/web/quran_phonemes.json").read_text(encoding="utf-8-sig"))
    by_surah: dict[int, list[dict[str, Any]]] = {}
    for verse in raw:
        verse["surah"] = int(verse["surah"])
        verse["ayah"] = int(verse["ayah"])
        verse["_text_norm"] = _normalize_arabic(verse.get("text_uthmani", ""))
        by_surah.setdefault(verse["surah"], []).append(verse)
    for verses in by_surah.values():
        verses.sort(key=lambda v: v["ayah"])
    return raw, by_surah


def _window(by_surah: dict[int, list[dict[str, Any]]], surah: int, ayah: int, failed: int) -> list[dict[str, Any]]:
    verses = by_surah.get(surah, [])
    if not verses:
        return []
    last = verses[-1]["ayah"]
    if failed == 0:
        start, end = ayah, ayah + 4
    elif failed == 1:
        start, end = max(1, ayah - 8), ayah + 6
    elif failed == 2:
        start, end = max(1, ayah - 12), ayah + 8
    else:
        start, end = 1, last
    return [v for v in verses if start <= v["ayah"] <= min(end, last)]


def _match_text(text: str, verses: list[dict[str, Any]]) -> dict[str, Any] | None:
    norm = _normalize_arabic(text).replace(" ", "")
    if len(norm) < 4:
        return None
    best: dict[str, Any] | None = None
    for verse in verses:
        ref = verse.get("_text_norm", "").replace(" ", "")
        if not ref:
            continue
        score = max(ratio(norm, ref), _fragment_score(norm, ref))
        if best is None or score > best["score"]:
            best = {
                "surah": verse["surah"],
                "ayah": verse["ayah"],
                "score": float(score),
            }
    if best and best["score"] >= 0.58:
        return best
    return None


def _run_case(
    session: ort.InferenceSession,
    id_to_token: dict[int, str],
    by_surah: dict[int, list[dict[str, Any]]],
    case: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    audio = _load_audio(_audio_path(case, manifest_path))
    interval_sec = float(case.get("interval_sec", 1.5))
    window_sec = float(case.get("window_sec", 8.0))
    min_window_sec = float(case.get("min_window_sec", 1.5))
    stride = max(1, int(interval_sec * SAMPLE_RATE))
    window = max(stride, int(window_sec * SAMPLE_RATE))
    start = max(stride, int(min_window_sec * SAMPLE_RATE))
    current_surah = int(case.get("start_surah", case.get("surah", 0)) or 0)
    current_ayah = int(case.get("start_ayah", case.get("ayah", 0)) or 0)
    failed = 0
    hits: list[dict[str, Any]] = []
    latencies: list[float] = []
    input_names = [i.name for i in session.get_inputs()]

    for end in range(start, len(audio) + 1, stride):
        features, length = _features(audio[max(0, end - window):end].copy())
        t0 = time.perf_counter()
        logprobs = session.run(None, {input_names[0]: features, input_names[1]: length})[0]
        latencies.append(time.perf_counter() - t0)
        transcript = _decode_ctc(logprobs, id_to_token)
        candidates = _window(by_surah, current_surah, current_ayah, failed)
        match = _match_text(transcript, candidates)
        if match is None and failed >= 2:
            match = _match_text(transcript, by_surah.get(current_surah, []))
        if match:
            current_surah = int(match["surah"])
            current_ayah = int(match["ayah"])
            failed = 0
            hits.append({
                "t": round(end / SAMPLE_RATE, 3),
                "surah": current_surah,
                "ayah": current_ayah,
                "score": round(float(match["score"]), 4),
                "transcript": transcript[:80],
            })
        else:
            failed += 1

    failures: list[str] = []
    expected_final_surah = int(case.get("expected_final_surah", 0) or 0)
    expected_final_ayah_min = int(case.get("expected_final_ayah_min", 0) or 0)
    expected_final_ayah_max = int(case.get("expected_final_ayah_max", 0) or 0)
    if expected_final_surah and current_surah != expected_final_surah:
        failures.append(f"Final surah {current_surah}, expected {expected_final_surah}")
    if expected_final_ayah_min and current_ayah < expected_final_ayah_min:
        failures.append(f"Final ayah {current_ayah}, expected >= {expected_final_ayah_min}")
    if expected_final_ayah_max and current_ayah > expected_final_ayah_max:
        failures.append(f"Final ayah {current_ayah}, expected <= {expected_final_ayah_max}")

    return {
        "name": case.get("name", Path(case["audio"]).stem),
        "passed": not failures,
        "failures": failures,
        "final_surah": current_surah,
        "final_ayah": current_ayah,
        "verse_hits": hits,
        "summary": {
            "windows": max(0, (len(audio) - start) // stride + 1),
            "verse_hits": len(hits),
            "avg_latency": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "max_latency": round(max(latencies), 3) if latencies else 0.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an Arabic-text CTC ONNX model on Quran rolling windows.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    cases = _load_manifest(manifest_path)
    _, by_surah = _load_quran()
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    id_to_token = _load_tokens(Path(args.tokens))

    results = [_run_case(session, id_to_token, by_surah, case, manifest_path) for case in cases]
    passed = sum(1 for result in results if result["passed"])
    print(f"External text ASR regression: {passed}/{len(results)} passed")
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        summary = result["summary"]
        print(
            f"{status} {result['name']}: final={result['final_surah']}:{result['final_ayah']} "
            f"hits={summary['verse_hits']} avg_latency={summary['avg_latency']}s "
            f"max_latency={summary['max_latency']}s"
        )
        for failure in result["failures"]:
            print(f"  - {failure}")

    if args.out:
        Path(args.out).write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote detailed results to {args.out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
