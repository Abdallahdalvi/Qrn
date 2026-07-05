from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np


SAMPLE_RATE = 16000


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_backend(provider: str, decoder: str):
    if provider:
        os.environ["QURAN_ASR_PROVIDER"] = provider
    if decoder:
        os.environ["QURAN_ASR_DECODER"] = decoder

    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)

    import backend  # noqa: PLC0415

    return backend


def _load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _case_audio_path(case: dict[str, Any], manifest_path: Path | None) -> Path:
    raw = Path(case["audio"])
    if raw.is_absolute():
        return raw
    if manifest_path is not None:
        candidate = (manifest_path.parent / raw).resolve()
        if candidate.exists():
            return candidate
    return (_repo_root() / raw).resolve()


def _iter_cases(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path | None]:
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = data.get("cases", data if isinstance(data, list) else [])
        if not isinstance(cases, list):
            raise ValueError("Manifest must be a list or an object with a 'cases' list.")
        return cases, manifest_path

    if not args.audio:
        raise ValueError("Provide --manifest or --audio.")

    return [{
        "name": Path(args.audio).stem,
        "audio": args.audio,
        "start_surah": args.surah,
        "start_ayah": args.ayah,
        "taraweeh": args.taraweeh,
        "expect_no_mistake": args.expect_no_mistake,
        "expected_final_surah": args.expected_final_surah,
        "expected_final_ayah_min": args.expected_final_ayah_min,
        "expected_final_ayah_max": args.expected_final_ayah_max,
    }], None


def _run_case(backend, case: dict[str, Any], manifest_path: Path | None) -> dict[str, Any]:
    audio_path = _case_audio_path(case, manifest_path)
    audio = _load_audio(audio_path)

    interval_sec = float(case.get("interval_sec", 1.5))
    window_sec = float(case.get("window_sec", 8.0))
    min_window_sec = float(case.get("min_window_sec", 1.5))
    stride = max(1, int(interval_sec * SAMPLE_RATE))
    window = max(stride, int(window_sec * SAMPLE_RATE))
    start = max(stride, int(min_window_sec * SAMPLE_RATE))

    current_surah = int(case.get("start_surah", case.get("surah", 0)) or 0)
    current_ayah = int(case.get("start_ayah", case.get("ayah", 0)) or 0)
    taraweeh = bool(case.get("taraweeh", current_surah > 0))
    failed_matches = 0
    assisted_surah = int(case.get("assisted_surah", 0) or 0)
    assisted_ayah = int(case.get("assisted_ayah", 0) or 0)
    post_recovery_lock = bool(case.get("post_recovery_lock", False))

    observations: list[dict[str, Any]] = []
    mistake_events: list[dict[str, Any]] = []
    verse_hits: list[dict[str, Any]] = []
    latencies: list[float] = []
    start_time = time.perf_counter()

    for end in range(start, len(audio) + 1, stride):
        chunk = audio[max(0, end - window):end].copy()
        result = backend.predict_audio(
            chunk,
            current_surah,
            current_ayah,
            failed_matches,
            taraweeh,
            assisted_surah,
            assisted_ayah,
            post_recovery_lock,
        )
        metrics = result.get("metrics", {}) or {}
        latency = float(metrics.get("total_latency", 0.0) or 0.0)
        if latency > 0:
            latencies.append(latency)

        observation = {
            "t": round(end / SAMPLE_RATE, 3),
            "surah": result.get("surah", 0),
            "ayah": result.get("ayah", 0),
            "score": result.get("score", 0.0),
            "word_index": result.get("word_index", 0),
            "progress_coverage": result.get("progress_coverage", 0.0),
            "error": result.get("error", ""),
            "tracking_mode": result.get("tracking_mode", ""),
            "decoder": metrics.get("selected_decoder", metrics.get("decoder", "")),
            "latency": latency,
        }
        observations.append(observation)

        if result.get("mistake_candidate") or str(result.get("error", "")).lower().startswith("mistake"):
            mistake = {
                **observation,
                "mistake_surah": result.get("mistake_surah", 0),
                "mistake_ayah": result.get("mistake_ayah", 0),
                "mistake_score": result.get("mistake_score", 0.0),
            }
            mistake_events.append(mistake)

        if result.get("surah", 0) > 0 and result.get("ayah", 0) > 0:
            matched_surah = int(result["surah"])
            matched_ayah = int(result["ayah"])
            verse_hits.append(observation)
            current_surah = matched_surah
            current_ayah = matched_ayah
            failed_matches = 0
        else:
            failed_matches += 1

    failures: list[str] = []
    if case.get("expect_no_mistake", False) and mistake_events:
        first = mistake_events[0]
        failures.append(
            "Unexpected mistake candidate at "
            f"{first['t']}s near {first.get('mistake_surah')}:{first.get('mistake_ayah')}"
        )

    expected_final_surah = int(case.get("expected_final_surah", 0) or 0)
    expected_final_min = int(case.get("expected_final_ayah_min", 0) or 0)
    expected_final_max = int(case.get("expected_final_ayah_max", 0) or 0)
    if expected_final_surah and current_surah != expected_final_surah:
        failures.append(f"Final surah {current_surah} != expected {expected_final_surah}")
    if expected_final_min and current_ayah < expected_final_min:
        failures.append(f"Final ayah {current_ayah} < expected minimum {expected_final_min}")
    if expected_final_max and current_ayah > expected_final_max:
        failures.append(f"Final ayah {current_ayah} > expected maximum {expected_final_max}")

    if taraweeh and case.get("fail_on_cross_surah", True):
        locked_surah = int(case.get("start_surah", current_surah) or current_surah)
        cross_surah = [hit for hit in verse_hits if hit["surah"] and hit["surah"] != locked_surah]
        if cross_surah:
            first = cross_surah[0]
            failures.append(f"Cross-surah hit {first['surah']}:{first['ayah']} at {first['t']}s")

    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    max_latency = round(max(latencies), 3) if latencies else 0.0
    return {
        "name": case.get("name", audio_path.stem),
        "audio": str(audio_path),
        "duration_sec": round(len(audio) / SAMPLE_RATE, 3),
        "passed": not failures,
        "failures": failures,
        "final_surah": current_surah,
        "final_ayah": current_ayah,
        "mistake_events": mistake_events,
        "verse_hits": verse_hits,
        "observations": observations,
        "summary": {
            "windows": len(observations),
            "verse_hits": len(verse_hits),
            "mistake_events": len(mistake_events),
            "avg_latency": avg_latency,
            "max_latency": max_latency,
            "wall_time": round(time.perf_counter() - start_time, 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay Quran ASR audio through backend.predict_audio in rolling windows.",
    )
    parser.add_argument("--manifest", help="JSON manifest containing regression cases.")
    parser.add_argument("--audio", help="Single audio file to replay.")
    parser.add_argument("--surah", type=int, default=0, help="Start/current surah for --audio mode.")
    parser.add_argument("--ayah", type=int, default=0, help="Start/current ayah for --audio mode.")
    parser.add_argument("--taraweeh", action="store_true", help="Enable Taraweeh locked-surah mode.")
    parser.add_argument("--expect-no-mistake", action="store_true", help="Fail if a mistake candidate appears.")
    parser.add_argument("--expected-final-surah", type=int, default=0)
    parser.add_argument("--expected-final-ayah-min", type=int, default=0)
    parser.add_argument("--expected-final-ayah-max", type=int, default=0)
    parser.add_argument("--provider", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--decoder", default="greedy", choices=["greedy", "beam", "context_beam"])
    parser.add_argument("--out", help="Write detailed JSON results to this path.")
    args = parser.parse_args()

    cases, manifest_path = _iter_cases(args)
    backend = _load_backend(args.provider, args.decoder)
    results = [_run_case(backend, case, manifest_path) for case in cases]

    passed = sum(1 for result in results if result["passed"])
    print(f"ASR regression: {passed}/{len(results)} passed")
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        summary = result["summary"]
        print(
            f"{status} {result['name']}: final={result['final_surah']}:{result['final_ayah']} "
            f"hits={summary['verse_hits']} mistakes={summary['mistake_events']} "
            f"avg_latency={summary['avg_latency']}s max_latency={summary['max_latency']}s"
        )
        for failure in result["failures"]:
            print(f"  - {failure}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
        print(f"Wrote detailed results to {out_path}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
