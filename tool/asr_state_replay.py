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
    os.environ["QURAN_ASR_PROVIDER"] = provider
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
        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
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
        "expected_mistake_min": args.expected_mistake_min,
    }], None


def _expected_from_candidate(
    backend,
    result: dict[str, Any],
    current_surah: int,
    current_ayah: int,
    progress: float,
    last_ayah_change_t: float,
    now_t: float,
    taraweeh: bool,
) -> tuple[int, int]:
    reason = result.get("mistake_reason", "")
    prefer_current = (
        reason == "foreign_recitation"
        and (
            progress < 0.98
            or backend._is_rahman_repeated_refrain(current_surah, current_ayah)
            or (
                last_ayah_change_t > 0
                and now_t - last_ayah_change_t
                < backend._transition_grace_seconds(current_surah, current_ayah, progress)
            )
        )
    )
    return backend._expected_prompt_position(
        current_surah,
        current_ayah,
        progress,
        taraweeh,
        prefer_current=prefer_current,
    )


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

    current_progress = 0.0
    current_word_index = 0
    pending_forward = {"surah": 0, "ayah": 0, "count": 0}
    pending_rewind = {"ayah": 0, "count": 0}
    pending_loop = {"surah": 0, "ayah": 0, "count": 0}
    pending_tail_key: tuple[Any, ...] | None = None
    pending_tail_count = 0
    pending_mistake_key: tuple[Any, ...] | None = None
    pending_mistake_count = 0
    last_progress_t = 0.0
    last_ayah_change_t = 0.0
    has_committed_match = False
    mistake_cooldown_until = 0.0

    events: list[dict[str, Any]] = []
    verse_hits: list[dict[str, Any]] = []
    latencies: list[float] = []
    wall_start = time.perf_counter()

    def add_event(now_t: float, reason: str, expected: tuple[int, int], detected: tuple[int, int], score: float, extra: dict[str, Any] | None = None):
        event = {
            "t": round(now_t, 3),
            "reason": reason,
            "expected_surah": expected[0],
            "expected_ayah": expected[1],
            "detected_surah": detected[0],
            "detected_ayah": detected[1],
            "score": round(float(score or 0.0), 4),
            "state_surah": current_surah,
            "state_ayah": current_ayah,
            "progress_coverage": round(float(current_progress or 0.0), 4),
        }
        if extra:
            event.update(extra)
        events.append(event)

    for end in range(start, len(audio) + 1, stride):
        now_t = end / SAMPLE_RATE
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
        if metrics.get("total_latency"):
            latencies.append(float(metrics["total_latency"]))

        if result.get("surah", 0) > 0:
            matched_surah = int(result["surah"])
            matched_ayah = int(result["ayah_end"] if result.get("ayah_end") and result.get("progress_coverage", 0.0) >= 0.90 else result["ayah"])
            next_progress = float(result.get("progress_coverage", result.get("word_coverage", 0.0)) or 0.0)
            matched_word_index = int(result.get("word_index", 0) or 0)
            verse_hits.append({
                "t": round(now_t, 3),
                "surah": matched_surah,
                "ayah": matched_ayah,
                "score": result.get("score", 0.0),
                "tracking_mode": result.get("tracking_mode", ""),
                "progress_coverage": next_progress,
            })

            forward_distance = backend._forward_section_distance(
                current_surah,
                current_ayah,
                current_progress,
                matched_ayah,
                int(result.get("section_index", 0) or 0),
            )
            if taraweeh and current_surah > 0 and matched_surah == current_surah and matched_ayah < current_ayah:
                rewind_distance = current_ayah - matched_ayah
                recent_transition_overlap = (
                    rewind_distance == 1
                    and last_ayah_change_t > 0
                    and now_t - last_ayah_change_t < backend._REWIND_TRANSITION_GRACE_SECONDS
                )
                weak_near_rewind = rewind_distance == 1 and float(result.get("score", 0.0) or 0.0) < 0.88
                if recent_transition_overlap or weak_near_rewind:
                    pending_rewind = {"ayah": 0, "count": 0}
                    continue

                if (
                    current_surah == 55
                    and matched_surah == 55
                    and (
                        backend._is_rahman_repeated_refrain(matched_surah, matched_ayah)
                        or backend._is_rahman_repeated_refrain(current_surah, current_ayah)
                    )
                ):
                    pending_rewind = {"ayah": 0, "count": 0}
                    pending_loop = {"surah": 0, "ayah": 0, "count": 0}
                    failed_matches += 1
                    continue

                recent_tracker_relock = (
                    rewind_distance >= 2
                    and last_ayah_change_t > 0
                    and now_t - last_ayah_change_t < backend._REWIND_TRANSITION_GRACE_SECONDS
                    and float(result.get("score", 0.0) or 0.0) >= 0.80
                )
                if recent_tracker_relock:
                    pending_rewind = {"ayah": 0, "count": 0}
                    pending_loop = {"surah": 0, "ayah": 0, "count": 0}
                elif rewind_distance >= 2 and assisted_ayah == 0:
                    if pending_loop["surah"] == matched_surah and pending_loop["ayah"] == matched_ayah:
                        pending_loop["count"] += 1
                    else:
                        pending_loop = {"surah": matched_surah, "ayah": matched_ayah, "count": 1}
                    if pending_loop["count"] < 2:
                        continue
                    pending_loop = {"surah": 0, "ayah": 0, "count": 0}

                if not recent_tracker_relock:
                    if pending_rewind["ayah"] == matched_ayah:
                        pending_rewind["count"] += 1
                    else:
                        pending_rewind = {"ayah": matched_ayah, "count": 1}
                    required_rewind = 3 if rewind_distance == 1 else 2
                    if pending_rewind["count"] < required_rewind:
                        continue
                    pending_rewind = {"ayah": 0, "count": 0}
            else:
                pending_rewind = {"ayah": 0, "count": 0}
                pending_loop = {"surah": 0, "ayah": 0, "count": 0}

            skipped = max(0, matched_ayah - current_ayah - 1)
            required_forward = backend._required_forward_jump_confirmations(
                skipped,
                forward_distance,
                current_progress,
                float(result.get("score", 0.0) or 0.0),
            )
            ambiguous_rahman = (
                taraweeh
                and assisted_ayah == 0
                and current_surah == 55
                and matched_surah == 55
                and matched_ayah > current_ayah
                and backend._is_rahman_repeated_refrain(matched_surah, matched_ayah)
                and (
                    matched_ayah > current_ayah + 1
                    or backend._is_rahman_repeated_refrain(current_surah, current_ayah)
                )
            )
            if ambiguous_rahman:
                failed_matches += 1
                continue

            if (
                taraweeh
                and current_surah > 0
                and matched_surah == current_surah
                and matched_ayah > current_ayah
                and required_forward > 0
                and assisted_ayah == 0
            ):
                if pending_forward["surah"] == matched_surah and pending_forward["ayah"] == matched_ayah:
                    pending_forward["count"] += 1
                else:
                    pending_forward = {"surah": matched_surah, "ayah": matched_ayah, "count": 1}
                if pending_forward["count"] >= required_forward and skipped > 0 and now_t >= mistake_cooldown_until:
                    expected = backend._expected_prompt_position(
                        current_surah,
                        current_ayah,
                        current_progress,
                        taraweeh,
                    )
                    elapsed_since_ayah_change = (
                        now_t - last_ayah_change_t
                        if last_ayah_change_t > 0
                        else 999.0
                    )
                    min_forward_elapsed = backend._minimum_forward_jump_elapsed_seconds(
                        current_surah,
                        current_ayah,
                        matched_ayah,
                    )
                    if skipped > 8:
                        pending_forward = {"surah": 0, "ayah": 0, "count": 0}
                    elif elapsed_since_ayah_change < min_forward_elapsed:
                        add_event(
                            now_t,
                            "forward_jump",
                            expected,
                            (matched_surah, matched_ayah),
                            float(result.get("score", 0.0) or 0.0),
                            {
                                "confirmations": pending_forward["count"],
                                "required_confirmations": required_forward,
                                "elapsed_since_ayah_change": round(elapsed_since_ayah_change, 3),
                                "minimum_elapsed": round(min_forward_elapsed, 3),
                            },
                        )
                        mistake_cooldown_until = now_t + 3.0
                        pending_forward = {"surah": 0, "ayah": 0, "count": 0}
                        failed_matches = 0
                        continue
                    pending_forward = {"surah": 0, "ayah": 0, "count": 0}
                else:
                    continue
            pending_forward = {"surah": 0, "ayah": 0, "count": 0}

            previous_surah = current_surah
            previous_ayah = current_ayah
            previous_progress = current_progress
            previous_word_index = current_word_index
            current_surah = matched_surah
            current_ayah = matched_ayah
            has_committed_match = True
            same_position = matched_surah == previous_surah and matched_ayah == previous_ayah
            if not same_position:
                last_ayah_change_t = now_t
                current_word_index = matched_word_index
                current_progress = next_progress
            else:
                current_word_index = max(current_word_index, matched_word_index)
                current_progress = max(current_progress, next_progress)
            if (
                not same_position
                or current_word_index > previous_word_index
                or current_progress > previous_progress + backend._PROGRESS_STALE_DELTA
            ):
                last_progress_t = now_t
            failed_matches = 0

            tail = backend._same_ayah_tail_mismatch(result)
            if (
                taraweeh
                and assisted_ayah == 0
                and now_t >= mistake_cooldown_until
                and tail
            ):
                tail_key = (current_surah, current_ayah, tail["word_index"], tail["expected"])
                if pending_tail_key == tail_key:
                    pending_tail_count += 1
                else:
                    pending_tail_key = tail_key
                    pending_tail_count = 1
                if pending_tail_count >= 2:
                    add_event(
                        now_t,
                        "same_ayah_tail_mismatch",
                        (current_surah, current_ayah),
                        (current_surah, current_ayah),
                        max(0.01, 1.0 - float(tail["skeleton_ratio"])),
                        {
                            "word_index": tail["word_index"],
                            "expected_word": tail["expected"],
                            "heard_word": tail["got"],
                        },
                    )
                    mistake_cooldown_until = now_t + 3.0
                    pending_tail_key = None
                    pending_tail_count = 0
            else:
                pending_tail_key = None
                pending_tail_count = 0
            continue

        if result.get("mistake_candidate") and now_t >= mistake_cooldown_until:
            expected = _expected_from_candidate(
                backend,
                result,
                current_surah,
                current_ayah,
                current_progress,
                last_ayah_change_t,
                now_t,
                taraweeh,
            )
            mistake_reason = result.get("mistake_reason", "") or "mistake_candidate"
            mistake_surah = int(result.get("mistake_surah", 0) or 0)
            mistake_ayah = int(result.get("mistake_ayah", 0) or 0)
            if mistake_reason == "foreign_recitation" and mistake_surah > 0 and mistake_surah != current_surah:
                mistake_key = (expected[0], expected[1], "foreign_recitation")
            else:
                mistake_key = (expected[0], expected[1], mistake_surah, mistake_ayah)
            foreign_immediate_allowed = (
                mistake_reason != "foreign_recitation"
                or mistake_surah <= 0
                or mistake_surah == current_surah
                or current_progress >= 0.65
                or float(result.get("mistake_score", 0.0) or 0.0) >= 0.97
            )
            required = 1 if result.get("mistake_immediate") and foreign_immediate_allowed else 2
            if pending_mistake_key == mistake_key:
                pending_mistake_count += 1
            else:
                pending_mistake_key = mistake_key
                pending_mistake_count = 1
            if pending_mistake_count >= required:
                add_event(
                    now_t,
                    mistake_reason,
                    expected,
                    (mistake_surah, mistake_ayah),
                    float(result.get("mistake_score", 0.0) or 0.0),
                    {"confirmations": pending_mistake_count, "required_confirmations": required},
                )
                mistake_cooldown_until = now_t + 3.0
                pending_mistake_key = None
                pending_mistake_count = 0
            continue

        if result.get("speech_detected", True) and current_surah > 0:
            failed_matches += 1
            required_failures, stall_seconds = backend._stall_detection_thresholds(
                current_surah,
                current_ayah,
                current_progress,
                taraweeh,
            )
            if (
                taraweeh
                and failed_matches >= required_failures
                and now_t - last_progress_t >= stall_seconds
                and assisted_ayah == 0
                and has_committed_match
                and now_t >= mistake_cooldown_until
                and not backend._at_surah_end(current_surah, current_ayah)
            ):
                expected = backend._expected_prompt_position(
                    current_surah,
                    current_ayah,
                    current_progress,
                    taraweeh,
                )
                add_event(
                    now_t,
                    "no_context_progress",
                    expected,
                    (0, 0),
                    0.0,
                    {"failed_matches": failed_matches, "stall_seconds": stall_seconds},
                )
                failed_matches = 0

    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    max_latency = round(max(latencies), 3) if latencies else 0.0
    failures: list[str] = []
    if case.get("expect_no_mistake", False) and events:
        first = events[0]
        failures.append(f"Unexpected correction {first['reason']} at {first['t']}s")
    expected_mistake_min = int(case.get("expected_mistake_min", 0) or 0)
    if len(events) < expected_mistake_min:
        failures.append(f"Only {len(events)} correction(s), expected at least {expected_mistake_min}")
    expected_final_surah = int(case.get("expected_final_surah", 0) or 0)
    if expected_final_surah and current_surah != expected_final_surah:
        failures.append(f"Final surah {current_surah}, expected {expected_final_surah}")
    expected_final_ayah_min = int(case.get("expected_final_ayah_min", 0) or 0)
    expected_final_ayah_max = int(case.get("expected_final_ayah_max", 0) or 0)
    if expected_final_ayah_min and current_ayah < expected_final_ayah_min:
        failures.append(f"Final ayah {current_ayah}, expected >= {expected_final_ayah_min}")
    if expected_final_ayah_max and current_ayah > expected_final_ayah_max:
        failures.append(f"Final ayah {current_ayah}, expected <= {expected_final_ayah_max}")

    return {
        "name": case.get("name", audio_path.stem),
        "audio": str(audio_path),
        "duration_sec": round(len(audio) / SAMPLE_RATE, 3),
        "passed": not failures,
        "failures": failures,
        "final_surah": current_surah,
        "final_ayah": current_ayah,
        "correction_events": events,
        "verse_hits": verse_hits,
        "summary": {
            "windows": max(0, (len(audio) - start) // stride + 1),
            "verse_hits": len(verse_hits),
            "correction_events": len(events),
            "avg_latency": avg_latency,
            "max_latency": max_latency,
            "wall_time": round(time.perf_counter() - wall_start, 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay Quran ASR audio through a lightweight Taraweeh state machine.",
    )
    parser.add_argument("--manifest")
    parser.add_argument("--audio")
    parser.add_argument("--surah", type=int, default=0)
    parser.add_argument("--ayah", type=int, default=0)
    parser.add_argument("--taraweeh", action="store_true")
    parser.add_argument("--expect-no-mistake", action="store_true")
    parser.add_argument("--expected-mistake-min", type=int, default=0)
    parser.add_argument("--provider", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--decoder", default="context_beam", choices=["greedy", "beam", "context_beam"])
    parser.add_argument("--out")
    args = parser.parse_args()

    backend = _load_backend(args.provider, args.decoder)
    cases, manifest_path = _iter_cases(args)
    results = [_run_case(backend, case, manifest_path) for case in cases]

    passed = sum(1 for result in results if result["passed"])
    print(f"ASR state replay: {passed}/{len(results)} passed")
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        summary = result["summary"]
        print(
            f"{status} {result['name']}: final={result['final_surah']}:{result['final_ayah']} "
            f"corrections={summary['correction_events']} hits={summary['verse_hits']} "
            f"avg_latency={summary['avg_latency']}s max_latency={summary['max_latency']}s"
        )
        for failure in result["failures"]:
            print(f"  - {failure}")
        for event in result["correction_events"][:5]:
            print(
                "  * "
                f"{event['t']}s {event['reason']} expected "
                f"{event['expected_surah']}:{event['expected_ayah']} detected "
                f"{event['detected_surah']}:{event['detected_ayah']}"
            )

    if args.out:
        Path(args.out).write_text(
            json.dumps({"results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote detailed results to {args.out}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
