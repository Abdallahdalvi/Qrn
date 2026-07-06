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

_REASON_ALIASES = {
    "pause": {"pause", "stall", "no_context_progress"},
    "tail_mismatch": {"tail_mismatch", "same_ayah_tail_mismatch"},
    "foreign_surah": {"foreign_surah", "foreign_recitation"},
    "low_confidence": {"low_confidence", "word_correction"},
    "wrong_recitation": {
        "wrong_recitation",
        "same_surah_wrong_recitation",
        "same_ayah_tail_mismatch",
        "tail_mismatch",
        "foreign_recitation",
        "foreign_surah",
        "low_confidence",
        "word_correction",
    },
}


def _reason_matches(actual: str, expected: str) -> bool:
    actual_norm = (actual or "").strip()
    expected_norm = (expected or "").strip()
    if not expected_norm:
        return True
    if actual_norm == expected_norm:
        return True
    aliases = _REASON_ALIASES.get(expected_norm, {expected_norm})
    return actual_norm in aliases


def _event_matches_expected(event: dict[str, Any], expected: dict[str, Any]) -> bool:
    reason = expected.get("reason", "")
    reasons = expected.get("reasons")
    if reasons:
        if not any(_reason_matches(event.get("reason", ""), str(item)) for item in reasons):
            return False
    elif reason and not _reason_matches(event.get("reason", ""), str(reason)):
        return False

    t_min = expected.get("t_min")
    t_max = expected.get("t_max")
    if t_min is not None and float(event.get("t", 0.0)) < float(t_min):
        return False
    if t_max is not None and float(event.get("t", 0.0)) > float(t_max):
        return False

    for field in ("expected_surah", "expected_ayah", "detected_surah", "detected_ayah"):
        if expected.get(field) is not None and int(event.get(field, 0) or 0) != int(expected[field]):
            return False

    min_score = expected.get("min_score")
    if min_score is not None and float(event.get("score", 0.0) or 0.0) < float(min_score):
        return False

    return True


def _validate_expected_corrections(
    events: list[dict[str, Any]],
    case: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    failures: list[str] = []
    checks: list[dict[str, Any]] = []
    for index, expected in enumerate(case.get("expected_corrections", []) or [], start=1):
        label = expected.get("label") or expected.get("reason") or f"expected correction {index}"
        matches = [event for event in events if _event_matches_expected(event, expected)]
        matched = matches[0] if matches else None
        checks.append({
            "label": label,
            "matched": matched is not None,
            "expected": expected,
            "event": matched,
        })
        if matched is None:
            failures.append(f"Missing expected correction: {label}")
    return failures, checks


def _validate_expected_non_corrections(
    events: list[dict[str, Any]],
    case: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    failures: list[str] = []
    checks: list[dict[str, Any]] = []
    for index, expected in enumerate(case.get("expected_non_corrections", []) or [], start=1):
        label = expected.get("label") or f"expected no correction {index}"
        matches = [event for event in events if _event_matches_expected(event, expected)]
        matched = matches[0] if matches else None
        checks.append({
            "label": label,
            "matched": matched is None,
            "expected": expected,
            "event": matched,
        })
        if matched is not None:
            failures.append(f"Unexpected correction during no-prompt window: {label}")
    return failures, checks


def _find_unexpected_corrections(
    events: list[dict[str, Any]],
    expected_checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matched_event_ids = {
        id(check["event"])
        for check in expected_checks
        if check.get("event") is not None
    }
    matched_targets = {
        (
            int((check.get("expected") or {}).get("expected_surah", 0) or 0),
            int((check.get("expected") or {}).get("expected_ayah", 0) or 0),
        )
        for check in expected_checks
        if check.get("matched")
    }
    return [
        event
        for event in events
        if id(event) not in matched_event_ids
        and (
            int(event.get("expected_surah", 0) or 0),
            int(event.get("expected_ayah", 0) or 0),
        )
        not in matched_targets
    ]


def _format_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{(numerator / denominator) * 100:.1f}%"


def _report_group_name(result: dict[str, Any]) -> str:
    raw = str(result.get("name") or "")
    if raw:
        return raw.replace("_local", "").replace("_", " ").title()
    return Path(str(result.get("audio") or "")).stem


def _expected_status_line(check: dict[str, Any]) -> str:
    status = "PASS" if check.get("matched") else "FAIL"
    expected = check.get("expected", {}) or {}
    event = check.get("event") or {}
    details: list[str] = []
    if expected.get("reason"):
        details.append(str(expected["reason"]))
    elif expected.get("reasons"):
        details.append("/".join(str(reason) for reason in expected["reasons"]))
    if expected.get("expected_surah") and expected.get("expected_ayah"):
        details.append(f"expected {expected['expected_surah']}:{expected['expected_ayah']}")
    if event:
        details.append(
            f"caught at {event.get('t')}s as {event.get('reason')} "
            f"detected {event.get('detected_surah')}:{event.get('detected_ayah')}"
        )
    return f"{status} {check.get('label', 'expected correction')} ({'; '.join(details)})"


def _non_correction_status_line(check: dict[str, Any]) -> str:
    status = "PASS" if check.get("matched") else "FAIL"
    expected = check.get("expected", {}) or {}
    event = check.get("event") or {}
    details: list[str] = []
    if expected.get("t_min") is not None and expected.get("t_max") is not None:
        details.append(f"{expected['t_min']}-{expected['t_max']}s")
    if event:
        details.append(
            f"unexpected {event.get('reason')} at {event.get('t')}s "
            f"expected {event.get('expected_surah')}:{event.get('expected_ayah')}"
        )
    return f"{status} no-prompt: {check.get('label', 'expected no correction')} ({'; '.join(details)})"


def _event_status_line(event: dict[str, Any]) -> str:
    details = [
        f"{event.get('t')}s",
        str(event.get("reason", "")),
        f"expected {event.get('expected_surah')}:{event.get('expected_ayah')}",
        f"detected {event.get('detected_surah')}:{event.get('detected_ayah')}",
        f"confidence {float(event.get('score', 0.0) or 0.0):.3f}",
    ]
    if event.get("expected_word") or event.get("heard_word"):
        details.append(
            f"word {event.get('expected_word', '')}->{event.get('heard_word', '')}"
        )
    if event.get("elapsed_for_jump") is not None:
        details.append(
            f"elapsed {event.get('elapsed_for_jump')}s/min {event.get('minimum_elapsed')}s"
        )
    return "- " + "; ".join(details)


def _build_regression_report(results: list[dict[str, Any]], label: str) -> str:
    correct = [result for result in results if result.get("expect_no_mistake")]
    wrong = [result for result in results if not result.get("expect_no_mistake")]

    correct_passed = sum(1 for result in correct if not result.get("correction_events"))
    expected_checks = [
        check
        for result in wrong
        for check in result.get("expected_corrections", [])
    ]
    expected_passed = sum(1 for check in expected_checks if check.get("matched"))
    no_prompt_checks = [
        check
        for result in wrong
        for check in result.get("expected_non_corrections", [])
    ]
    no_prompt_passed = sum(1 for check in no_prompt_checks if check.get("matched"))

    lines: list[str] = [
        label,
        "",
        "Correct Files",
        "",
    ]
    if correct:
        for result in correct:
            false_luqmah = len(result.get("correction_events", []))
            status = "PASS" if false_luqmah == 0 else "FAIL"
            lines.append(f"{status} {_report_group_name(result)}")
            lines.append(f"{false_luqmah} false Luqmah")
            if false_luqmah:
                lines.append("Actual Luqmah Events:")
                for event in result.get("correction_events", []):
                    lines.append(_event_status_line(event))
            lines.append("")
    else:
        lines.append("No correct-recitation files configured.")
        lines.append("")

    lines.extend(["Wrong Files", ""])
    if wrong:
        for result in wrong:
            checks = result.get("expected_corrections", [])
            lines.append(_report_group_name(result))
            if checks:
                for check in checks:
                    lines.append(_expected_status_line(check))
                matched = sum(1 for check in checks if check.get("matched"))
                lines.append(f"Accuracy: {matched}/{len(checks)} ({_format_pct(matched, len(checks))})")
            else:
                corrections = len(result.get("correction_events", []))
                lines.append(f"No individual expected corrections configured; observed {corrections} correction(s).")
            no_prompt = result.get("expected_non_corrections", [])
            if no_prompt:
                for check in no_prompt:
                    lines.append(_non_correction_status_line(check))
                clean = sum(1 for check in no_prompt if check.get("matched"))
                lines.append(f"No-prompt guards: {clean}/{len(no_prompt)} ({_format_pct(clean, len(no_prompt))})")
            events = result.get("correction_events", [])
            if events:
                lines.append("Actual Luqmah Events:")
                for event in events:
                    lines.append(_event_status_line(event))
            else:
                lines.append("Actual Luqmah Events: none")
            unexpected = result.get("unexpected_corrections", [])
            if unexpected:
                lines.append("Unexpected Luqmah Events:")
                for event in unexpected:
                    lines.append(_event_status_line(event))
            else:
                lines.append("Unexpected Luqmah Events: none")
            lines.append("")
            lines.append("--------------------")
            lines.append("")
    else:
        lines.append("No wrong-recitation files configured.")
        lines.append("")

    lines.extend([
        "Overall",
        "",
        f"Correct: {correct_passed}/{len(correct)} ({_format_pct(correct_passed, len(correct))})",
        f"Wrong: {expected_passed}/{len(expected_checks)} ({_format_pct(expected_passed, len(expected_checks))})",
        f"No-Prompt: {no_prompt_passed}/{len(no_prompt_checks)} ({_format_pct(no_prompt_passed, len(no_prompt_checks))})",
        f"Overall Detection: {_format_pct(correct_passed + expected_passed, len(correct) + len(expected_checks))}",
    ])
    return "\n".join(lines).rstrip() + "\n"


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
    simulate_client_prompts = bool(case.get("simulate_client_prompts", True))
    simulate_client_pause = bool(case.get("simulate_client_pause", True))
    client_prompt_mute_sec = max(0.0, float(case.get("client_prompt_mute_sec", 4.0)))
    client_audio_muted_until = 0.0
    client_recovery_surah = 0
    client_recovery_ayah = 0

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
    state_trace: list[dict[str, Any]] = []
    latencies: list[float] = []
    wall_start = time.perf_counter()

    def add_event(now_t: float, reason: str, expected: tuple[int, int], detected: tuple[int, int], score: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
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
        return event

    def simulate_client_prompt(event: dict[str, Any], now_t: float) -> None:
        nonlocal assisted_surah, assisted_ayah, post_recovery_lock
        nonlocal pending_forward, pending_mistake_key, pending_mistake_count
        nonlocal pending_tail_key, pending_tail_count, client_audio_muted_until
        nonlocal client_recovery_surah, client_recovery_ayah

        if not simulate_client_prompts or not taraweeh:
            return
        if event.get("reason") == "no_context_progress":
            return
        if int(event.get("expected_surah", 0) or 0) <= 0 or int(event.get("expected_ayah", 0) or 0) <= 0:
            return
        if int(event.get("detected_surah", 0) or 0) <= 0 or int(event.get("detected_ayah", 0) or 0) <= 0:
            return
        if float(event.get("score", 0.0) or 0.0) <= 0.0:
            return

        # The Flutter client sends discard_audio + assisted_prompt after a
        # confident mistake, so replay should pin the same recovery target.
        assisted_surah = int(event["expected_surah"])
        assisted_ayah = int(event["expected_ayah"])
        client_recovery_surah = assisted_surah
        client_recovery_ayah = assisted_ayah
        post_recovery_lock = False
        pending_forward = {"surah": 0, "ayah": 0, "count": 0}
        pending_mistake_key = None
        pending_mistake_count = 0
        pending_tail_key = None
        pending_tail_count = 0
        client_audio_muted_until = max(client_audio_muted_until, now_t + client_prompt_mute_sec)

    for end in range(start, len(audio) + 1, stride):
        now_t = end / SAMPLE_RATE
        if now_t < client_audio_muted_until:
            state_trace.append({
                "t": round(now_t, 3),
                "state_surah": current_surah,
                "state_ayah": current_ayah,
                "state_progress": round(float(current_progress or 0.0), 4),
                "matched_surah": 0,
                "matched_ayah": 0,
                "matched_score": 0.0,
                "tracking_mode": "CLIENT_PROMPT_MUTE",
                "mistake_candidate": False,
                "mistake_reason": "",
                "mistake_surah": 0,
                "mistake_ayah": 0,
            })
            continue
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
        state_trace.append({
            "t": round(now_t, 3),
            "state_surah": current_surah,
            "state_ayah": current_ayah,
            "state_progress": round(float(current_progress or 0.0), 4),
            "matched_surah": int(result.get("surah", 0) or 0),
            "matched_ayah": int(result.get("ayah", 0) or 0),
            "matched_score": round(float(result.get("score", 0.0) or 0.0), 4),
            "tracking_mode": result.get("tracking_mode", ""),
            "mistake_candidate": bool(result.get("mistake_candidate")),
            "mistake_reason": result.get("mistake_reason", ""),
            "mistake_surah": int(result.get("mistake_surah", 0) or 0),
            "mistake_ayah": int(result.get("mistake_ayah", 0) or 0),
        })

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

            forward_recovery_ayah = 0
            if (
                taraweeh
                and current_surah > 0
                and matched_surah == current_surah
                and matched_ayah > current_ayah
                and required_forward > 0
                and assisted_ayah == 0
            ):
                same_forward_region = (
                    pending_forward["surah"] == matched_surah
                    and pending_forward["ayah"] > current_ayah
                    and abs(matched_ayah - pending_forward["ayah"]) <= 3
                )
                if same_forward_region:
                    pending_forward["count"] += 1
                    pending_forward["ayah"] = max(pending_forward["ayah"], matched_ayah)
                else:
                    pending_forward = {"surah": matched_surah, "ayah": matched_ayah, "count": 1}
                effective_matched_ayah = max(matched_ayah, pending_forward["ayah"])
                effective_skipped = max(0, effective_matched_ayah - current_ayah - 1)
                effective_forward_distance = backend._forward_section_distance(
                    current_surah,
                    current_ayah,
                    current_progress,
                    effective_matched_ayah,
                    int(result.get("section_index", 0) or 0)
                    if effective_matched_ayah == matched_ayah
                    else 1,
                )
                effective_required = min(
                    required_forward,
                    backend._required_forward_jump_confirmations(
                        effective_skipped,
                        effective_forward_distance,
                        current_progress,
                        float(result.get("score", 0.0) or 0.0),
                    ),
                )
                if pending_forward["count"] >= effective_required and effective_skipped > 0 and now_t >= mistake_cooldown_until:
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
                    elapsed_since_progress = (
                        now_t - last_progress_t
                        if last_progress_t > 0
                        else elapsed_since_ayah_change
                    )
                    elapsed_for_jump = min(elapsed_since_ayah_change, elapsed_since_progress)
                    min_forward_elapsed = backend._minimum_forward_jump_elapsed_seconds(
                        current_surah,
                        current_ayah,
                        effective_matched_ayah,
                    )
                    forward_prompt_eligible = (
                        current_progress >= 0.90
                        and float(result.get("score", 0.0) or 0.0) >= 0.82
                    )
                    if effective_skipped > 8:
                        forward_recovery_ayah = effective_matched_ayah
                        pending_forward = {"surah": 0, "ayah": 0, "count": 0}
                    elif elapsed_for_jump < min_forward_elapsed and forward_prompt_eligible:
                        event = add_event(
                            now_t,
                            "forward_jump",
                            expected,
                            (matched_surah, effective_matched_ayah),
                            float(result.get("score", 0.0) or 0.0),
                            {
                                "confirmations": pending_forward["count"],
                                "required_confirmations": effective_required,
                                "elapsed_since_ayah_change": round(elapsed_since_ayah_change, 3),
                                "elapsed_since_progress": round(elapsed_since_progress, 3),
                                "elapsed_for_jump": round(elapsed_for_jump, 3),
                                "minimum_elapsed": round(min_forward_elapsed, 3),
                                "raw_detected_ayah": matched_ayah,
                            },
                        )
                        simulate_client_prompt(event, now_t)
                        mistake_cooldown_until = now_t + 3.0
                        pending_forward = {"surah": 0, "ayah": 0, "count": 0}
                        failed_matches = 0
                        continue
                    elif elapsed_for_jump < min_forward_elapsed:
                        pending_forward = {"surah": 0, "ayah": 0, "count": 0}
                        failed_matches += 1
                        continue
                    else:
                        forward_recovery_ayah = effective_matched_ayah
                    pending_forward = {"surah": 0, "ayah": 0, "count": 0}
                else:
                    continue
            pending_forward = {"surah": 0, "ayah": 0, "count": 0}

            if forward_recovery_ayah > matched_ayah:
                matched_ayah = forward_recovery_ayah
                recovered_verse = backend.get_verse(matched_surah, matched_ayah) or {}
                matched_word_index = max(
                    matched_word_index,
                    len((recovered_verse.get("phonemes_joined", "") or "").split()),
                )
                next_progress = max(next_progress, 1.0)

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
            if assisted_ayah > 0 and current_surah == assisted_surah and current_ayah == assisted_ayah:
                assisted_surah = 0
                assisted_ayah = 0
                post_recovery_lock = True
            elif post_recovery_lock:
                post_recovery_lock = False
            if (
                client_recovery_ayah > 0
                and (
                    current_surah != client_recovery_surah
                    or current_ayah > client_recovery_ayah
                )
            ):
                client_recovery_surah = 0
                client_recovery_ayah = 0
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
                required_tail = 1 if tail.get("immediate") else 2
                if pending_tail_count >= required_tail:
                    event = add_event(
                        now_t,
                        "same_ayah_tail_mismatch",
                        (current_surah, current_ayah),
                        (current_surah, current_ayah),
                        max(0.01, 1.0 - float(tail["skeleton_ratio"])),
                        {
                            "word_index": tail["word_index"],
                            "expected_word": tail["expected"],
                            "heard_word": tail["got"],
                            "required_confirmations": required_tail,
                            "immediate": bool(tail.get("immediate")),
                        },
                    )
                    simulate_client_prompt(event, now_t)
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
                event = add_event(
                    now_t,
                    mistake_reason,
                    expected,
                    (mistake_surah, mistake_ayah),
                    float(result.get("mistake_score", 0.0) or 0.0),
                    {"confirmations": pending_mistake_count, "required_confirmations": required},
                )
                simulate_client_prompt(event, now_t)
                mistake_cooldown_until = now_t + 3.0
                pending_mistake_key = None
                pending_mistake_count = 0
            continue

        if current_surah > 0 and (
            result.get("speech_detected", True) or simulate_client_pause
        ):
            failed_matches += 1
            required_failures, stall_seconds = backend._stall_detection_thresholds(
                current_surah,
                current_ayah,
                current_progress,
                taraweeh,
            )
            recovery_prompt_owned = (
                client_recovery_surah == current_surah
                and client_recovery_ayah == current_ayah
            )
            if (
                taraweeh
                and failed_matches >= required_failures
                and now_t - last_progress_t >= stall_seconds
                and assisted_ayah == 0
                and has_committed_match
                and now_t >= mistake_cooldown_until
                and not backend._at_surah_end(current_surah, current_ayah)
                and not recovery_prompt_owned
            ):
                expected = backend._expected_prompt_position(
                    current_surah,
                    current_ayah,
                    current_progress,
                    taraweeh,
                )
                add_event(
                    now_t,
                    "no_context_progress" if result.get("speech_detected", True) else "pause",
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
    expected_failures, expected_checks = _validate_expected_corrections(events, case)
    failures.extend(expected_failures)
    unexpected_corrections = _find_unexpected_corrections(events, expected_checks)
    if case.get("fail_on_unexpected_corrections", False) and unexpected_corrections:
        first = unexpected_corrections[0]
        failures.append(
            "Unexpected correction "
            f"{first['reason']} at {first['t']}s "
            f"expected {first['expected_surah']}:{first['expected_ayah']}"
        )
    no_prompt_failures, no_prompt_checks = _validate_expected_non_corrections(events, case)
    failures.extend(no_prompt_failures)
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
        "expect_no_mistake": bool(case.get("expect_no_mistake", False)),
        "duration_sec": round(len(audio) / SAMPLE_RATE, 3),
        "passed": not failures,
        "failures": failures,
        "final_surah": current_surah,
        "final_ayah": current_ayah,
        "correction_events": events,
        "expected_corrections": expected_checks,
        "unexpected_corrections": unexpected_corrections,
        "expected_non_corrections": no_prompt_checks,
        "verse_hits": verse_hits,
        "state_trace": state_trace,
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
    parser.add_argument("--report-out")
    parser.add_argument("--checkpoint-label", default="Regression Report")
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
        for check in result.get("expected_corrections", []):
            status = "PASS" if check.get("matched") else "FAIL"
            print(f"  [{status}] expected: {check.get('label', 'correction')}")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote detailed results to {args.out}")

    report = _build_regression_report(results, args.checkpoint_label)
    print()
    print(report, end="")

    if args.report_out:
        Path(args.report_out).write_text(report, encoding="utf-8")
        print(f"Wrote regression report to {args.report_out}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
