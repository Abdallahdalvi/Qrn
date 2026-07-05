import asyncio
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import librosa
from Levenshtein import ratio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Tarteel Recognition Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Paths
ONNX_MODEL_PATH = Path("assets/web/fastconformer_phoneme_q8.onnx")
QURAN_PHONEMES_PATH = Path("assets/web/quran_phonemes.json")

# Vocab Definition
PHONEME_VOCAB = [
    "a", "u", "i", "A", "U", "I", "aa", "uu", "ii", "AA", "UU", "II",
    "<", "b", "t", "^", "j", "H", "x", "d", "*", "r", "z", "s", "$",
    "S", "D", "T", "Z", "E", "g", "f", "q", "k", "l", "m", "n", "h", "w", "y",
    "<<", "bb", "tt", "^^", "jj", "HH", "xx", "dd", "**", "rr", "zz", "ss", "$$",
    "SS", "DD", "TT", "ZZ", "EE", "gg", "ff", "qq", "kk", "ll", "mm", "nn", "hh", "ww", "yy",
    "|",
]
BLANK_ID = len(PHONEME_VOCAB)  # 69
_BSM_PHONEMES_JOINED = "bismi allahi arraHmaani arraHiimi"

# Mutashabihat (Similar Verses) Database
# (Surah, Ayah) -> List of Similar (Surah, Ayah)
MUTASHABIHAT_DB = {
    (2, 58): [(7, 161)],
    (7, 161): [(2, 58)],
    (2, 136): [(3, 84)],
    (3, 84): [(2, 136)],
    (3, 108): [(2, 252), (45, 6)],
    (2, 252): [(3, 108), (45, 6)],
    (45, 6): [(2, 252), (3, 108)],
}

# Global model state
_onnx_session = None
_verses = []
_by_surah = {}

def load_recognition_engine():
    global _onnx_session, _verses, _by_surah
    if _onnx_session is not None:
        return

    print(f"[Backend] Loading ONNX model from {ONNX_MODEL_PATH}...")
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model file not found at {ONNX_MODEL_PATH}")
    
    _onnx_session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    print(f"[Backend] Loading Quran phonemes from {QURAN_PHONEMES_PATH}...")
    if not QURAN_PHONEMES_PATH.exists():
        raise FileNotFoundError(f"Quran phonemes database not found at {QURAN_PHONEMES_PATH}")

    with open(QURAN_PHONEMES_PATH, "r", encoding="utf-8") as f:
        _verses = json.load(f)

    signature_counts = {}
    for v in _verses:
        joined = v.get("phonemes_joined", "")
        v["_phonemes_joined_ns"] = joined.replace(" ", "")

        # Strip bismillah prefix for ayah 1 (except surahs 1 and 9)
        no_bsm = None
        if v["ayah"] == 1 and v["surah"] != 1 and v["surah"] != 9 and joined.startswith(_BSM_PHONEMES_JOINED):
            no_bsm = joined[len(_BSM_PHONEMES_JOINED):].strip() or None
        v["_phonemes_joined_no_bsm"] = no_bsm
        v["_phonemes_joined_no_bsm_ns"] = no_bsm.replace(" ", "") if no_bsm else None
        signature_key = v["_phonemes_joined_no_bsm_ns"] or v["_phonemes_joined_ns"]
        if signature_key:
            signature_counts[signature_key] = signature_counts.get(signature_key, 0) + 1

        _by_surah.setdefault(v["surah"], []).append(v)

    # Sort each surah's verses by ayah
    for surah_num, verses in _by_surah.items():
        verses.sort(key=lambda x: x["ayah"])
        for index, verse in enumerate(verses):
            verse["_surah_index"] = index
            signature_key = verse.get("_phonemes_joined_no_bsm_ns") or verse.get("_phonemes_joined_ns")
            verse["_ambiguous_signature_count"] = signature_counts.get(signature_key, 0) if signature_key else 0

    print(f"[Backend] Loaded {len(_verses)} verses across {len(_by_surah)} Surahs.")

# Load the engine on startup
load_recognition_engine()

def get_ayah_text(surah: int, ayah: int) -> str:
    if surah <= 0 or ayah <= 0:
        return ""
    verses = _by_surah.get(surah, [])
    for v in verses:
        if v["ayah"] == ayah:
            return v.get("text_uthmani", "")
    return ""

def normalize_ayah_position(surah: int, ayah: int) -> tuple[int, int]:
    surah = min(114, max(1, int(surah or 1)))
    ayah = max(1, int(ayah or 1))
    while surah < 114:
        verses = _by_surah.get(surah, [])
        last_ayah = verses[-1]["ayah"] if verses else 1
        if ayah <= last_ayah:
            break
        ayah -= last_ayah
        surah += 1

    verses = _by_surah.get(surah, [])
    if verses:
        ayah = min(ayah, verses[-1]["ayah"])
    return surah, ayah

def last_ayah_in_surah(surah: int) -> int:
    verses = _by_surah.get(surah, [])
    return verses[-1]["ayah"] if verses else 1

def _at_surah_end(surah: int, ayah: int) -> bool:
    return surah > 0 and ayah >= last_ayah_in_surah(surah)

def _expected_prompt_position(
    current_surah: int,
    current_ayah: int,
    progress_coverage: float,
    taraweeh_mode: bool,
) -> tuple[int, int]:
    if (
        taraweeh_mode
        and current_surah > 0
        and _at_surah_end(current_surah, current_ayah)
    ):
        return current_surah, last_ayah_in_surah(current_surah)
    return normalize_ayah_position(
        current_surah,
        current_ayah
        if progress_coverage < _PROMPT_ADVANCE_COVERAGE
        else current_ayah + 1,
    )

def get_verse(surah: int, ayah: int) -> dict | None:
    return next(
        (v for v in _by_surah.get(surah, []) if v["ayah"] == ayah),
        None,
    )

def _is_ambiguous_signature_verse(surah: int, ayah: int) -> bool:
    verse = get_verse(surah, ayah)
    if not verse:
        return False
    return int(verse.get("_ambiguous_signature_count", 0) or 0) > 1

_QURAN_STOP_SIGNS = set("ۖۗۘۙۚۛۜ۝")
_LONG_AYAH_WORD_THRESHOLD = 36
_PROMPT_ADVANCE_COVERAGE = 0.92
_PROGRESS_STALE_DELTA = 0.01
_REWIND_TRANSITION_GRACE_SECONDS = 7.0

def _word_count(text: str) -> int:
    return len([w for w in (text or "").split() if w.strip()])

def _compact_char_count(text: str) -> int:
    return len((text or "").replace(" ", ""))

def _is_ambiguous_short_phrase(phoneme_text: str, verse: dict | None = None) -> bool:
    words = _word_count(phoneme_text)
    chars = _compact_char_count(phoneme_text)
    verse_words = _word_count((verse or {}).get("phonemes_joined", "")) or _word_count(
        (verse or {}).get("text_uthmani", "")
    )

    if words <= 3 or chars < 18:
        return True
    if words <= 4 and chars < 24:
        return True
    if verse_words <= 4 and chars < 28:
        return True
    return False

def _target_section_count(total_words: int) -> int:
    if total_words <= 0:
        return 0
    if total_words <= 5:
        return 2
    if total_words <= 11:
        return 3
    if total_words <= 22:
        return 4
    if total_words <= 32:
        return 5
    if total_words <= 45:
        return 6
    if total_words <= 70:
        return 8
    return int(min(18, max(9, math.ceil(total_words / 9))))

def _adaptive_section_boundaries(verse: dict) -> list[int]:
    words = verse.get("phoneme_words") or [
        w for w in verse.get("phonemes_joined", "").split() if w.strip()
    ]
    total_words = len(words)
    if total_words == 0:
        return []

    section_count = min(total_words, _target_section_count(total_words))
    if section_count <= 1:
        return [total_words]

    stop_positions = [
        idx
        for idx, word in enumerate(words, start=1)
        if any(sign in word for sign in _QURAN_STOP_SIGNS)
    ]

    boundaries = []
    previous = 0
    for section_idx in range(1, section_count):
        ideal = int(round(total_words * section_idx / section_count))
        remaining_sections = section_count - section_idx
        min_boundary = previous + 1
        max_boundary = total_words - remaining_sections

        chosen = None
        best_distance = None
        for stop in stop_positions:
            if stop < min_boundary or stop > max_boundary:
                continue
            distance = abs(stop - ideal)
            if best_distance is None or distance < best_distance:
                chosen = stop
                best_distance = distance

        if chosen is None:
            chosen = max(min_boundary, min(max_boundary, ideal))

        if boundaries and chosen <= boundaries[-1]:
            chosen = min(max_boundary, boundaries[-1] + 1)

        if chosen >= total_words:
            break
        boundaries.append(chosen)
        previous = chosen

    if not boundaries or boundaries[-1] != total_words:
        boundaries.append(total_words)
    return boundaries

def _section_progress_for_word(verse: dict, word_index: int = 0) -> dict:
    boundaries = _adaptive_section_boundaries(verse)
    total_sections = len(boundaries)
    if total_sections == 0:
        return {
            "section": 0,
            "total_sections": 0,
            "coverage": 0.0,
            "section_start_word": 1,
            "section_end_word": 0,
        }

    safe_word = max(0, int(word_index or 0))
    completed = 0
    previous_boundary = 0
    for idx, boundary in enumerate(boundaries):
        span = max(1, boundary - previous_boundary)
        tolerance = 0 if idx == total_sections - 1 else (1 if span >= 4 else 0)
        if safe_word >= boundary - tolerance:
            completed += 1
        else:
            break
        previous_boundary = boundary

    active_section = min(total_sections, completed + (1 if safe_word > 0 else 0))
    start_word = 1 if active_section <= 1 else boundaries[active_section - 2] + 1
    end_word = boundaries[min(max(active_section, 1), total_sections) - 1]
    return {
        "section": active_section,
        "total_sections": total_sections,
        "coverage": round(completed / total_sections, 4),
        "section_start_word": start_word,
        "section_end_word": end_word,
    }

def _forward_section_distance(
    surah: int,
    current_ayah: int,
    current_coverage: float,
    matched_ayah: int,
    matched_section: int = 0,
) -> int:
    if surah <= 0 or matched_ayah <= current_ayah:
        return 0

    distance = 0
    current_verse = get_verse(surah, current_ayah)
    if current_verse:
        current_total = len(_adaptive_section_boundaries(current_verse))
        current_completed = int(round(max(0.0, min(1.0, current_coverage)) * current_total))
        distance += max(0, current_total - current_completed)

    for ayah in range(current_ayah + 1, matched_ayah):
        verse = get_verse(surah, ayah)
        distance += len(_adaptive_section_boundaries(verse)) if verse else 0

    if matched_section > 0:
        distance += matched_section
    else:
        matched_verse = get_verse(surah, matched_ayah)
        distance += min(1, len(_adaptive_section_boundaries(matched_verse))) if matched_verse else 1

    return distance

def _overall_progress_coverage(word_info: dict, section_info: dict) -> float:
    return round(
        max(
            float(word_info.get("coverage", 0.0) or 0.0),
            float(section_info.get("coverage", 0.0) or 0.0),
        ),
        4,
    )

def _stall_detection_thresholds(
    surah: int,
    ayah: int,
    progress_coverage: float,
    taraweeh_mode: bool,
) -> tuple[int, float]:
    required_failures = 3
    stall_seconds = 4.0
    if not taraweeh_mode:
        return required_failures, stall_seconds

    required_failures = 4
    stall_seconds = 5.5
    verse = get_verse(surah, ayah) or {}
    words = _word_count(verse.get("text_uthmani", ""))
    if ayah == 1 and words <= 2:
        stall_seconds = 8.5
    elif words <= 2:
        stall_seconds = 7.0
    elif words >= 10:
        stall_seconds = 7.0
    elif words >= 6:
        stall_seconds = 6.0

    if progress_coverage >= 0.65:
        stall_seconds += 0.75

    return required_failures, stall_seconds

def _transition_grace_seconds(
    surah: int,
    ayah: int,
    progress_coverage: float,
) -> float:
    verse = get_verse(surah, ayah) or {}
    words = _word_count(verse.get("text_uthmani", ""))
    grace = 5.0
    if ayah == 1 and words <= 2:
        grace = 10.5
    elif words <= 2:
        grace = 8.5
    elif words >= 12:
        grace = 8.0
    elif words >= 8:
        grace = 6.5

    if progress_coverage >= 0.90:
        grace += 1.0
    elif progress_coverage >= 0.75:
        grace += 0.5

    return grace

def _required_forward_jump_confirmations(
    skipped_ayahs: int,
    forward_section_distance: int,
    progress_coverage: float,
    score: float,
) -> int:
    if skipped_ayahs <= 0:
        if forward_section_distance > 8 and score < 0.82:
            return 2
        return 0

    required = 3
    if skipped_ayahs >= 2 or forward_section_distance > 10:
        required = 4
    elif progress_coverage >= 0.95 and score >= 0.92:
        required = 2
    elif progress_coverage >= 0.90 and score >= 0.88:
        required = 2

    return required

def _phrase_boundary_for_luqmah(verse: dict, word_index: int = 0) -> dict:
    text = verse.get("text_uthmani", "")
    words = [w for w in text.split() if w.strip()]
    total_words = len(words)
    safe_word_index = max(0, min(int(word_index or 0), total_words))
    if total_words < _LONG_AYAH_WORD_THRESHOLD or safe_word_index <= 0:
        return {
            "start_word": 1,
            "phrase_text": text,
            "strategy": "whole_ayah",
            "estimated_start_ms": 0,
        }

    candidate = 0
    for idx, word in enumerate(words[:safe_word_index], start=1):
        if any(sign in word for sign in _QURAN_STOP_SIGNS):
            candidate = idx

    # If the nearest stop is too close to the current point, step back one more
    # phrase so Luqmah has enough context to be useful.
    if candidate and safe_word_index - candidate < 4:
        previous = 0
        for idx, word in enumerate(words[:max(candidate - 1, 0)], start=1):
            if any(sign in word for sign in _QURAN_STOP_SIGNS):
                previous = idx
        candidate = previous or candidate

    if candidate <= 0:
        return {
            "start_word": 1,
            "phrase_text": text,
            "strategy": "whole_ayah_long_no_boundary",
            "estimated_start_ms": 0,
        }

    start_word = min(total_words, candidate + 1)
    # Timing is an estimate until the app has a per-word recitation timing map.
    estimated_start_ms = max(0, int((start_word - 1) * 620) - 700)
    return {
        "start_word": start_word,
        "phrase_text": " ".join(words[start_word - 1:]),
        "strategy": "phrase_boundary",
        "estimated_start_ms": estimated_start_ms,
    }

def build_luqmah_verses(surah: int, ayah: int, count: int = 1, word_index: int = 0) -> list[dict]:
    """Return a configurable continuous Luqmah prompt."""
    prompt_surah, prompt_ayah = normalize_ayah_position(surah, ayah)
    count = min(3, max(1, int(count or 1)))

    selected = []
    seen = set()
    for offset in range(count):
        verse_surah, verse_ayah = normalize_ayah_position(
            prompt_surah,
            prompt_ayah + offset,
        )
        key = (verse_surah, verse_ayah)
        if key in seen:
            break
        verse = get_verse(verse_surah, verse_ayah)
        if not verse:
            break
        selected.append(verse)
        seen.add(key)

    prompt = []
    for index, verse in enumerate(selected):
        boundary = _phrase_boundary_for_luqmah(verse, word_index if index == 0 else 0)
        prompt.append({
            "surah": verse["surah"],
            "ayah": verse["ayah"],
            "ayah_text": verse.get("text_uthmani", ""),
            "prompt_text": boundary["phrase_text"],
            "start_word": boundary["start_word"],
            "prompt_strategy": boundary["strategy"],
            "estimated_start_ms": boundary["estimated_start_ms"],
            "total_words": _word_count(verse.get("text_uthmani", "")),
        })
    return prompt

def _collect_window(surah: int, start_ayah: int, end_ayah: int) -> list[dict]:
    verses = _by_surah.get(surah, [])
    if not verses:
        return []
    last_ayah = verses[-1]["ayah"]
    start_ayah = max(1, start_ayah)
    end_ayah = min(last_ayah, end_ayah)
    if end_ayah < start_ayah:
        return []
    return [v for v in verses if start_ayah <= v["ayah"] <= end_ayah]

def _merge_verse_windows(*windows: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for window in windows:
        for verse in window:
            key = (verse["surah"], verse["ayah"])
            if key in seen:
                continue
            merged.append(verse)
            seen.add(key)
    return merged

# Levenshtein Matching Logic
def semi_global_distance(query: str, ref: str) -> int:
    if not query:
        return 0
    if not ref:
        return len(query)
    m, n = len(query), len(ref)
    prev = list(range(m + 1))
    best = prev[m]
    for j in range(1, n + 1):
        curr = [0] * (m + 1)
        for i in range(1, m + 1):
            cost = 0 if query[i - 1] == ref[j - 1] else 1
            curr[i] = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
        best = min(best, curr[m])
        prev = curr
    return best

def fragment_score(query: str, ref: str) -> float:
    if not query:
        return 1.0
    return max(0.0, 1.0 - semi_global_distance(query, ref) / len(query))

import re
def _squash(text: str) -> str:
    return re.sub(r'(.)\1+', r'\1', text.replace(' ', ''))

def robust_ratio(query: str, ref: str) -> float:
    raw = ratio(query, ref)
    sq_query = _squash(query)
    sq_ref = _squash(ref)
    if not sq_query or not sq_ref:
        return raw
    return max(raw, ratio(sq_query, sq_ref))

def _phoneme_word_confidence(got: str, expected: str) -> float:
    base = robust_ratio(got or "", expected or "")
    expected_key = _squash(expected or "")
    got_key = _squash(got or "")
    if len(expected_key) < 5 or not got_key:
        return base

    # ASR often glues multiple recited words together. If the expected word is
    # inside that longer chunk, treat it as recognized instead of warning.
    fragment = fragment_score(expected_key, got_key)
    if expected_key in got_key:
        fragment = max(fragment, 0.98)
    return max(base, fragment)

def _is_embedded_word_match(expected: str, got: str) -> bool:
    expected_key = _squash(expected or "")
    got_key = _squash(got or "")
    if len(expected_key) < 5 or len(got_key) <= len(expected_key):
        return False
    return _phoneme_word_confidence(got, expected) >= 0.82

def _phoneme_skeleton(text: str) -> str:
    """Drop vowel-like phonemes so wrong consonant tails stand out."""
    return re.sub(r"[aeiouAUI]", "", _squash(text or ""))

def _same_ayah_tail_mismatch(result: dict) -> dict | None:
    expected = result.get("word_expected", "")
    got = result.get("word_got", "")
    word_index = int(result.get("word_index", 0) or 0)
    total_words = int(result.get("total_words", 0) or 0)
    surah = int(result.get("surah", 0) or 0)
    ayah = int(result.get("ayah", 0) or 0)
    progress_coverage = float(result.get("progress_coverage", 0.0) or 0.0)

    if not (
        surah > 0
        and ayah > 0
        and total_words >= 4
        and word_index >= total_words
        and progress_coverage >= 0.72
        and expected
        and got
    ):
        return None

    if _is_ambiguous_signature_verse(surah, ayah):
        return None

    expected_key = _squash(expected)
    expected_skeleton = _phoneme_skeleton(expected)
    if len(expected_key) < 5 or len(expected_skeleton) < 3:
        return None

    transcript_words = result.get("transcript", "").strip().split()
    candidates = [got]
    for tail_len in (2, 3):
        if len(transcript_words) >= tail_len:
            candidates.append("".join(transcript_words[-tail_len:]))

    best_strict_ratio = 0.0
    best_skeleton_ratio = 0.0
    for candidate in candidates:
        candidate_key = _squash(candidate)
        candidate_skeleton = _phoneme_skeleton(candidate)
        if not candidate_key or len(candidate_key) < 4:
            continue
        best_strict_ratio = max(best_strict_ratio, ratio(candidate_key, expected_key))
        if candidate_skeleton:
            best_skeleton_ratio = max(
                best_skeleton_ratio,
                ratio(candidate_skeleton, expected_skeleton),
            )

    # Overall ayah matching can hide a bad final word because nearby words raise
    # sequence confidence. Require the consonant skeleton to disagree strongly,
    # and confirm it twice in the websocket loop before playing Luqmah.
    if best_strict_ratio >= 0.78 or best_skeleton_ratio >= 0.70:
        return None

    return {
        "word_index": word_index,
        "expected": expected,
        "got": got,
        "strict_ratio": round(best_strict_ratio, 4),
        "skeleton_ratio": round(best_skeleton_ratio, 4),
    }

def _should_emit_word_correction(result: dict, word_confidence: float) -> bool:
    expected = result.get("word_expected", "")
    got = result.get("word_got", "")
    if not (
        result.get("word_index", 0) > 0
        and 0.0 < word_confidence < 0.55
        and expected
        and got
    ):
        return False

    expected_len = len(_squash(expected))
    got_len = len(_squash(got))
    if expected_len <= 3:
        return False
    if got_len < max(4, math.ceil(expected_len * 0.55)):
        return False
    if _is_embedded_word_match(expected, got):
        return False

    return True

def _short_query_boost(no_space_text: str, verse: dict, use_no_bsm: bool = False) -> float:
    if use_no_bsm:
        candidate = verse.get("_phonemes_joined_no_bsm_ns", "") or verse.get("_phonemes_joined_ns", "")
    else:
        candidate = verse.get("_phonemes_joined_ns", "")
    if not candidate:
        return 0.0

    prefix_window = min(len(candidate), len(no_space_text) + 6)
    prefix = robust_ratio(no_space_text, candidate[:prefix_window])

    if use_no_bsm:
        joined = verse.get("_phonemes_joined_no_bsm", "") or ""
    else:
        joined = verse.get("phonemes_joined", "")
    first_word = joined.split(" ")[0] if joined else ""
    first_word_score = robust_ratio(no_space_text, first_word) if first_word else 0.0

    return max(prefix, first_word_score)

def _match_phoneme_text(phoneme_text: str, top_k: int = 5, search_verses: list = None) -> list[dict]:
    if not phoneme_text.strip():
        return []

    no_space_text = phoneme_text.replace(" ", "")

    # Pass 1: Global score + short-query boost
    scored = []
    target_verses = search_verses if search_verses is not None else _verses
    for verse in target_verses:
        ref = verse.get("phonemes_joined", "")
        if not ref:
            continue
        raw = robust_ratio(phoneme_text, ref)

        if len(no_space_text) <= 10:
            raw = max(raw, _short_query_boost(no_space_text, verse))

        no_bsm = verse.get("_phonemes_joined_no_bsm")
        if no_bsm:
            raw = max(raw, robust_ratio(phoneme_text, no_bsm))
            if len(no_space_text) <= 10:
                raw = max(raw, _short_query_boost(no_space_text, verse, use_no_bsm=True))

        scored.append([verse, raw, raw])

    scored.sort(key=lambda x: x[2], reverse=True)

    # Pass 1.5: fragment boost
    if len(no_space_text) >= 8:
        resorted = False
        # Semi-global distance is intentionally limited to the strongest raw
        # candidates. Running its Python dynamic-programming loop across all
        # 6,236 ayahs stalls the WebSocket and causes delayed tracking.
        for i, (verse, raw, _) in enumerate(scored[:160]):
            ref_ns = verse.get("_phonemes_joined_ns", "")
            if not ref_ns:
                continue
            if len(no_space_text) >= len(ref_ns) * 0.8:
                continue

            frag = fragment_score(no_space_text, ref_ns)
            no_bsm_ns = verse.get("_phonemes_joined_no_bsm_ns")
            if no_bsm_ns:
                frag = max(frag, fragment_score(no_space_text, no_bsm_ns))

            if frag > raw:
                boosted = raw + (frag - raw) * 0.7
                scored[i] = [verse, boosted, boosted]
                resorted = True

        if resorted:
            scored.sort(key=lambda x: x[2], reverse=True)

    # Pass 2: multi-verse spans (up to 3 verses). Only evaluate spans around
    # strong single-verse candidates instead of exhaustively rebuilding every
    # possible span in the Quran on every inference.
    span_results = []
    candidate_starts = set()
    for verse, _, _ in scored[:32]:
        index = verse.get("_surah_index", 0)
        candidate_starts.add((verse["surah"], max(0, index - 1)))
        candidate_starts.add((verse["surah"], index))

    for surah_num, i in candidate_starts:
        verses = _by_surah.get(surah_num, [])
        for span in range(2, 4):
            if i + span > len(verses):
                break
            chunk = verses[i:i + span]
            first_phonemes = chunk[0].get("_phonemes_joined_no_bsm") or chunk[0].get("phonemes_joined", "")
            span_phonemes = first_phonemes + " " + " ".join(
                v.get("phonemes_joined", "") for v in chunk[1:]
            )
            raw = robust_ratio(phoneme_text, span_phonemes)
            span_results.append({
                "surah": surah_num,
                "ayah": chunk[0]["ayah"],
                "ayah_end": chunk[-1]["ayah"],
                "score": round(raw, 4),
                "surah_name": chunk[0]["surah_name"],
                "surah_name_en": chunk[0]["surah_name_en"],
                "phonemes_joined": span_phonemes,
            })

    # Combine singles + spans
    singles = []
    for verse, raw, boosted in scored[:32]:
        singles.append({
            "surah": verse["surah"],
            "ayah": verse["ayah"],
            "ayah_end": None,
            "score": round(boosted, 4),
            "surah_name": verse["surah_name"],
            "surah_name_en": verse["surah_name_en"],
            "phonemes_joined": verse.get("phonemes_joined", ""),
        })

    combined = singles + span_results
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]

def _greedy_decode_phonemes(logprobs: np.ndarray) -> str:
    ids = logprobs.argmax(axis=1)
    prev = -1
    tokens = []
    for idx in ids:
        if idx != prev and idx != BLANK_ID:
            if idx < len(PHONEME_VOCAB):
                tokens.append(PHONEME_VOCAB[idx])
        prev = idx

    words = []
    cur = []
    for t in tokens:
        if t == "|":
            if cur:
                words.append("".join(cur))
            cur = []
        else:
            cur.append(t)
    if cur:
        words.append("".join(cur))
    return " ".join(words)

def _compute_logprobs(audio: np.ndarray) -> np.ndarray:
    audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    audio = audio - float(np.mean(audio))

    # Conservative normalization improves quiet recitation without turning
    # room noise into speech or clipping strong consonants.
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms > 1e-6:
        target_rms = 0.08
        gain = min(4.0, max(0.5, target_rms / rms))
        audio = np.clip(audio * gain, -0.98, 0.98)

    # NeMo-compatible mel spectrogram extraction
    # Preemphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    mel = librosa.feature.melspectrogram(
        y=audio, sr=16000, n_fft=512, hop_length=160, win_length=400,
        n_mels=80, fmax=8000, htk=True, norm="slaney",
    )
    mel = np.log(mel + 1e-5)
    mel = (mel - mel.mean(axis=1, keepdims=True)) / (mel.std(axis=1, keepdims=True) + 1e-10)

    features = mel.astype(np.float32)[np.newaxis]  # [1, 80, T]
    length = np.array([mel.shape[1]], dtype=np.int64)

    input_names = [inp.name for inp in _onnx_session.get_inputs()]
    results = _onnx_session.run(None, {
        input_names[0]: features,
        input_names[1]: length,
    })

    logprobs = results[0]  # [1, T, vocab_size]
    return logprobs[0]  # [T, vocab_size]

def get_word_position(transcript: str, verse_text: str) -> dict:
    t_words = transcript.strip().split()
    v_words = verse_text.strip().split()
    if not t_words or not v_words:
        return {
            "word": 0,
            "total": len(v_words),
            "confidence": 0.0,
            "coverage": 0.0,
            "expected": "",
            "got": "",
        }

    best_score = -1.0
    best_idx = 0
    best_word_conf = 0.0
    best_sequence_conf = 0.0
    tail_len = min(len(t_words), 4)

    for i in range(len(v_words)):
        word_conf = _phoneme_word_confidence(t_words[-1], v_words[i])
        score = word_conf

        weighted = 0.0
        total_weight = 0.0
        for back in range(tail_len):
            t_idx = len(t_words) - 1 - back
            v_idx = i - back
            if t_idx < 0 or v_idx < 0:
                break
            conf = _phoneme_word_confidence(t_words[t_idx], v_words[v_idx])
            weight = 1.0 if back == 0 else 0.7
            weighted += conf * weight
            total_weight += weight
        sequence_conf = weighted / total_weight if total_weight > 0 else word_conf
        score += sequence_conf * 0.45

        if score > best_score:
            best_score = score
            best_idx = i
            best_word_conf = word_conf
            best_sequence_conf = sequence_conf

    coverage = (best_idx + 1) / len(v_words)
    return {
        "word": best_idx + 1,
        "total": len(v_words),
        "confidence": round(max(best_word_conf, best_sequence_conf), 4),
        "coverage": round(coverage, 4),
        "expected": v_words[best_idx] if 0 <= best_idx < len(v_words) else "",
        "got": t_words[-1],
    }

def get_search_context(current_surah: int, current_ayah: int, failed_matches: int, taraweeh_mode: bool, assisted_surah: int = 0, assisted_ayah: int = 0, post_recovery_lock: bool = False) -> tuple[list, str, str]:
    search_surah = assisted_surah if assisted_surah > 0 and assisted_ayah > 0 else current_surah
    search_ayah = assisted_ayah if assisted_surah > 0 and assisted_ayah > 0 else current_ayah
    if search_surah == 0 or search_ayah == 0:
        return _verses, "GLOBAL_SEARCH", "Entire Quran"
        
    surah_verses = _by_surah.get(search_surah, [])
    if not surah_verses:
        return _verses, "GLOBAL_SEARCH", "Entire Quran"
        
    last_ayah_in_surah = surah_verses[-1]["ayah"]
    
    if assisted_ayah > 0:
        assisted_start = max(1, assisted_ayah - 8) if taraweeh_mode else max(1, assisted_ayah - 2)
        assisted_end = assisted_ayah + 2
        assisted_window = _collect_window(assisted_surah, assisted_start, assisted_end)
        if taraweeh_mode and current_surah > 0 and current_ayah > 0:
            recovery_start = max(1, current_ayah - 10)
            recovery_end = current_ayah + 2
            recovery_window = _collect_window(current_surah, recovery_start, recovery_end)
            merged = _merge_verse_windows(assisted_window, recovery_window)
            if merged:
                return merged, "ASSISTED_TRACKING_RECOVERY", (
                    f"Prompt {assisted_surah}:{assisted_start}-{assisted_end} + "
                    f"Recovery {current_surah}:{recovery_start}-{recovery_end}"
                )
        return assisted_window, "ASSISTED_TRACKING", f"Surah {assisted_surah}: {assisted_start}-{assisted_end}"
    elif post_recovery_lock:
        start_ayah = max(1, search_ayah - 1)
        end_ayah = search_ayah + 2
        mode = "RECOVERY_CONFIRMED"
    elif failed_matches == 0:
        start_ayah = search_ayah
        end_ayah = search_ayah + 4
        mode = "NORMAL"
    elif failed_matches == 1:
        start_ayah = max(1, search_ayah - 8) if taraweeh_mode else max(1, search_ayah - 2)
        end_ayah = search_ayah + 6
        mode = "EXPANDED_WINDOW_1"
    elif failed_matches == 2:
        start_ayah = max(1, search_ayah - 12) if taraweeh_mode else max(1, search_ayah - 4)
        end_ayah = search_ayah + 8
        mode = "EXPANDED_WINDOW_2"
    elif failed_matches == 3:
        if (
            taraweeh_mode
            and (
                (search_surah, search_ayah) in MUTASHABIHAT_DB
                or _is_ambiguous_signature_verse(search_surah, search_ayah)
            )
        ):
            # STRICT LOCKDOWN: Do not allow full surah search if we are in a mutashabihat verse.
            # This prevents jumping to the similar verse in the other surah.
            start_ayah = max(1, search_ayah - 12)
            end_ayah = search_ayah + 8
            mode = "MUTASHABIHAT_LOCKED"
        else:
            start_ayah = 1
            end_ayah = last_ayah_in_surah
            mode = "SURAH_SEARCH"
    else:
        if taraweeh_mode:
            return surah_verses, "SURAH_SEARCH", f"Surah {search_surah} Only"
        else:
            return _verses, "GLOBAL_SEARCH", "Entire Quran"
            
    local_verses = []
    allow_surah_overflow = (
        end_ayah > last_ayah_in_surah
        and search_surah < 114
        and not taraweeh_mode
    )
    if allow_surah_overflow:
        for v in surah_verses:
            if v["ayah"] >= start_ayah:
                local_verses.append(v)
        next_surah_verses = _by_surah.get(search_surah + 1, [])
        overflow = end_ayah - last_ayah_in_surah
        for v in next_surah_verses:
            if v["ayah"] <= overflow:
                local_verses.append(v)
        window_str = f"Surah {search_surah}:{max(1, start_ayah)} -> Surah {search_surah + 1}:{overflow}"
    else:
        for v in surah_verses:
            if start_ayah <= v["ayah"] <= end_ayah:
                local_verses.append(v)
        window_str = f"Surah {search_surah}: {max(1, start_ayah)}-{min(end_ayah, last_ayah_in_surah)}"
        
    return local_verses, mode, window_str

def predict_audio(audio: np.ndarray, current_surah: int = 0, current_ayah: int = 0, failed_matches: int = 0, taraweeh_mode: bool = False, assisted_surah: int = 0, assisted_ayah: int = 0, post_recovery_lock: bool = False) -> dict:
    metrics = {"audio_len": round(len(audio) / 16000, 3)}
    t_start = time.perf_counter()

    if len(audio) < 16000 * 1.5:  # Min 1.5 seconds (was 2.0)
        return {"surah": 0, "ayah": 0, "score": 0.0, "transcript": "", "error": "Listening for recitation", "speech_detected": False, "metrics": metrics}

    clean_audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    clean_audio = clean_audio - float(np.mean(clean_audio))
    raw_rms = float(np.sqrt(np.mean(clean_audio**2)))
    raw_peak = float(np.max(np.abs(clean_audio)))
    raw_rms_db = float(20.0 * np.log10(max(raw_rms, 1e-8)))
    metrics["rms_db"] = round(raw_rms_db, 2)
    metrics["peak"] = round(raw_peak, 4)
    if raw_rms_db < -52.0 or raw_peak < 0.004:
        return {"surah": 0, "ayah": 0, "score": 0.0, "transcript": "", "error": "Listening for recitation", "speech_detected": False, "metrics": metrics}

    t0 = time.perf_counter()
    logprobs = _compute_logprobs(clean_audio)
    metrics["inference_time"] = round(time.perf_counter() - t0, 3)
    
    phoneme_text = _greedy_decode_phonemes(logprobs)
    print(f"[Backend] Decoded Phonemes ({len(phoneme_text)} chars): {phoneme_text}")
    
    no_space_text = phoneme_text.replace(" ", "")
    if len(no_space_text) < 7:  # Min 7 decoded characters (was 10)
        return {"surah": 0, "ayah": 0, "score": 0.0, "transcript": phoneme_text, "error": "Keep reciting", "speech_detected": True, "metrics": metrics}
    
    t0 = time.perf_counter()
    # Dynamic Search Context
    search_verses, tracking_mode, window_str = get_search_context(current_surah, current_ayah, failed_matches, taraweeh_mode, assisted_surah, assisted_ayah, post_recovery_lock)
    
    # print(f"[Backend] Tracking: {tracking_mode} | Window: {window_str}")
    top_matches = _match_phoneme_text(phoneme_text, top_k=5, search_verses=search_verses)
    
    # Apply score weighting if we are in Assisted Tracking mode
    if assisted_ayah > 0:
        for match in top_matches:
            distance = abs(match["ayah"] - assisted_ayah) if match["surah"] == assisted_surah else 99
            if distance == 1:
                match["score"] *= 0.95
            elif distance == 2:
                match["score"] *= 0.90
            elif distance > 2:
                match["score"] *= 0.80
        top_matches.sort(key=lambda x: x["score"], reverse=True)
        
    best_match = None
    for match in top_matches:
        is_jump = False
        if tracking_mode in ["SURAH_SEARCH", "GLOBAL_SEARCH", "SURAH_TRANSITION"]:
            if current_surah > 0 and current_ayah > 0:
                current_surah_verses = _by_surah.get(current_surah, [])
                last_current_ayah = current_surah_verses[-1]["ayah"] if current_surah_verses else current_ayah
                if match["surah"] == current_surah:
                    distance = match["ayah"] - current_ayah
                    # Penalize backward jumps or forward jumps > 1 ayah
                    if distance < 0 or distance > 1:
                        is_jump = True
                elif match["surah"] == current_surah + 1 and tracking_mode == "SURAH_TRANSITION":
                    is_jump = current_ayah < last_current_ayah
                elif taraweeh_mode:
                    # STRICT RULE: Never jump to a different Surah during Taraweeh mode unless transitioning
                    is_jump = True
                    match["score"] = 0.0 # Force reject
                else:
                    is_jump = True
                    
        # Anti-jump protection
        if is_jump and match["score"] < 0.85:
            print(f"[Backend] Rejecting jump to {match['surah']}:{match['ayah']} due to low confidence ({match['score']:.2f})")
            continue
            
        minimum_score = 0.68 if tracking_mode == "GLOBAL_SEARCH" else 0.60
        if match["score"] >= minimum_score:
            best_match = match
            break
    metrics["matching_time"] = round(time.perf_counter() - t0, 3)
    metrics["total_latency"] = round(time.perf_counter() - t_start, 3)
    
    if not best_match:
        # --- Mistake / Restart Fallback ---
        if taraweeh_mode and current_surah > 0 and tracking_mode != "GLOBAL_SEARCH":
            global_matches = _match_phoneme_text(phoneme_text, top_k=1, search_verses=_verses)
            if global_matches and global_matches[0]["score"] > 0.75:
                gm = global_matches[0]
                gm_verse = get_verse(gm["surah"], gm["ayah"]) or {}
                duration_sec = len(audio) / 16000
                current_surah_verses = _by_surah.get(current_surah, [])
                last_current_ayah = (
                    current_surah_verses[-1]["ayah"]
                    if current_surah_verses
                    else current_ayah
                )
                at_taraweeh_surah_end = current_ayah >= last_current_ayah

                # Ignore weak global matches on short audio segments (Tajweed noise)
                if duration_sec < 3.0 and gm["score"] < 0.82:
                    pass
                elif at_taraweeh_surah_end and gm["surah"] != current_surah:
                    pass
                elif (
                    gm["surah"] != current_surah
                    and _is_ambiguous_short_phrase(phoneme_text, gm_verse)
                ):
                    pass
                else:
                    ayah_delta = gm["ayah"] - current_ayah if gm["surah"] == current_surah else 99
                    if gm["surah"] == current_surah and -16 <= ayah_delta <= 3:
                        best_match = gm
                        tracking_mode = "GLOBAL_RECOVERY"
                        window_str = f"Recovered in Surah {current_surah}"
                    elif gm["score"] >= 0.84:
                        print(f"[Backend] Immediate mistake near {current_surah}:{current_ayah}; globally matched {gm['surah']}:{gm['ayah']} (Conf: {gm['score']:.2f})")
                        return {
                            "surah": 0, "ayah": 0, "score": 0.0, "transcript": phoneme_text,
                            "error": "Mistake detected", "speech_detected": True,
                            "tracking_mode": tracking_mode, "search_window": window_str,
                            "metrics": metrics,
                            "mistake_candidate": True,
                            "mistake_surah": gm["surah"],
                            "mistake_ayah": gm["ayah"],
                            "mistake_score": gm["score"],
                            "mistake_immediate": True
                        }
                    elif gm["surah"] != current_surah or abs(ayah_delta) > 2:
                        print(f"[Backend] Mistake candidate near {current_surah}:{current_ayah}; globally matched {gm['surah']}:{gm['ayah']} (Conf: {gm['score']:.2f})")
                        return {
                        "surah": 0, "ayah": 0, "score": 0.0, "transcript": phoneme_text,
                        "error": "Possible mistake", "speech_detected": True,
                        "tracking_mode": tracking_mode, "search_window": window_str,
                        "metrics": metrics,
                        "mistake_candidate": True,
                        "mistake_surah": gm["surah"],
                        "mistake_ayah": gm["ayah"],
                        "mistake_score": gm["score"]
                    }
        # ----------------------------------
    if not best_match:
        return {"surah": 0, "ayah": 0, "score": 0.0, "transcript": phoneme_text, "error": "Keep reciting", "speech_detected": True, "tracking_mode": tracking_mode, "search_window": window_str, "metrics": metrics}

    best = best_match
    word_info = get_word_position(phoneme_text, best.get("phonemes_joined", ""))

    active_surah = best["surah"]
    active_ayah = best["ayah"]
    active_word_index = word_info.get("word", 0)

    if best.get("ayah_end"):
        accumulated_words = 0
        for a in range(best["ayah"], best["ayah_end"] + 1):
            v = get_verse(best["surah"], a)
            if v:
                v_words = len(v.get("phonemes_joined", "").split())
                if active_word_index <= accumulated_words + v_words:
                    active_ayah = a
                    active_word_index = max(0, active_word_index - accumulated_words)
                    break
                accumulated_words += v_words

    verse_for_sections = get_verse(active_surah, active_ayah) or {}
    section_info = _section_progress_for_word(
        verse_for_sections,
        active_word_index,
    )
    progress_coverage = _overall_progress_coverage(word_info, section_info)
        
    return {
        "surah": active_surah,
        "ayah": active_ayah,
        "ayah_end": None,
        "surah_name_en": best["surah_name_en"],
        "surah_name": best["surah_name"],
        "score": best["score"],
        "transcript": phoneme_text,
        "word_index": active_word_index,
        "total_words": word_info["total"],
        "word_confidence": word_info["confidence"],
        "word_coverage": word_info.get("coverage", 0.0),
        "section_index": section_info["section"],
        "total_sections": section_info["total_sections"],
        "section_coverage": section_info["coverage"],
        "progress_coverage": progress_coverage,
        "section_start_word": section_info["section_start_word"],
        "section_end_word": section_info["section_end_word"],
        "word_expected": word_info.get("expected", ""),
        "word_got": word_info.get("got", ""),
        "is_mutashabihat": (active_surah, active_ayah) in MUTASHABIHAT_DB,
        "speech_detected": True,
        "tracking_mode": tracking_mode,
        "search_window": window_str,
        "metrics": metrics
    }

# WebSocket Streaming Endpoint
@app.websocket("/ws/recitation")
async def websocket_recitation(websocket: WebSocket):
    import traceback
    await websocket.accept()
    print("[Backend] Client Connected.")
    
    MAX_BUFFER_SAMPLES = 16000 * 10
    audio_buffer = np.zeros(0, dtype=np.float32)
    last_process_time = time.time()
    
    # State variables
    current_surah = 0
    current_ayah = 0
    failed_local_matches = 0
    taraweeh_mode = False
    pending_rewind_ayah = 0
    pending_rewind_count = 0
    pending_loop_key = None
    pending_loop_count = 0
    current_progress_coverage = 0.0
    current_word_index = 0
    pending_forward_surah = 0
    pending_forward_ayah = 0
    pending_forward_count = 0
    assisted_surah = 0
    assisted_ayah = 0
    post_recovery_lock = False
    pending_mistake_key = None
    pending_mistake_count = 0
    pending_tail_mismatch_key = None
    pending_tail_mismatch_count = 0
    mistake_cooldown_until = 0.0
    completion_grace_until = 0.0
    last_progress_time = time.time()
    has_committed_match = False
    last_ayah_change_time = time.time()
    
    try:
        while True:
            try:
                # Priority 1: Use receive() to handle both binary and text frames safely without raising KeyError
                message = await websocket.receive()
                
                if "text" in message:
                    text_data = message["text"]
                    try:
                        cmd = json.loads(text_data)
                        if cmd.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                        elif cmd.get("type") == "reset":
                            current_surah = 0
                            current_ayah = 0
                            failed_local_matches = 0
                            taraweeh_mode = False
                            pending_rewind_ayah = 0
                            pending_rewind_count = 0
                            pending_loop_key = None
                            pending_loop_count = 0
                            current_progress_coverage = 0.0
                            current_word_index = 0
                            pending_forward_surah = 0
                            pending_forward_ayah = 0
                            pending_forward_count = 0
                            audio_buffer = np.zeros(0, dtype=np.float32)
                            assisted_surah = 0
                            assisted_ayah = 0
                            post_recovery_lock = False
                            pending_mistake_key = None
                            pending_mistake_count = 0
                            pending_tail_mismatch_key = None
                            pending_tail_mismatch_count = 0
                            mistake_cooldown_until = 0.0
                            completion_grace_until = 0.0
                            last_progress_time = time.time()
                            has_committed_match = False
                            last_ayah_change_time = time.time()
                            print("[Backend] Context reset by client.")
                        elif cmd.get("type") == "start_taraweeh":
                            current_surah, current_ayah = normalize_ayah_position(
                                cmd.get("surah", 1), cmd.get("ayah", 1)
                            )
                            taraweeh_mode = True
                            failed_local_matches = 0
                            pending_rewind_ayah = 0
                            pending_rewind_count = 0
                            pending_loop_key = None
                            pending_loop_count = 0
                            current_progress_coverage = 0.0
                            current_word_index = 0
                            pending_forward_surah = 0
                            pending_forward_ayah = 0
                            pending_forward_count = 0
                            audio_buffer = np.zeros(0, dtype=np.float32)
                            assisted_surah = 0
                            assisted_ayah = 0
                            post_recovery_lock = False
                            pending_mistake_key = None
                            pending_mistake_count = 0
                            pending_tail_mismatch_key = None
                            pending_tail_mismatch_count = 0
                            mistake_cooldown_until = 0.0
                            completion_grace_until = 0.0
                            last_progress_time = time.time()
                            has_committed_match = False
                            last_ayah_change_time = time.time()
                            print(f"[Backend] Taraweeh mode started: Surah {current_surah}, Ayah {current_ayah}")
                            verse = next((v for v in _verses if v["surah"] == current_surah and v["ayah"] == current_ayah), None)
                            if verse:
                                await websocket.send_json({
                                    "type": "assisted_verse_text",
                                    "ayah_text": verse.get("text_uthmani", ""),
                                    "surah": current_surah,
                                    "ayah": current_ayah,
                                })
                        elif cmd.get("type") == "stop_taraweeh":
                            taraweeh_mode = False
                            current_progress_coverage = 0.0
                            current_word_index = 0
                            pending_loop_key = None
                            pending_loop_count = 0
                            pending_forward_surah = 0
                            pending_forward_ayah = 0
                            pending_forward_count = 0
                            assisted_surah = 0
                            assisted_ayah = 0
                            post_recovery_lock = False
                            pending_mistake_key = None
                            pending_mistake_count = 0
                            pending_tail_mismatch_key = None
                            pending_tail_mismatch_count = 0
                            completion_grace_until = 0.0
                            last_progress_time = time.time()
                            has_committed_match = False
                            last_ayah_change_time = time.time()
                            print("[Backend] Taraweeh mode stopped.")
                        elif cmd.get("type") == "assisted_prompt":
                            requested_surah = cmd.get("surah", current_surah)
                            requested_ayah = cmd.get("ayah", 0)
                            prompt_count = cmd.get("count", 1)
                            prompt_word_index = cmd.get("word_index", 0)
                            assisted_surah = 0
                            assisted_ayah = 0
                            pending_forward_surah = 0
                            pending_forward_ayah = 0
                            pending_forward_count = 0
                            pending_mistake_key = None
                            pending_mistake_count = 0
                            pending_tail_mismatch_key = None
                            pending_tail_mismatch_count = 0
                            if taraweeh_mode and current_surah > 0:
                                requested_surah_int = int(requested_surah or current_surah)
                                requested_ayah_int = int(requested_ayah or 0)
                                last_taraweeh_ayah = last_ayah_in_surah(current_surah)
                                if (
                                    requested_surah_int != current_surah
                                    or requested_ayah_int > last_taraweeh_ayah
                                ):
                                    print(
                                        "[Backend] Suppressed cross-surah Taraweeh prompt "
                                        f"{requested_surah_int}:{requested_ayah_int}; "
                                        f"holding Surah {current_surah}."
                                    )
                                    await websocket.send_json({
                                        "type": "status",
                                        "message": f"Surah {current_surah} complete; no next-surah Luqmah in Taraweeh mode.",
                                        "tracking_mode": "TARAWEEH_SURAH_COMPLETE",
                                        "search_window": f"Surah {current_surah} only",
                                    })
                                    continue
                            prompt_verses = build_luqmah_verses(
                                requested_surah,
                                requested_ayah,
                                prompt_count,
                                prompt_word_index,
                            )
                            if prompt_verses:
                                first_prompt = prompt_verses[0]
                                assisted_surah = first_prompt["surah"]
                                assisted_ayah = first_prompt["ayah"]
                                print(f"[Backend] Assisted prompt pinned to {assisted_surah}:{assisted_ayah}")
                                await websocket.send_json({
                                    "type": "assisted_verse_text",
                                    "ayah_text": "\n".join(v.get("prompt_text") or v["ayah_text"] for v in prompt_verses),
                                    "surah": first_prompt["surah"],
                                    "ayah": first_prompt["ayah"],
                                    "prompt_verses": prompt_verses,
                                })
                        elif cmd.get("type") == "clear_assisted_prompt":
                            assisted_surah = 0
                            assisted_ayah = 0
                            pending_forward_surah = 0
                            pending_forward_ayah = 0
                            pending_forward_count = 0
                            pending_mistake_key = None
                            pending_mistake_count = 0
                            pending_tail_mismatch_key = None
                            pending_tail_mismatch_count = 0
                            completion_grace_until = 0.0
                            print("[Backend] Assisted prompt cleared")
                        elif cmd.get("type") == "discard_audio":
                            audio_buffer = np.zeros(0, dtype=np.float32)
                            last_process_time = time.time()
                            print("[Backend] Pending audio discarded.")
                        else:
                            print(f"[Backend] Unknown text command: {text_data}")
                    except Exception as e:
                        print(f"[Backend] Error parsing text message: {e}")
                    continue
                    
                if "bytes" not in message:
                    continue
                    
                data = message["bytes"]
                # print(f"[Backend] Audio Packet Received. Size: {len(data)} bytes")
                
                pcm16 = np.frombuffer(data, dtype=np.int16)
                if len(pcm16) == 0:
                    continue
                    
                chunk_float = pcm16.astype(np.float32) / 32768.0
                audio_buffer = np.append(audio_buffer, chunk_float)
                
                if len(audio_buffer) > MAX_BUFFER_SAMPLES:
                    audio_buffer = audio_buffer[-MAX_BUFFER_SAMPLES:]
                
                current_time = time.time()
                if current_time - last_process_time >= 1.5 and len(audio_buffer) >= 16000 * 1.5:
                    last_process_time = current_time
                    
                    # print("[Backend] Inference Started.")
                    loop = asyncio.get_event_loop()
                    inference_audio = audio_buffer[-(16000 * 8):].copy()
                    result = await loop.run_in_executor(None, predict_audio, inference_audio, current_surah, current_ayah, failed_local_matches, taraweeh_mode, assisted_surah, assisted_ayah, post_recovery_lock)
                    
                    if result.get("surah", 0) > 0:
                        failed_local_matches = 0 # Reset fallback counter on success
                        pending_mistake_key = None
                        pending_mistake_count = 0
                        matched_surah = result["surah"]
                        next_coverage = result.get(
                            "progress_coverage",
                            result.get(
                                "word_coverage",
                                result.get("section_coverage", 0.0),
                            ),
                        )
                        matched_word_index = result.get("word_index", 0)
                        matched_ayah = (
                            result["ayah_end"]
                            if result.get("ayah_end") and next_coverage >= 0.90
                            else result["ayah"]
                        )
                        
                        # Handle Rewind Logic
                        if taraweeh_mode and matched_surah == current_surah and matched_ayah < current_ayah:
                            rewind_distance = current_ayah - matched_ayah
                            recent_transition_overlap = (
                                rewind_distance == 1
                                and time.time() - last_ayah_change_time < _REWIND_TRANSITION_GRACE_SECONDS
                            )
                            weak_near_rewind = (
                                rewind_distance == 1
                                and result.get("score", 0.0) < 0.88
                            )
                            if recent_transition_overlap or weak_near_rewind:
                                pending_rewind_ayah = 0
                                pending_rewind_count = 0
                                await websocket.send_json({
                                    "type": "status",
                                    "message": f"Previous ayah overlap near boundary; holding Ayah {current_ayah}.",
                                    "tracking_mode": "REWIND_OVERLAP_IGNORED",
                                    "search_window": result.get("search_window", ""),
                                    "fallback_count": failed_local_matches,
                                    "metrics": result.get("metrics", {})
                                })
                                continue

                            if rewind_distance >= 2 and assisted_ayah == 0:
                                loop_key = (matched_surah, matched_ayah)
                                if pending_loop_key == loop_key:
                                    pending_loop_count += 1
                                else:
                                    pending_loop_key = loop_key
                                    pending_loop_count = 1

                                expected_surah, expected_ayah = _expected_prompt_position(
                                    current_surah,
                                    current_ayah,
                                    current_progress_coverage,
                                    taraweeh_mode,
                                )
                                if pending_loop_count >= 2 and time.time() >= mistake_cooldown_until:
                                    await websocket.send_json({
                                        "type": "mistake_detected",
                                        "expected_surah": expected_surah,
                                        "expected_ayah": expected_ayah,
                                        "detected_surah": matched_surah,
                                        "detected_ayah": matched_ayah,
                                        "score": result.get("score", 0.0),
                                        "reason": "repetition_loop",
                                    })
                                    mistake_cooldown_until = time.time() + 3.0
                                    pending_loop_key = None
                                    pending_loop_count = 0
                                    audio_buffer = np.zeros(0, dtype=np.float32)
                                else:
                                    await websocket.send_json({
                                        "type": "status",
                                        "message": f"Possible repeated earlier ayah {matched_ayah}; holding current position.",
                                        "tracking_mode": "REPETITION_LOOP_CHECK",
                                        "search_window": result.get("search_window", ""),
                                        "fallback_count": failed_local_matches,
                                        "metrics": result.get("metrics", {}),
                                    })
                                continue

                            if pending_rewind_ayah == matched_ayah:
                                pending_rewind_count += 1
                            else:
                                pending_rewind_ayah = matched_ayah
                                pending_rewind_count = 1
                                
                            required_rewind_confirmations = 3 if rewind_distance == 1 else 2
                            if pending_rewind_count < required_rewind_confirmations:
                                # Not confirmed yet
                                await websocket.send_json({
                                    "type": "status",
                                    "message": (
                                        f"Rewind detected to Ayah {matched_ayah}. "
                                        f"Confirming {pending_rewind_count}/{required_rewind_confirmations}..."
                                    ),
                                    "tracking_mode": "REWIND DETECTED (Confirming...)",
                                    "search_window": result.get("search_window", ""),
                                    "fallback_count": failed_local_matches,
                                    "metrics": result.get("metrics", {})
                                })
                                continue # Skip applying the match
                            else:
                                print(f"[Backend] Confirmed Rewind to Ayah {matched_ayah}")
                                pending_rewind_count = 0
                                pending_rewind_ayah = 0
                        else:
                            pending_rewind_count = 0
                            pending_rewind_ayah = 0
                            pending_loop_key = None
                            pending_loop_count = 0

                        forward_section_distance = _forward_section_distance(
                            current_surah,
                            current_ayah,
                            current_progress_coverage,
                            matched_ayah,
                            result.get("section_index", 0),
                        )
                        skipped_ayahs = max(0, matched_ayah - current_ayah - 1)
                        required_forward_confirmations = _required_forward_jump_confirmations(
                            skipped_ayahs,
                            forward_section_distance,
                            current_progress_coverage,
                            result.get("score", 0.0),
                        )

                        if (
                            taraweeh_mode
                            and current_surah > 0
                            and matched_surah == current_surah
                            and matched_ayah > current_ayah
                            and required_forward_confirmations > 0
                            and assisted_ayah == 0
                        ):
                            if pending_forward_surah == matched_surah and pending_forward_ayah == matched_ayah:
                                pending_forward_count += 1
                            else:
                                pending_forward_surah = matched_surah
                                pending_forward_ayah = matched_ayah
                                pending_forward_count = 1

                            if pending_forward_count < required_forward_confirmations:
                                skipped_message = (
                                    f" skipping {skipped_ayahs} ayah(s);"
                                    if skipped_ayahs > 0
                                    else ""
                                )
                                await websocket.send_json({
                                    "type": "status",
                                    "message": (
                                        f"Possible forward jump to Ayah {matched_ayah} "
                                        f"({forward_section_distance} sections ahead;{skipped_message} "
                                        f"confirmation {pending_forward_count}/{required_forward_confirmations})."
                                    ),
                                    "tracking_mode": "FORWARD_JUMP_CHECK",
                                    "search_window": result.get("search_window", ""),
                                    "fallback_count": failed_local_matches,
                                    "metrics": result.get("metrics", {})
                                })
                                continue
                            pending_forward_surah = 0
                            pending_forward_ayah = 0
                            pending_forward_count = 0
                        else:
                            pending_forward_surah = 0
                            pending_forward_ayah = 0
                            pending_forward_count = 0
                            
                        previous_surah = current_surah
                        previous_ayah = current_ayah
                        previous_coverage = current_progress_coverage
                        previous_word_index = current_word_index
                        current_surah = matched_surah
                        current_ayah = matched_ayah
                        has_committed_match = True
                        same_position = (
                            matched_surah == previous_surah
                            and matched_ayah == previous_ayah
                        )
                        if not same_position:
                            last_ayah_change_time = time.time()
                        if same_position:
                            current_word_index = max(current_word_index, matched_word_index)
                            current_progress_coverage = max(
                                current_progress_coverage,
                                next_coverage,
                            )
                        else:
                            current_word_index = matched_word_index
                            current_progress_coverage = next_coverage

                        if (
                            not same_position
                            or current_word_index > previous_word_index
                            or current_progress_coverage > previous_coverage + _PROGRESS_STALE_DELTA
                        ):
                            last_progress_time = time.time()

                        total_words = int(result.get("total_words", 0) or 0)
                        total_sections = int(result.get("total_sections", 0) or 0)
                        section_index = int(result.get("section_index", 0) or 0)
                        near_completion = (
                            current_progress_coverage >= 0.72
                            or (total_words > 0 and current_word_index >= max(1, total_words - 2))
                            or (total_sections > 0 and section_index >= max(1, total_sections - 1))
                        )
                        if taraweeh_mode and near_completion:
                            completion_grace_until = max(
                                completion_grace_until,
                                time.time() + _transition_grace_seconds(
                                    current_surah,
                                    current_ayah,
                                    current_progress_coverage,
                                ),
                            )
                        
                        if assisted_ayah > 0 and current_surah == assisted_surah and current_ayah == assisted_ayah:
                            # Auto-clear assisted_ayah once matched to enforce Post-Prompt Lock
                            assisted_surah = 0
                            assisted_ayah = 0
                            post_recovery_lock = True
                            print(f"[Backend] Assisted match successful. Locking to Ayah {current_ayah}")
                        elif post_recovery_lock:
                            post_recovery_lock = False
                            print(f"[Backend] Post-recovery lock cleared. Returning to normal tracking.")
                        
                        current_text = get_ayah_text(current_surah, current_ayah)
                        prev_text = get_ayah_text(current_surah, current_ayah - 1)
                        next_text = get_ayah_text(current_surah, current_ayah + 1)
                        if not next_text and current_surah < 114 and not taraweeh_mode:
                            next_text = get_ayah_text(current_surah + 1, 1)
                        surah_complete = taraweeh_mode and _at_surah_end(
                            current_surah,
                            current_ayah,
                        )
                            
                        await websocket.send_json({
                            "type": "word_progress",
                            "surah": result["surah"],
                            "ayah": result["ayah"],
                            "word_index": result.get("word_index", 0),
                            "total_words": result.get("total_words", 0),
                            "confidence": result.get("word_confidence", 0.0),
                            "word_coverage": result.get("word_coverage", 0.0),
                            "section_index": result.get("section_index", 0),
                            "total_sections": result.get("total_sections", 0),
                            "section_coverage": result.get("section_coverage", 0.0),
                            "progress_coverage": current_progress_coverage,
                            "section_start_word": result.get("section_start_word", 1),
                            "section_end_word": result.get("section_end_word", 0),
                            "is_mutashabihat": result.get("is_mutashabihat", False),
                            "tracking_mode": result.get("tracking_mode", ""),
                            "search_window": result.get("search_window", ""),
                            "fallback_count": failed_local_matches,
                            "current_ayah_text": current_text,
                            "next_ayah_text": next_text,
                            "prev_ayah_text": prev_text,
                            "surah_complete": surah_complete
                        })

                        word_confidence = result.get("word_confidence", 1.0)
                        if _should_emit_word_correction(result, word_confidence):
                            await websocket.send_json({
                                "type": "word_correction",
                                "surah": result["surah"],
                                "ayah": result["ayah"],
                                "corrections": [{
                                    "word_index": result["word_index"],
                                    "expected": result["word_expected"],
                                    "got": result["word_got"],
                                    "confidence": word_confidence,
                                    "error_type": "low_confidence",
                                }],
                            })

                        tail_mismatch = _same_ayah_tail_mismatch(result)
                        if (
                            taraweeh_mode
                            and assisted_ayah == 0
                            and matched_surah == current_surah
                            and matched_ayah == current_ayah
                            and time.time() >= mistake_cooldown_until
                            and tail_mismatch
                        ):
                            tail_key = (
                                current_surah,
                                current_ayah,
                                tail_mismatch["word_index"],
                                tail_mismatch["expected"],
                            )
                            if pending_tail_mismatch_key == tail_key:
                                pending_tail_mismatch_count += 1
                            else:
                                pending_tail_mismatch_key = tail_key
                                pending_tail_mismatch_count = 1

                            if pending_tail_mismatch_count >= 2:
                                print(
                                    "[Backend] Same-ayah tail mismatch "
                                    f"{current_surah}:{current_ayah} word "
                                    f"{tail_mismatch['word_index']}: expected "
                                    f"{tail_mismatch['expected']}, heard "
                                    f"{tail_mismatch['got']} "
                                    f"(strict={tail_mismatch['strict_ratio']}, "
                                    f"skeleton={tail_mismatch['skeleton_ratio']})"
                                )
                                await websocket.send_json({
                                    "type": "mistake_detected",
                                    "expected_surah": current_surah,
                                    "expected_ayah": current_ayah,
                                    "detected_surah": current_surah,
                                    "detected_ayah": current_ayah,
                                    "score": max(
                                        0.01,
                                        1.0 - tail_mismatch["skeleton_ratio"],
                                    ),
                                    "reason": "same_ayah_tail_mismatch",
                                    "word_index": tail_mismatch["word_index"],
                                    "expected_word": tail_mismatch["expected"],
                                    "heard_word": tail_mismatch["got"],
                                })
                                pending_tail_mismatch_key = None
                                pending_tail_mismatch_count = 0
                                mistake_cooldown_until = time.time() + 3.0
                                audio_buffer = np.zeros(0, dtype=np.float32)
                                continue

                            await websocket.send_json({
                                "type": "status",
                                "message": (
                                    "Possible final-word mismatch; confirming..."
                                ),
                                "tracking_mode": "TAIL_WORD_CHECK",
                                "search_window": result.get("search_window", ""),
                                "fallback_count": failed_local_matches,
                                "metrics": result.get("metrics", {}),
                            })
                            continue
                        else:
                            pending_tail_mismatch_key = None
                            pending_tail_mismatch_count = 0
                        
                        await websocket.send_json({
                            "type": "verse_match",
                            "surah": result["surah"],
                            "ayah": result["ayah"],
                            "ayah_end": result["ayah_end"],
                            "surah_name_en": result["surah_name_en"],
                            "surah_name": result["surah_name"],
                            "confidence": result["score"],
                            "transcript": result["transcript"],
                            "word_index": result.get("word_index", 0),
                            "total_words": result.get("total_words", 0),
                            "word_confidence": result.get("word_confidence", 0.0),
                            "word_coverage": result.get("word_coverage", 0.0),
                            "section_index": result.get("section_index", 0),
                            "total_sections": result.get("total_sections", 0),
                            "section_coverage": result.get("section_coverage", 0.0),
                            "progress_coverage": current_progress_coverage,
                            "section_start_word": result.get("section_start_word", 1),
                            "section_end_word": result.get("section_end_word", 0),
                            "tracking_mode": result.get("tracking_mode", ""),
                            "search_window": result.get("search_window", ""),
                            "fallback_count": failed_local_matches,
                            "metrics": result.get("metrics", {}),
                            "current_ayah_text": current_text,
                            "next_ayah_text": next_text,
                            "prev_ayah_text": prev_text,
                            "surah_complete": surah_complete
                        })
                        # Keep a short overlap so the next ayah can be acquired
                        # without repeatedly matching old completed audio.
                        if len(audio_buffer) > 16000:
                            audio_buffer = audio_buffer[-16000:]
                    else:
                        pending_tail_mismatch_key = None
                        pending_tail_mismatch_count = 0
                        if result.get("mistake_candidate"):
                            expected_surah, expected_ayah = _expected_prompt_position(
                                current_surah,
                                current_ayah,
                                current_progress_coverage,
                                taraweeh_mode,
                            )
                            transcript_text = result.get("transcript", "")
                            transcript_chars = _compact_char_count(transcript_text)
                            detected_verse = (
                                get_verse(result["mistake_surah"], result["mistake_ayah"])
                                if result["mistake_surah"] > 0 and result["mistake_ayah"] > 0
                                else None
                            )
                            cross_surah_candidate = (
                                result["mistake_surah"] > 0
                                and result["mistake_surah"] != current_surah
                            )
                            recent_ayah_transition = (
                                last_ayah_change_time > 0
                                and time.time() - last_ayah_change_time < 3.5
                            )
                            if cross_surah_candidate and (
                                _is_ambiguous_short_phrase(transcript_text, detected_verse)
                                or (recent_ayah_transition and transcript_chars < 32)
                            ):
                                pending_mistake_key = None
                                pending_mistake_count = 0
                                await websocket.send_json({
                                    "type": "status",
                                    "message": f"Keep reciting | Transcript: {transcript_text[:25]}...",
                                    "tracking_mode": result.get("tracking_mode", ""),
                                    "search_window": result.get("search_window", ""),
                                    "fallback_count": failed_local_matches,
                                    "metrics": result.get("metrics", {}),
                                })
                                continue
                            mistake_key = (
                                expected_surah,
                                expected_ayah,
                                result["mistake_surah"],
                                result["mistake_ayah"],
                            )
                            if time.time() >= mistake_cooldown_until and assisted_ayah == 0:
                                required_confirmations = 1 if result.get("mistake_immediate") else 2
                                expected_is_ambiguous = _is_ambiguous_signature_verse(
                                    expected_surah,
                                    expected_ayah,
                                )
                                detected_is_ambiguous = (
                                    result["mistake_surah"] > 0
                                    and result["mistake_ayah"] > 0
                                    and _is_ambiguous_signature_verse(
                                        result["mistake_surah"],
                                        result["mistake_ayah"],
                                    )
                                )
                                strong_immediate_mistake = (
                                    bool(result.get("mistake_immediate"))
                                    and result.get("mistake_score", 0.0) >= 0.84
                                    and transcript_chars >= 20
                                    and not expected_is_ambiguous
                                    and not detected_is_ambiguous
                                )
                                if expected_is_ambiguous or detected_is_ambiguous:
                                    # Repeated openings like "حم" or "الم" are
                                    # globally ambiguous, so never interrupt
                                    # Taraweeh from a single stray hit.
                                    required_confirmations = max(required_confirmations, 3)
                                elif (
                                    current_progress_coverage >= 0.35
                                    and not strong_immediate_mistake
                                ):
                                    required_confirmations = max(required_confirmations, 2)

                                if (
                                    result["mistake_surah"] > 0
                                    and result["mistake_surah"] != current_surah
                                    and current_progress_coverage >= 0.25
                                    and not strong_immediate_mistake
                                ):
                                    required_confirmations = max(required_confirmations, 2)

                                if pending_mistake_key == mistake_key:
                                    pending_mistake_count += 1
                                else:
                                    pending_mistake_key = mistake_key
                                    pending_mistake_count = 1

                                if pending_mistake_count >= required_confirmations:
                                    print(f"[Backend] Sending mistake correction for {result['mistake_surah']}:{result['mistake_ayah']}")
                                    await websocket.send_json({
                                        "type": "mistake_detected",
                                        "expected_surah": expected_surah,
                                        "expected_ayah": expected_ayah,
                                        "detected_surah": result["mistake_surah"],
                                        "detected_ayah": result["mistake_ayah"],
                                        "score": result.get("mistake_score", 0.0),
                                    })
                                    pending_mistake_key = None
                                    pending_mistake_count = 0
                                    mistake_cooldown_until = time.time() + 3.0
                                    audio_buffer = np.zeros(0, dtype=np.float32)
                                else:
                                    await websocket.send_json({
                                        "type": "status",
                                        "message": "Possible mistake detected; confirming...",
                                        "tracking_mode": result.get("tracking_mode", ""),
                                        "search_window": result.get("search_window", ""),
                                        "fallback_count": failed_local_matches,
                                        "metrics": result.get("metrics", {}),
                                    })
                                    continue

                        if result.get("speech_detected", True) and current_surah > 0:
                            failed_local_matches += 1
                            required_failures, stall_seconds = _stall_detection_thresholds(
                                current_surah,
                                current_ayah,
                                current_progress_coverage,
                                taraweeh_mode,
                            )
                            if (
                                taraweeh_mode
                                and failed_local_matches >= required_failures
                                and time.time() - last_progress_time >= stall_seconds
                                and assisted_ayah == 0
                                and has_committed_match
                                and time.time() >= mistake_cooldown_until
                                and time.time() >= completion_grace_until
                            ):
                                if (
                                    taraweeh_mode
                                    and _at_surah_end(current_surah, current_ayah)
                                ):
                                    failed_local_matches = 0
                                    pending_mistake_key = None
                                    pending_mistake_count = 0
                                    await websocket.send_json({
                                        "type": "status",
                                        "message": f"Surah {current_surah} complete; holding Taraweeh lock.",
                                        "tracking_mode": "TARAWEEH_SURAH_COMPLETE",
                                        "search_window": f"Surah {current_surah} only",
                                        "fallback_count": failed_local_matches,
                                        "metrics": result.get("metrics", {}),
                                    })
                                    continue
                                expected_surah, expected_ayah = _expected_prompt_position(
                                    current_surah,
                                    current_ayah,
                                    current_progress_coverage,
                                    taraweeh_mode,
                                )
                                await websocket.send_json({
                                    "type": "status",
                                    "message": (
                                        f"Stall detected near {expected_surah}:{expected_ayah}; "
                                        "waiting for local pause prompt."
                                    ),
                                    "tracking_mode": "TARAWEEH_STALL",
                                    "search_window": result.get("search_window", ""),
                                    "fallback_count": failed_local_matches,
                                    "metrics": result.get("metrics", {}),
                                    "expected_surah": expected_surah,
                                    "expected_ayah": expected_ayah,
                                    "reason": "no_context_progress",
                                })
                                failed_local_matches = 0
                                pending_mistake_key = None
                                pending_mistake_count = 0
                                audio_buffer = np.zeros(0, dtype=np.float32)
                                continue
                            if failed_local_matches >= 4 and not taraweeh_mode:
                                print(f"[Backend] 4 consecutive local search failures. Clearing context lock.")
                                current_surah = 0
                                current_ayah = 0
                                current_word_index = 0
                                current_progress_coverage = 0.0
                                failed_local_matches = 0
                                
                        await websocket.send_json({
                            "type": "status",
                            "message": f"{result.get('error', 'Listening...')} | Transcript: {result.get('transcript', '')[:25]}...",
                            "tracking_mode": result.get("tracking_mode", ""),
                            "search_window": result.get("search_window", ""),
                            "fallback_count": failed_local_matches,
                            "metrics": result.get("metrics", {})
                        })
                        if len(audio_buffer) > 16000 * 6:
                            audio_buffer = audio_buffer[-(16000 * 6):]
            except Exception as inner_e:
                if isinstance(inner_e, WebSocketDisconnect):
                    print("[Backend] Client Disconnected gracefully.")
                    break
                if isinstance(inner_e, RuntimeError) and "disconnect" in str(inner_e):
                    print("[Backend] Client Disconnected (RuntimeError).")
                    break
                print("[Backend] Packet Processing Error:")
                print(traceback.format_exc())
                # Do not break the loop on a single packet failure
                continue
                
    except WebSocketDisconnect:
        print("[Backend] Client Disconnected gracefully.")
    except Exception as e:
        print("[Backend] Fatal WebSocket loop error:")
        print(traceback.format_exc())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
