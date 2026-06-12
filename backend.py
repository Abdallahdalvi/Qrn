import asyncio
import json
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

    for v in _verses:
        joined = v.get("phonemes_joined", "")
        v["_phonemes_joined_ns"] = joined.replace(" ", "")

        # Strip bismillah prefix for ayah 1 (except surahs 1 and 9)
        no_bsm = None
        if v["ayah"] == 1 and v["surah"] != 1 and v["surah"] != 9 and joined.startswith(_BSM_PHONEMES_JOINED):
            no_bsm = joined[len(_BSM_PHONEMES_JOINED):].strip() or None
        v["_phonemes_joined_no_bsm"] = no_bsm
        v["_phonemes_joined_no_bsm_ns"] = no_bsm.replace(" ", "") if no_bsm else None

        _by_surah.setdefault(v["surah"], []).append(v)

    # Sort each surah's verses by ayah
    for surah_num, verses in _by_surah.items():
        verses.sort(key=lambda x: x["ayah"])

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

def _short_query_boost(no_space_text: str, verse: dict, use_no_bsm: bool = False) -> float:
    if use_no_bsm:
        candidate = verse.get("_phonemes_joined_no_bsm_ns", "") or verse.get("_phonemes_joined_ns", "")
    else:
        candidate = verse.get("_phonemes_joined_ns", "")
    if not candidate:
        return 0.0

    prefix_window = min(len(candidate), len(no_space_text) + 6)
    prefix = ratio(no_space_text, candidate[:prefix_window])

    if use_no_bsm:
        joined = verse.get("_phonemes_joined_no_bsm", "") or ""
    else:
        joined = verse.get("phonemes_joined", "")
    first_word = joined.split(" ")[0] if joined else ""
    first_word_score = ratio(no_space_text, first_word) if first_word else 0.0

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
        raw = ratio(phoneme_text, ref)

        if len(no_space_text) <= 10:
            raw = max(raw, _short_query_boost(no_space_text, verse))

        no_bsm = verse.get("_phonemes_joined_no_bsm")
        if no_bsm:
            raw = max(raw, ratio(phoneme_text, no_bsm))
            if len(no_space_text) <= 10:
                raw = max(raw, _short_query_boost(no_space_text, verse, use_no_bsm=True))

        scored.append([verse, raw, raw])

    scored.sort(key=lambda x: x[2], reverse=True)

    # Top Surah candidates
    pass2_surahs = []
    for entry in scored:
        s = entry[0]["surah"]
        if s not in pass2_surahs:
            pass2_surahs.append(s)
        if len(pass2_surahs) >= 15:
            break

    # Pass 1.5: fragment boost
    if len(no_space_text) >= 8:
        resorted = False
        for i, (verse, raw, _) in enumerate(scored):
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

    # Pass 2: multi-verse spans (up to 3 verses)
    span_results = []
    for surah_num in pass2_surahs:
        verses = _by_surah.get(surah_num, [])
        for i in range(len(verses)):
            for span in range(2, 4):
                if i + span > len(verses):
                    break
                chunk = verses[i:i + span]
                first_phonemes = chunk[0].get("_phonemes_joined_no_bsm") or chunk[0].get("phonemes_joined", "")
                span_phonemes = first_phonemes + " " + " ".join(
                    v.get("phonemes_joined", "") for v in chunk[1:]
                )
                raw = ratio(phoneme_text, span_phonemes)
                span_results.append({
                    "surah": surah_num,
                    "ayah": chunk[0]["ayah"],
                    "ayah_end": chunk[-1]["ayah"],
                    "score": round(raw, 4),
                    "surah_name": chunk[0]["surah_name"],
                    "surah_name_en": chunk[0]["surah_name_en"],
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
        return {"word": 0, "total": len(v_words), "confidence": 0.0}
        
    best_score = -1.0
    best_idx = 0
    best_word_conf = 0.0
    
    for i in range(len(v_words)):
        # Calculate distinct word confidence for the last transcribed word
        word_conf = ratio(t_words[-1], v_words[i])
        score = word_conf
        
        # Add context from previous word if available
        if len(t_words) > 1 and i > 0:
            score += ratio(t_words[-2], v_words[i-1]) * 0.5
            
        if score > best_score:
            best_score = score
            best_idx = i
            best_word_conf = word_conf
            
    return {"word": best_idx + 1, "total": len(v_words), "confidence": round(best_word_conf, 4)}

def get_search_context(current_surah: int, current_ayah: int, failed_matches: int, taraweeh_mode: bool, assisted_ayah: int = 0, post_recovery_lock: bool = False) -> tuple[list, str, str]:
    if current_surah == 0 or current_ayah == 0:
        return _verses, "GLOBAL_SEARCH", "Entire Quran"
        
    surah_verses = _by_surah.get(current_surah, [])
    if not surah_verses:
        return _verses, "GLOBAL_SEARCH", "Entire Quran"
        
    last_ayah_in_surah = surah_verses[-1]["ayah"]
    
    if assisted_ayah > 0:
        start_ayah = max(1, assisted_ayah - 2)
        end_ayah = assisted_ayah + 2
        mode = "ASSISTED_TRACKING"
    elif post_recovery_lock:
        start_ayah = max(1, current_ayah - 1)
        end_ayah = current_ayah + 2
        mode = "RECOVERY_CONFIRMED"
    elif failed_matches == 0:
        start_ayah = current_ayah
        end_ayah = current_ayah + 4
        mode = "NORMAL"
    elif failed_matches == 1:
        start_ayah = max(1, current_ayah - 8) if taraweeh_mode else max(1, current_ayah - 2)
        end_ayah = current_ayah + 6
        mode = "EXPANDED_WINDOW_1"
    elif failed_matches == 2:
        start_ayah = max(1, current_ayah - 12) if taraweeh_mode else max(1, current_ayah - 4)
        end_ayah = current_ayah + 8
        mode = "EXPANDED_WINDOW_2"
    elif failed_matches == 3:
        if (current_surah, current_ayah) in MUTASHABIHAT_DB and taraweeh_mode:
            # STRICT LOCKDOWN: Do not allow full surah search if we are in a mutashabihat verse.
            # This prevents jumping to the similar verse in the other surah.
            start_ayah = max(1, current_ayah - 12)
            end_ayah = current_ayah + 8
            mode = "MUTASHABIHAT_LOCKED"
        else:
            start_ayah = 1
            end_ayah = last_ayah_in_surah
            mode = "SURAH_SEARCH"
    else:
        if taraweeh_mode:
            if current_ayah >= last_ayah_in_surah - 3 and current_surah < 114:
                next_surah_verses = _by_surah.get(current_surah + 1, [])
                return surah_verses + next_surah_verses, "SURAH_TRANSITION", f"Surah {current_surah} & {current_surah + 1}"
            else:
                return surah_verses, "SURAH_SEARCH", f"Surah {current_surah} Only"
        else:
            return _verses, "GLOBAL_SEARCH", "Entire Quran"
            
    local_verses = []
    if end_ayah > last_ayah_in_surah and current_surah < 114:
        for v in surah_verses:
            if v["ayah"] >= start_ayah:
                local_verses.append(v)
        next_surah_verses = _by_surah.get(current_surah + 1, [])
        overflow = end_ayah - last_ayah_in_surah
        for v in next_surah_verses:
            if v["ayah"] <= overflow:
                local_verses.append(v)
        window_str = f"Surah {current_surah}:{max(1, start_ayah)} -> Surah {current_surah + 1}:{overflow}"
    else:
        for v in surah_verses:
            if start_ayah <= v["ayah"] <= end_ayah:
                local_verses.append(v)
        window_str = f"Surah {current_surah}: {max(1, start_ayah)}-{min(end_ayah, last_ayah_in_surah)}"
        
    return local_verses, mode, window_str

def predict_audio(audio: np.ndarray, current_surah: int = 0, current_ayah: int = 0, failed_matches: int = 0, taraweeh_mode: bool = False, assisted_ayah: int = 0, post_recovery_lock: bool = False) -> dict:
    metrics = {"audio_len": round(len(audio) / 16000, 3)}
    t_start = time.perf_counter()
    
    if len(audio) < 16000 * 2.0:  # Priority 4: Min 2 seconds
        return {"surah": 0, "ayah": 0, "score": 0.0, "transcript": "", "error": "Insufficient audio duration (<2s)", "metrics": metrics}

    t0 = time.perf_counter()
    logprobs = _compute_logprobs(audio)
    metrics["inference_time"] = round(time.perf_counter() - t0, 3)
    
    phoneme_text = _greedy_decode_phonemes(logprobs)
    print(f"[Backend] Decoded Phonemes ({len(phoneme_text)} chars): {phoneme_text}")
    
    no_space_text = phoneme_text.replace(" ", "")
    if len(no_space_text) < 10:  # Priority 4: Min 10 decoded characters
        return {"surah": 0, "ayah": 0, "score": 0.0, "transcript": phoneme_text, "error": "Insufficient transcript length (<10 chars)", "metrics": metrics}
    
    t0 = time.perf_counter()
    # Dynamic Search Context
    search_verses, tracking_mode, window_str = get_search_context(current_surah, current_ayah, failed_matches, taraweeh_mode, assisted_ayah, post_recovery_lock)
    
    # print(f"[Backend] Tracking: {tracking_mode} | Window: {window_str}")
    top_matches = _match_phoneme_text(phoneme_text, top_k=5, search_verses=search_verses)
    
    # Apply score weighting if we are in Assisted Tracking mode
    if assisted_ayah > 0:
        for match in top_matches:
            distance = abs(match["ayah"] - assisted_ayah)
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
                if match["surah"] == current_surah:
                    distance = match["ayah"] - current_ayah
                    # Penalize backward jumps or forward jumps > 5 ayahs
                    if distance < -1 or distance > 5:
                        is_jump = True
                elif match["surah"] == current_surah + 1 and tracking_mode == "SURAH_TRANSITION":
                    is_jump = False
                else:
                    is_jump = True
                    
        # Anti-jump protection
        if is_jump and match["score"] < 0.85:
            print(f"[Backend] Rejecting jump to {match['surah']}:{match['ayah']} due to low confidence ({match['score']:.2f})")
            continue
            
        if match["score"] >= 0.60:
            best_match = match
            break

    metrics["matching_time"] = round(time.perf_counter() - t0, 3)
    metrics["total_latency"] = round(time.perf_counter() - t_start, 3)
    
    if not best_match:
        return {"surah": 0, "ayah": 0, "score": 0.0, "transcript": phoneme_text, "error": "No confident match found", "tracking_mode": tracking_mode, "search_window": window_str, "metrics": metrics}
    
    best = best_match
    
    word_info = get_word_position(phoneme_text, best.get("phonemes_joined", ""))
        
    return {
        "surah": best["surah"],
        "ayah": best["ayah"],
        "ayah_end": best["ayah_end"],
        "surah_name_en": best["surah_name_en"],
        "surah_name": best["surah_name"],
        "score": best["score"],
        "transcript": phoneme_text,
        "word_index": word_info["word"],
        "total_words": word_info["total"],
        "word_confidence": word_info["confidence"],
        "is_mutashabihat": (best["surah"], best["ayah"]) in MUTASHABIHAT_DB,
        "metrics": metrics
    }

# WebSocket Streaming Endpoint
@app.websocket("/ws/recitation")
async def websocket_recitation(websocket: WebSocket):
    import traceback
    await websocket.accept()
    print("[Backend] Client Connected.")
    
    MAX_BUFFER_SAMPLES = 16000 * 20
    audio_buffer = np.zeros(0, dtype=np.float32)
    last_process_time = time.time()
    
    # State variables
    current_surah = 0
    current_ayah = 0
    failed_local_matches = 0
    taraweeh_mode = False
    pending_rewind_ayah = 0
    pending_rewind_count = 0
    assisted_ayah = 0
    post_recovery_lock = False
    
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
                            audio_buffer = np.zeros(0, dtype=np.float32)
                            post_recovery_lock = False
                            print("[Backend] Context reset by client.")
                        elif cmd.get("type") == "start_taraweeh":
                            current_surah = cmd.get("surah", 1)
                            current_ayah = cmd.get("ayah", 1)
                            taraweeh_mode = True
                            failed_local_matches = 0
                            pending_rewind_ayah = 0
                            pending_rewind_count = 0
                            audio_buffer = np.zeros(0, dtype=np.float32)
                            assisted_ayah = 0
                            post_recovery_lock = False
                            print(f"[Backend] Taraweeh mode started: Surah {current_surah}, Ayah {current_ayah}")
                        elif cmd.get("type") == "assisted_prompt":
                            assisted_ayah = cmd.get("ayah", 0)
                            print(f"[Backend] Assisted prompt activated for Ayah {assisted_ayah}")
                        elif cmd.get("type") == "clear_assisted_prompt":
                            assisted_ayah = 0
                            print("[Backend] Assisted prompt cleared")
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
                    result = await loop.run_in_executor(None, predict_audio, audio_buffer, current_surah, current_ayah, failed_local_matches, taraweeh_mode, assisted_ayah, post_recovery_lock)
                    
                    if result.get("surah", 0) > 0:
                        failed_local_matches = 0 # Reset fallback counter on success
                        matched_surah = result["surah"]
                        matched_ayah = result["ayah_end"] if result.get("ayah_end") else result["ayah"]
                        
                        # Handle Rewind Logic
                        if taraweeh_mode and matched_surah == current_surah and matched_ayah < current_ayah:
                            if pending_rewind_ayah == matched_ayah:
                                pending_rewind_count += 1
                            else:
                                pending_rewind_ayah = matched_ayah
                                pending_rewind_count = 1
                                
                            if pending_rewind_count < 2:
                                # Not confirmed yet
                                await websocket.send_json({
                                    "type": "status",
                                    "message": f"Rewind detected to Ayah {matched_ayah}. Confirming...",
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
                            
                        current_surah = matched_surah
                        current_ayah = matched_ayah
                        
                        if assisted_ayah > 0 and current_ayah == assisted_ayah:
                            # Auto-clear assisted_ayah once matched to enforce Post-Prompt Lock
                            assisted_ayah = 0
                            post_recovery_lock = True
                            print(f"[Backend] Assisted match successful. Locking to Ayah {current_ayah}")
                        elif post_recovery_lock:
                            post_recovery_lock = False
                            print(f"[Backend] Post-recovery lock cleared. Returning to normal tracking.")
                        
                        current_text = get_ayah_text(current_surah, current_ayah)
                        prev_text = get_ayah_text(current_surah, current_ayah - 1)
                        next_text = get_ayah_text(current_surah, current_ayah + 1)
                        if not next_text and current_surah < 114:
                            next_text = get_ayah_text(current_surah + 1, 1)
                            
                        await websocket.send_json({
                            "type": "word_progress",
                            "surah": result["surah"],
                            "ayah": result["ayah"],
                            "word_index": result.get("word_index", 0),
                            "total_words": result.get("total_words", 0),
                            "confidence": result.get("word_confidence", 0.0),
                            "is_mutashabihat": result.get("is_mutashabihat", False),
                            "tracking_mode": result.get("tracking_mode", ""),
                            "search_window": result.get("search_window", ""),
                            "fallback_count": failed_local_matches,
                            "current_ayah_text": current_text,
                            "next_ayah_text": next_text,
                            "prev_ayah_text": prev_text
                        })
                        
                        await websocket.send_json({
                            "type": "verse_match",
                            "surah": result["surah"],
                            "ayah": result["ayah"],
                            "ayah_end": result["ayah_end"],
                            "surah_name_en": result["surah_name_en"],
                            "surah_name": result["surah_name"],
                            "confidence": result["score"],
                            "transcript": result["transcript"],
                            "tracking_mode": result.get("tracking_mode", ""),
                            "search_window": result.get("search_window", ""),
                            "fallback_count": failed_local_matches,
                            "metrics": result.get("metrics", {}),
                            "current_ayah_text": current_text,
                            "next_ayah_text": next_text,
                            "prev_ayah_text": prev_text
                        })
                        # Advance buffer, keep only last 4 seconds to allow smooth transitions
                        if len(audio_buffer) > 16000 * 4:
                            audio_buffer = audio_buffer[-(16000 * 4):]
                    else:
                        if current_surah > 0:
                            failed_local_matches += 1
                            if failed_local_matches >= 4 and not taraweeh_mode:
                                print(f"[Backend] 4 consecutive local search failures. Clearing context lock.")
                                current_surah = 0
                                current_ayah = 0
                                failed_local_matches = 0
                                
                        await websocket.send_json({
                            "type": "status",
                            "message": f"{result.get('error', 'Listening...')} | Transcript: {result.get('transcript', '')[:25]}...",
                            "tracking_mode": result.get("tracking_mode", ""),
                            "search_window": result.get("search_window", ""),
                            "fallback_count": failed_local_matches,
                            "metrics": result.get("metrics", {})
                        })
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
