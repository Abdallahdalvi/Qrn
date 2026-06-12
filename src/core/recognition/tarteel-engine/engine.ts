import { computeMelSpectrogram } from './worker/mel';
import { CTCDecoder } from './worker/ctc-decode';
import { runInference, setSession } from './worker/session';
import { buildTrie, type CompactTrie } from './lib/phoneme-trie';
import { QuranDB } from './lib/quran-db';
import { RecitationTracker, type TranscribeResult, type TrackerDiagnosticEvent, type BeamVerseMatch } from './lib/tracker';
import { beamSearchDecode } from './worker/beam-decode';
import type { InferenceSession, Tensor } from 'onnxruntime-common';
import type { QuranVerse, WorkerOutbound } from './lib/types';

const JOINT03_BEAM_WIDTH = 6;
const JOINT03_TOP_SYMBOLS = 8;
const JOINT03_MAX_HYPOTHESES = 4;
const JOINT03_SECOND_PASS_MATCH_GATE = 0.63;
const JOINT03_SECOND_PASS_FRAME_MEAN_MAX_LOGP = -0.42;
const JOINT03_DECODE2_BEAM_WIDTH = 11;
const JOINT03_DECODE2_TOP_SYMBOLS = 12;
const JOINT03_DECODE2_MAX_EXTRA_STRINGS = 4;

function logAddExp(a: number | undefined, b: number): number {
  if (a === undefined) return b;
  const hi = Math.max(a, b);
  const lo = Math.min(a, b);
  return hi + Math.log1p(Math.exp(lo - hi));
}

function topSymbolsForFrame(logprobs: Float32Array, offset: number, vocabSize: number, topSymbols: number): number[] {
  const keep = Math.min(topSymbols, vocabSize);
  const out: Array<{ id: number; value: number }> = [];
  for (let id = 0; id < vocabSize; id++) {
    const value = logprobs[offset + id];
    if (out.length < keep) {
      out.push({ id, value });
      out.sort((a, b) => b.value - a.value);
    } else if (value > out[out.length - 1].value) {
      out[out.length - 1] = { id, value };
      out.sort((a, b) => b.value - a.value);
    }
  }
  return out.map((entry) => entry.id);
}

function ctcPrefixBeamDecode(
  logprobs: Float32Array, timeSteps: number, vocabSize: number, blankId: number, beamWidth: number, topSymbols: number
): Array<{ ids: number[]; score: number }> {
  let beam = new Map<string, { ids: number[]; score: number }>();
  beam.set("", { ids: [], score: 0 });

  for (let t = 0; t < timeSteps; t++) {
    const offset = t * vocabSize;
    const symbols = topSymbolsForFrame(logprobs, offset, vocabSize, topSymbols);
    const nextBeam = new Map<string, { ids: number[]; score: number }>();
    const items = [...beam.values()].sort((a, b) => b.score - a.score).slice(0, Math.max(beamWidth * 3, beamWidth));

    for (const item of items) {
      for (const c of symbols) {
        const logp = item.score + logprobs[offset + c];
        let ids = item.ids;
        if (c !== blankId && item.ids[item.ids.length - 1] !== c) {
          ids = [...item.ids, c];
        }
        const key = ids.join(",");
        nextBeam.set(key, { ids, score: logAddExp(nextBeam.get(key)?.score, logp) });
      }
    }

    beam = new Map([...nextBeam.entries()].sort((a, b) => b[1].score - a[1].score).slice(0, beamWidth));
  }
  return [...beam.values()].sort((a, b) => b.score - a.score).slice(0, beamWidth);
}

function appendBeamHypotheses(
  hypotheses: string[], seen: Set<string>, decoder: CTCDecoder, logprobs: Float32Array, timeSteps: number, vocabSize: number, blankId: number, beamWidth: number, topSymbols: number, maxExtra: number
) {
  for (const result of ctcPrefixBeamDecode(logprobs, timeSteps, vocabSize, blankId, beamWidth, topSymbols)) {
    const text = decoder.tokenIdsToText(result.ids);
    if (!text.trim() || seen.has(text)) continue;
    seen.add(text);
    hypotheses.push(text);
    if (hypotheses.length >= maxExtra) break;
  }
}

function meanFrameMaxLogp(logprobs: Float32Array, timeSteps: number, vocabSize: number): number {
  let sum = 0;
  for (let t = 0; t < timeSteps; t++) {
    const offset = t * vocabSize;
    let max = logprobs[offset];
    for (let v = 1; v < vocabSize; v++) {
      max = Math.max(max, logprobs[offset + v]);
    }
    sum += max;
  }
  return sum / Math.max(1, timeSteps);
}

export class TarteelEngine {
  private decoder: CTCDecoder;
  private db: QuranDB;
  private trie: CompactTrie;
  private tracker: RecitationTracker;
  private debugEnabled = false;

  constructor(
    vocabJson: Record<string, string>,
    quranData: QuranVerse[],
    ortSession: InferenceSession,
    ortTensor: typeof Tensor,
    onTrackerEvent?: (event: TrackerDiagnosticEvent) => void
  ) {
    setSession(ortSession, ortTensor);
    this.decoder = new CTCDecoder(vocabJson);
    this.db = new QuranDB(quranData, this.decoder);
    
    // Build verse/span trie for constrained beam search
    const built = buildTrie(quranData, vocabJson, 3);
    this.trie = built.trie;

    this.tracker = new RecitationTracker(this.db, this.transcribe.bind(this), {
      onDiagnostic: onTrackerEvent,
    });
  }

  public async feedAudio(samples: Float32Array): Promise<WorkerOutbound[]> {
    return await this.tracker.feed(samples);
  }

  private async transcribe(audio: Float32Array): Promise<TranscribeResult> {
    const { features, timeFrames } = await computeMelSpectrogram(audio);
    const numMels = 80;
    const { logprobs, timeSteps, vocabSize } = await runInference(features, numMels, timeFrames);

    const greedy = this.decoder.decode(logprobs, timeSteps, vocabSize);
    const blankId = this.decoder.getBlankId();
    const hypotheses: string[] = [];
    const seenHypotheses = new Set<string>();
    
    if (greedy.text.trim()) {
      hypotheses.push(greedy.text);
      seenHypotheses.add(greedy.text);
    }
    
    appendBeamHypotheses(
      hypotheses, seenHypotheses, this.decoder, logprobs, timeSteps, vocabSize, blankId,
      JOINT03_BEAM_WIDTH, JOINT03_TOP_SYMBOLS, JOINT03_MAX_HYPOTHESES
    );
    
    const firstPass = this.db.bestJoint03MatchForHypotheses(hypotheses);
    if (
      !firstPass ||
      firstPass.match.score < JOINT03_SECOND_PASS_MATCH_GATE ||
      meanFrameMaxLogp(logprobs, timeSteps, vocabSize) < JOINT03_SECOND_PASS_FRAME_MEAN_MAX_LOGP
    ) {
      const before = hypotheses.length;
      appendBeamHypotheses(
        hypotheses, seenHypotheses, this.decoder, logprobs, timeSteps, vocabSize, blankId,
        JOINT03_DECODE2_BEAM_WIDTH, JOINT03_DECODE2_TOP_SYMBOLS, before + JOINT03_DECODE2_MAX_EXTRA_STRINGS
      );
    }
    
    const champion = this.db.bestJoint03MatchForHypotheses(hypotheses);
    const championTranscript = champion?.transcript;

    let beamMatches: BeamVerseMatch[] | undefined;
    if (this.trie) {
      const beamResults = beamSearchDecode(logprobs, timeSteps, vocabSize, blankId, this.trie, 8);
      const seen = new Set<string>();
      beamMatches = [];
      for (const result of beamResults) {
        for (const ref of result.matchedVerses) {
          const key = `${ref.verseIndex}:${ref.spanLength}`;
          if (!seen.has(key)) {
            seen.add(key);
            beamMatches.push({ verseIndex: ref.verseIndex, spanLength: ref.spanLength, score: result.score });
          }
        }
      }
    }

    return {
      ...greedy,
      text: championTranscript ?? greedy.text,
      acoustic: { logprobs, timeSteps, vocabSize, blankId },
      beamMatches,
      championMatch: champion?.match,
      championTranscript,
    };
  }
}
