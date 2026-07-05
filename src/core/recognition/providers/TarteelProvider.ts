import { RecitationRecognizer, RecognitionResult } from '../RecitationRecognizer';
import { TarteelEngine } from '../tarteel-engine/engine';
import type { InferenceSession, Tensor } from 'onnxruntime-common';
import type { QuranVerse, WorkerOutbound } from '../tarteel-engine/lib/types';

export interface AssetLoader {
  loadVocab(): Promise<Record<string, string>>;
  loadQuranData(): Promise<QuranVerse[]>;
  loadModel(): Promise<ArrayBuffer>;
  getOrtSessionClass(): typeof InferenceSession;
  getOrtTensorClass(): typeof Tensor;
}

export class TarteelProvider implements RecitationRecognizer {
  private engine: TarteelEngine | null = null;
  private isListening: boolean = false;
  private assetLoader: AssetLoader;
  private onResultCallback: ((result: RecognitionResult) => void) | null = null;
  private onMessageCallback: ((message: WorkerOutbound) => void) | null = null;

  constructor(assetLoader: AssetLoader) {
    this.assetLoader = assetLoader;
  }

  public async initialize(): Promise<void> {
    console.log('Initializing TarteelProvider...');
    this.onMessageCallback?.({ type: 'loading_status', message: 'Loading phoneme vocabulary...' });
    const vocab = await this.assetLoader.loadVocab();
    this.onMessageCallback?.({ type: 'loading_status', message: 'Loading Quran phoneme index...' });
    const quranData = await this.assetLoader.loadQuranData();
    this.onMessageCallback?.({ type: 'loading_status', message: 'Loading recognition model...' });
    const modelBuffer = await this.assetLoader.loadModel();
    const OrtSession = this.assetLoader.getOrtSessionClass();
    const OrtTensor = this.assetLoader.getOrtTensorClass();

    this.onMessageCallback?.({ type: 'loading_status', message: 'Starting ONNX Runtime session...' });
    const session = await OrtSession.create(modelBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      executionMode: 'sequential',
    });
    
    this.engine = new TarteelEngine(vocab, quranData, session, OrtTensor, (event) => {
      // Diagnostic events from the tracker
      if (event.type === 'commit' && this.onResultCallback) {
        // TarteelTracker emits 'commit' with ref like "1:2"
        const [surah, ayah] = event.ref.split(':').map(Number);
        this.onResultCallback({
          surah,
          ayah,
          wordPosition: 0, // Tracker emits commits at verse level. We also have word_progress events
          confidenceScore: event.confidence,
          transcript: event.top_ref || undefined // optional
        });
      } else if (event.type === 'tracking_cycle' && this.onResultCallback) {
        // Provide granular word progress updates
        const [surah, ayah] = event.ref.split(':').map(Number);
        this.onResultCallback({
            surah,
            ayah,
            wordPosition: event.word_matches,
            confidenceScore: 0.9, // Continuous tracking implies high confidence if advanced
        });
      }
    });

    console.log('TarteelProvider initialized.');
  }

  public async startListening(onResult: (result: RecognitionResult) => void): Promise<void> {
    if (this.isListening) {
      console.warn('TarteelProvider is already listening.');
      return;
    }
    console.log('TarteelProvider started listening.');
    this.isListening = true;
    this.onResultCallback = onResult;
  }

  public onResult(onResult: (result: RecognitionResult) => void): void {
    this.onResultCallback = onResult;
    this.isListening = true;
  }

  public onMessage(onMessage: (message: WorkerOutbound) => void): void {
    this.onMessageCallback = onMessage;
    this.isListening = true;
  }

  public async stopListening(): Promise<void> {
    console.log('TarteelProvider stopped listening.');
    this.isListening = false;
    this.onResultCallback = null;
    this.onMessageCallback = null;
  }

  public endAudioStream(): void {
    this.engine?.reset();
    this.isListening = true;
  }

  public resetAudioStream(): void {
    this.engine?.reset();
    this.isListening = true;
  }

  public startTaraweeh(surah: number, ayah: number): void {
    this.engine?.setTaraweehLock(surah, ayah);
    this.isListening = true;
  }

  public stopTaraweeh(): void {
    this.engine?.clearTaraweehLock();
    this.isListening = true;
  }

  public async processAudioChunk(samples: Float32Array): Promise<void> {
    if (!this.isListening || !this.engine) return;
    
    // Feed the raw Float32Array to the offline ONNX engine
    const messages = await this.engine.feedAudio(samples);
    
    // We can also parse the direct Word_progress / Verse_match messages here
    for (const msg of messages) {
      this.onMessageCallback?.(msg);
      if (msg.type === 'word_progress' && this.onResultCallback) {
        this.onResultCallback({
          surah: msg.surah,
          ayah: msg.ayah,
          wordPosition: msg.word_index,
          confidenceScore: msg.confidence ?? 0.95
        });
      } else if (msg.type === 'verse_match' && this.onResultCallback) {
        this.onResultCallback({
            surah: msg.surah,
            ayah: msg.ayah,
            wordPosition: 0,
            confidenceScore: msg.confidence
        });
      }
    }
  }
}
