export interface RecognitionResult {
  surah: number;
  ayah: number;
  wordPosition: number;
  confidenceScore: number;
  transcript?: string;
}

export interface RecitationRecognizer {
  /**
   * Initializes the recognizer (e.g., loads models, establishes API connections).
   */
  initialize(): Promise<void>;

  /**
   * Starts listening/processing mode.
   * @param onResult Callback invoked when a position match is found.
   */
  startListening(onResult: (result: RecognitionResult) => void): Promise<void>;

  /**
   * Stops the recognizer and cleans up resources.
   */
  stopListening(): Promise<void>;

  /**
   * Feeds a chunk of 16kHz mono audio (Float32Array) to the recognizer.
   * Useful for platform-independent architectures where the audio capture
   * happens outside this provider.
   * @param samples Audio samples (-1.0 to 1.0)
   */
  processAudioChunk?(samples: Float32Array): Promise<void>;
}
