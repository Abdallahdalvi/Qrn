import { RecognitionResult } from '../RecitationRecognizer';

export interface QuranVerse {
  surah: number;
  ayah: number;
  text: string;
}

export class QuranMatcher {
  private verses: QuranVerse[] = [];

  constructor(verses: QuranVerse[]) {
    this.verses = verses;
  }

  /**
   * Matches a given transcript fragment against the Quranic text.
   * This is a simplified placeholder implementation.
   * In a real-world scenario, this would use a more robust fuzzy matching algorithm
   * (e.g., Levenshtein distance, tokenization, phoneme matching) to handle errors
   * in the ASR transcript.
   * 
   * @param transcript The text transcribed by the ASR model.
   * @returns A RecognitionResult if a match is found, otherwise null.
   */
  public match(transcript: string): RecognitionResult | null {
    // Normalize transcript (remove punctuation, normalize Arabic characters if needed)
    const normalizedTranscript = this.normalizeArabic(transcript);

    if (!normalizedTranscript) {
      return null;
    }

    let bestMatch: RecognitionResult | null = null;
    let highestScore = 0;

    for (const verse of this.verses) {
      const normalizedVerse = this.normalizeArabic(verse.text);
      
      // Simple substring search for demonstration
      // A production system needs dynamic programming / sequence alignment
      const index = normalizedVerse.indexOf(normalizedTranscript);
      
      if (index !== -1) {
        // Calculate a basic "word position" by counting spaces before the match
        const prefix = normalizedVerse.substring(0, index);
        const wordPosition = prefix.trim()
          ? prefix.trim().split(/\s+/).length
          : 0;

        // Simplified confidence score based on exact match length ratio
        const confidenceScore = normalizedTranscript.length / normalizedVerse.length;

        if (confidenceScore > highestScore) {
          highestScore = confidenceScore;
          bestMatch = {
            surah: verse.surah,
            ayah: verse.ayah,
            wordPosition: wordPosition,
            confidenceScore: confidenceScore,
            transcript: transcript
          };
        }
      }
    }

    return bestMatch;
  }

  private normalizeArabic(text: string): string {
    // Basic normalization: remove diacritics (tashkeel), normalize alef, etc.
    return text
      .replace(/[\u0617-\u061A\u064B-\u0652]/g, "") // Remove harakat
      .replace(/[أإآ]/g, "ا") // Normalize alef
      .trim();
  }
}
