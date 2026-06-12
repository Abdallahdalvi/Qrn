import { QuranMatcher, QuranVerse } from './QuranMatcher';

describe('QuranMatcher', () => {
  const mockDatabase: QuranVerse[] = [
    { surah: 1, ayah: 1, text: "بسم الله الرحمن الرحيم" },
    { surah: 1, ayah: 2, text: "الحمد لله رب العالمين" },
    { surah: 1, ayah: 3, text: "الرحمن الرحيم" },
    { surah: 1, ayah: 4, text: "مالك يوم الدين" }
  ];

  let matcher: QuranMatcher;

  beforeEach(() => {
    matcher = new QuranMatcher(mockDatabase);
  });

  it('should return null for empty transcript', () => {
    expect(matcher.match("")).toBeNull();
  });

  it('should find an exact match', () => {
    const result = matcher.match("الحمد لله");
    expect(result).not.toBeNull();
    if (result) {
      expect(result.surah).toBe(1);
      expect(result.ayah).toBe(2);
      expect(result.wordPosition).toBe(0);
      expect(result.transcript).toBe("الحمد لله");
    }
  });

  it('should find match with diacritics in transcript removed', () => {
    // Transcript with tashkeel
    const result = matcher.match("الْحَمْدُ لِلَّهِ");
    expect(result).not.toBeNull();
    if (result) {
      expect(result.surah).toBe(1);
      expect(result.ayah).toBe(2);
    }
  });

  it('should calculate word position correctly', () => {
    const result = matcher.match("رب العالمين");
    expect(result).not.toBeNull();
    if (result) {
      expect(result.surah).toBe(1);
      expect(result.ayah).toBe(2);
      // "الحمد لله " -> 2 words before "رب العالمين"
      expect(result.wordPosition).toBe(2);
    }
  });
});
