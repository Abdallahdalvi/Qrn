import 'dart:convert';
import 'package:http/http.dart' as http;

class QuranApi {
  static const String baseUrl = 'https://api.quran.com/api/v4';
  
  static Map<String, String> get _headers => {
    'User-Agent': 'AlFatih/1.0.0 (Mobile)',
    'Accept': 'application/json',
  };

  /// Fetches all 114 Chapters (Surahs)
  static Future<List<dynamic>> fetchChapters() async {
    final response = await http.get(Uri.parse('$baseUrl/chapters'), headers: _headers);
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['chapters'] as List<dynamic>;
    } else {
      throw Exception('Failed to load chapters: ${response.statusCode}');
    }
  }

  /// Fetches the list of all available Qaris (Reciters)
  static Future<List<dynamic>> fetchReciters() async {
    final response = await http.get(Uri.parse('$baseUrl/resources/recitations'), headers: _headers);
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['recitations'] as List<dynamic>;
    } else {
      throw Exception('Failed to load reciters: ${response.statusCode}');
    }
  }

  /// Fetches Uthmani text for a specific chapter
  static Future<List<dynamic>> fetchVerses(int chapterId) async {
    final response = await http.get(Uri.parse('$baseUrl/quran/verses/uthmani?chapter_number=$chapterId'), headers: _headers);
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['verses'] as List<dynamic>;
    } else {
      throw Exception('Failed to load verses for chapter $chapterId: ${response.statusCode}');
    }
  }

  /// Fetches the audio URL and word segment timestamps for a given chapter and reciter
  static Future<Map<String, dynamic>> fetchAudioData(int chapterId, int reciterId) async {
    final response = await http.get(Uri.parse('$baseUrl/chapter_recitations/$reciterId/$chapterId?segments=true'), headers: _headers);
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['audio_file'] as Map<String, dynamic>;
    } else {
      throw Exception('Failed to load audio data for chapter $chapterId: ${response.statusCode}');
    }
  }
}
