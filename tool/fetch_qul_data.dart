import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

Future<void> main(List<String> arguments) async {
  if (arguments.isEmpty) {
    print('Usage: dart run tool/fetch_qul_data.dart <chapter_number> [reciter_id]');
    print('Example: dart run tool/fetch_qul_data.dart 1 7');
    exit(1);
  }

  final chapterNumber = int.parse(arguments[0]);
  final reciterId = arguments.length > 1 ? int.parse(arguments[1]) : 7; // 7 is Mishary Al-Afasy by default

  final outputDir = Directory('assets/mock_recitations');
  if (!await outputDir.exists()) {
    await outputDir.create(recursive: true);
  }

  print('Fetching data for Chapter $chapterNumber, Reciter $reciterId...');

  // Using Quran.com v4 API which serves QUL segment data
  final uri = Uri.parse('https://api.quran.com/api/v4/chapter_recitations/$reciterId/$chapterNumber?segments=true');
  final response = await http.get(uri);

  if (response.statusCode != 200) {
    print('Failed to fetch API data: ${response.statusCode}');
    print(response.body);
    exit(1);
  }

  final data = jsonDecode(response.body);
  final audioUrl = data['audio_file']['audio_url'];
  
  // Clean up URL if it lacks scheme
  final fullAudioUrl = audioUrl.startsWith('http') ? audioUrl : 'https://api.quran.com$audioUrl'; // Sometimes it needs this or audio.qurancdn.com

  print('Downloading audio from $fullAudioUrl...');
  final audioResponse = await http.get(Uri.parse(audioUrl.startsWith('http') ? audioUrl : 'https://$audioUrl'));
  
  if (audioResponse.statusCode != 200) {
    print('Failed to download audio: ${audioResponse.statusCode}');
    // Attempt alternative CDN logic if needed
    final altResponse = await http.get(Uri.parse('https://audio.qurancdn.com/$audioUrl'));
    if (altResponse.statusCode == 200) {
       final audioFile = File('${outputDir.path}/$chapterNumber.mp3');
       await audioFile.writeAsBytes(altResponse.bodyBytes);
    } else {
       print('Failed alternative CDN too.');
       exit(1);
    }
  } else {
    final audioFile = File('${outputDir.path}/$chapterNumber.mp3');
    await audioFile.writeAsBytes(audioResponse.bodyBytes);
  }

  // Save the JSON segments
  final jsonFile = File('${outputDir.path}/$chapterNumber.json');
  await jsonFile.writeAsString(jsonEncode(data['audio_file']));

  print('Successfully saved MP3 and JSON data to ${outputDir.path}');
}
