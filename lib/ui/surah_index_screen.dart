import 'package:flutter/material.dart';
import '../api/quran_api.dart';
import 'qari_selection_screen.dart';
import 'playback_highlight_screen.dart';

class SurahIndexScreen extends StatefulWidget {
  const SurahIndexScreen({Key? key}) : super(key: key);

  @override
  _SurahIndexScreenState createState() => _SurahIndexScreenState();
}

class _SurahIndexScreenState extends State<SurahIndexScreen> {
  List<dynamic> _chapters = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadChapters();
  }

  Future<void> _loadChapters() async {
    try {
      final chapters = await QuranApi.fetchChapters();
      setState(() {
        _chapters = chapters;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to load Surahs: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Quran Reader'),
        actions: [
          IconButton(
            icon: const Icon(Icons.person_search),
            tooltip: 'Select Qari',
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const QariSelectionScreen()),
              );
            },
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : ListView.separated(
              itemCount: _chapters.length,
              separatorBuilder: (_, __) => const Divider(height: 1),
              itemBuilder: (context, index) {
                final chapter = _chapters[index];
                final chapterId = chapter['id'] as int;
                final nameSimple = chapter['name_simple'] ?? '';
                final nameArabic = chapter['name_arabic'] ?? '';
                final translatedName = chapter['translated_name']?['name'] ?? '';
                final verseCount = chapter['verses_count'] ?? 0;

                return ListTile(
                  leading: CircleAvatar(
                    backgroundColor: Colors.blue.shade50,
                    child: Text(
                      '$chapterId',
                      style: const TextStyle(color: Colors.blue, fontWeight: FontWeight.bold),
                    ),
                  ),
                  title: Text(nameSimple, style: const TextStyle(fontWeight: FontWeight.bold)),
                  subtitle: Text('$translatedName • $verseCount verses'),
                  trailing: Text(
                    nameArabic,
                    style: const TextStyle(fontSize: 20, color: Colors.blue),
                  ),
                  onTap: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => PlaybackHighlightScreen(
                          chapterNumber: chapterId,
                          chapterName: nameSimple,
                        ),
                      ),
                    );
                  },
                );
              },
            ),
    );
  }
}
