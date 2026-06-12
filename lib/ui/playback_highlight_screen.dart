import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import '../api/quran_api.dart';
import '../providers/quran_preferences.dart';

class PlaybackHighlightScreen extends StatefulWidget {
  final int chapterNumber;
  final String chapterName;
  
  const PlaybackHighlightScreen({
    Key? key, 
    required this.chapterNumber,
    this.chapterName = '',
  }) : super(key: key);

  @override
  _PlaybackHighlightScreenState createState() => _PlaybackHighlightScreenState();
}

class _PlaybackHighlightScreenState extends State<PlaybackHighlightScreen> {
  final AudioPlayer _audioPlayer = AudioPlayer();
  Map<String, dynamic>? _audioData;
  Map<String, List<String>> _versesText = {};
  String _reciterName = '';
  
  bool _isLoading = true;
  String? _errorMessage;
  
  int _currentWordIndex = -1;
  String _currentVerseKey = "";

  @override
  void initState() {
    super.initState();
    _initData();
  }

  Future<void> _initData() async {
    try {
      final reciterId = await QuranPreferences.getReciterId();
      _reciterName = await QuranPreferences.getReciterName();

      // Fetch Verses and Audio Data in parallel
      final results = await Future.wait([
        QuranApi.fetchVerses(widget.chapterNumber),
        QuranApi.fetchAudioData(widget.chapterNumber, reciterId),
      ]);

      final versesList = results[0] as List<dynamic>;
      final audioData = results[1] as Map<String, dynamic>;

      // Parse text
      final Map<String, List<String>> parsedText = {};
      for (var v in versesList) {
        final key = v['verse_key'] as String;
        final text = v['text_uthmani'] as String;
        parsedText[key] = text.split(' ');
      }

      // Parse audio URL
      String audioUrl = audioData['audio_url'] as String;
      if (!audioUrl.startsWith('http')) {
        audioUrl = 'https://audio.qurancdn.com/$audioUrl';
      }

      await _audioPlayer.setAudioSource(AudioSource.uri(Uri.parse(audioUrl)));

      _audioPlayer.positionStream.listen((position) {
        _updateHighlight(position.inMilliseconds);
      });

      setState(() {
        _versesText = parsedText;
        _audioData = audioData;
        _isLoading = false;
      });
    } catch (e) {
      print('Error initializing playback: $e');
      setState(() {
        _errorMessage = e.toString();
        _isLoading = false;
      });
    }
  }

  void _updateHighlight(int positionMs) {
    if (_audioData == null) return;
    
    final timestamps = _audioData!['timestamps'] as List<dynamic>? ?? [];
    
    for (var verse in timestamps) {
      if (positionMs >= verse['timestamp_from'] && positionMs <= verse['timestamp_to']) {
        final segments = verse['segments'] as List<dynamic>? ?? [];
        for (var segment in segments) {
          if (segment is List && segment.length >= 3) {
            final wordIndex = segment[0] as int;
            final startMs = segment[1] as int;
            final endMs = segment[2] as int;
            
            if (positionMs >= startMs && positionMs <= endMs) {
              if (_currentVerseKey != verse['verse_key'] || _currentWordIndex != wordIndex) {
                setState(() {
                  _currentVerseKey = verse['verse_key'];
                  _currentWordIndex = wordIndex;
                });
              }
              return;
            }
          }
        }
      }
    }
    
    if (_currentWordIndex != -1) {
      setState(() {
        _currentWordIndex = -1;
        _currentVerseKey = "";
      });
    }
  }

  @override
  void dispose() {
    _audioPlayer.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(widget.chapterName.isNotEmpty ? widget.chapterName : 'Surah ${widget.chapterNumber}'),
            if (_reciterName.isNotEmpty)
              Text(_reciterName, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.normal, color: Colors.white70)),
          ],
        ),
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }
    
    if (_errorMessage != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Text('Error: $_errorMessage', style: const TextStyle(color: Colors.red)),
        ),
      );
    }

    return Column(
      children: [
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: _buildVerses(),
          ),
        ),
        _buildControls(),
      ],
    );
  }

  Widget _buildVerses() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: _versesText.entries.map((entry) {
        final verseKey = entry.key;
        final words = entry.value;
        
        return Padding(
          padding: const EdgeInsets.only(bottom: 24.0),
          child: Wrap(
            alignment: WrapAlignment.end,
            textDirection: TextDirection.rtl,
            spacing: 8.0,
            runSpacing: 8.0,
            children: words.asMap().entries.map((wordEntry) {
              final apiWordIndex = wordEntry.key + 1;
              final isHighlighted = _currentVerseKey == verseKey && _currentWordIndex == apiWordIndex;
              
              return Container(
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                decoration: BoxDecoration(
                  color: isHighlighted ? Colors.green.withOpacity(0.3) : Colors.transparent,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  wordEntry.value,
                  style: TextStyle(
                    fontSize: 28,
                    color: isHighlighted ? Colors.green[800] : Colors.black87,
                  ),
                ),
              );
            }).toList(),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildControls() {
    return Container(
      padding: const EdgeInsets.all(16.0),
      color: Colors.grey[200],
      child: StreamBuilder<PlayerState>(
        stream: _audioPlayer.playerStateStream,
        builder: (context, snapshot) {
          final playerState = snapshot.data;
          final processingState = playerState?.processingState;
          final playing = playerState?.playing;
          if (processingState == ProcessingState.loading ||
              processingState == ProcessingState.buffering) {
            return Container(
              margin: const EdgeInsets.all(8.0),
              width: 64.0,
              height: 64.0,
              child: const CircularProgressIndicator(),
            );
          } else if (playing != true) {
            return IconButton(
              icon: const Icon(Icons.play_arrow),
              iconSize: 64.0,
              onPressed: _audioPlayer.play,
            );
          } else if (processingState != ProcessingState.completed) {
            return IconButton(
              icon: const Icon(Icons.pause),
              iconSize: 64.0,
              onPressed: _audioPlayer.pause,
            );
          } else {
            return IconButton(
              icon: const Icon(Icons.replay),
              iconSize: 64.0,
              onPressed: () => _audioPlayer.seek(Duration.zero),
            );
          }
        },
      ),
    );
  }
}
