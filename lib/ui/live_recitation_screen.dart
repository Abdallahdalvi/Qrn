import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import 'package:record/record.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../engine/socket_client.dart';
import '../audio/audio_capture.dart';
import '../core/global_logger.dart';
import 'settings_screen.dart';
import 'open_source_acknowledgements.dart';

class LiveRecitationScreen extends StatefulWidget {
  final TarteelSocketClient engine;
  final AudioCaptureService audioService;

  const LiveRecitationScreen({
    Key? key,
    required this.engine,
    required this.audioService,
  }) : super(key: key);

  @override
  _LiveRecitationScreenState createState() => _LiveRecitationScreenState();
}

enum PromptState {
  DISABLED,
  IDLE,
  WAITING_FOR_LOCK,
  PROMPT_READY,
  PAUSE_TIMER_RUNNING,
  PROMPT_DISPLAYED,
  PLAYING_AUDIO,
  PROMPT_REPEAT,
  PROMPT_EXPIRED,
  RECOVERY_MODE
}

class _LiveRecitationScreenState extends State<LiveRecitationScreen> {
  bool _isListening = false;
  StreamSubscription<Uint8List>? _audioSub;
  String _connectionStatus = 'Disconnected';
  String _recognitionStatus = 'Idle';
  int _currentSurah = 0;
  int _currentAyah = 0;
  int _lastSurah = 0;
  int _lastAyah = 0;
  String _surahNameEn = '';
  String _surahNameAr = '';
  num _confidence = 0.0;
  int _wordPosition = 0;
  int _totalWords = 0;
  String _transcript = '';
  
  String _trackingMode = '';
  String _searchWindow = '';
  int _fallbackCount = 0;
  bool _taraweehModeEnabled = false;
  bool _debugModeEnabled = false;
  bool _showDebugScreen = false;
  
  bool _promptModeEnabled = false;
  double _vadThreshold = -20.0;
  
  String _currentAyahText = '';
  String _nextAyahText = '';
  String _prevAyahText = '';
  
  int _promptTimeout = 15;
  String _promptAggressiveness = 'Normal';
  int _promptRepeatInterval = 10;
  int _promptMaxRepeats = 3;
  bool _isSpeaking = false;
  DateTime _lastSpeechTime = DateTime.now();
  DateTime _lastProgressionTime = DateTime.now();
  double _currentRMS = -100.0;
  PromptState _promptState = PromptState.DISABLED;
  String _promptStateMessage = 'PROMPT DISABLED';
  int _promptRepeatCount = 0;
  bool _isPromptConfirmed = false;
  int _assistedAyah = 0;
  Timer? _promptLoopTimer;
  StreamSubscription<Amplitude>? _ampSub;
  int _lastLoggedPauseSecond = -1;
  
  bool _isMutashabihat = false;
  final AudioPlayer _audioPlayer = AudioPlayer();
  bool _audioPlayedForCurrentPause = false;

  StreamSubscription<String>? _logSub;

  @override
  void initState() {
    super.initState();
    _logs.addAll(globalLogger.history);
    _logSub = globalLogger.onLog.listen((logMessage) {
      if (!mounted) return;
      setState(() {
        _logs.insert(0, logMessage);
        if (_logs.length > 200) _logs.removeLast();
      });
    });
    
    _listenToEngine();
    _initEngine();
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      _promptTimeout = prefs.getInt('prompt_timeout') ?? 15;
      _promptAggressiveness = prefs.getString('prompt_aggressiveness') ?? 'Normal';
      _promptRepeatInterval = prefs.getInt('prompt_repeat_interval') ?? 10;
      _promptMaxRepeats = prefs.getInt('prompt_max_repeats') ?? 3;
      _debugModeEnabled = prefs.getBool('debug_mode') ?? false;
      _promptModeEnabled = _promptAggressiveness != 'Off';
      _vadThreshold = prefs.getDouble('vad_threshold') ?? -20.0;
    });
  }

  @override
  void dispose() {
    _logSub?.cancel();
    _audioSub?.cancel();
    _ampSub?.cancel();
    _promptLoopTimer?.cancel();
    _audioPlayer.dispose();
    super.dispose();
  }

  Future<void> _initEngine() async {
    try {
      _addLog('Connecting to recognition server...');
      setState(() {
        _connectionStatus = 'Connecting...';
        _recognitionStatus = 'Initializing...';
      });
      await widget.engine.initialize();
    } catch (e) {
      _addLog('[ERROR] Critical connection failure: $e');
      setState(() {
        _connectionStatus = 'Failed';
        _recognitionStatus = 'Connection failure';
      });
    }
  }



  final List<String> _logs = [];

  void _addLog(String logMsg) {
    globalLogger.log(logMsg);
  }

  void _setPromptState(PromptState newState, String message) {
    if (_promptState != newState || _promptStateMessage != message) {
      if (_promptState != newState) {
        _addLog('[Prompt] Transition: ${_promptState.name} -> ${newState.name} ($message)');
      }
      setState(() {
         _promptState = newState;
         _promptStateMessage = message;
      });
    }
  }

  void _evaluatePromptState() {
    if (!mounted || !_isListening || !_promptModeEnabled) {
      _setPromptState(PromptState.DISABLED, 'PROMPT DISABLED');
      _cancelPrompt();
      return;
    }

    if (_promptAggressiveness == 'Off') {
       _setPromptState(PromptState.DISABLED, 'PROMPT DISABLED');
       _cancelPrompt();
       return;
    }

    // Determine Anchor Position
    int anchorSurah = _currentSurah > 0 ? _currentSurah : _lastSurah;
    int anchorAyah = _currentAyah > 0 ? _currentAyah : _lastAyah;
    bool hasValidAnchor = anchorSurah > 0 && anchorAyah > 0;

    if (!hasValidAnchor) {
      _setPromptState(PromptState.WAITING_FOR_LOCK, 'WAITING FOR FIRST LOCK OR TARAWEEH CONFIG');
      _cancelPrompt();
      return;
    }

    if (_isSpeaking) {
      _setPromptState(PromptState.PROMPT_READY, 'PROMPT READY (Speech Active)');
      _cancelPrompt();
      return;
    }

    final now = DateTime.now();
    final idleTimeSpeech = now.difference(_lastSpeechTime).inMilliseconds;
    final idleTimeProgression = now.difference(_lastProgressionTime).inMilliseconds;
    
    final idleMs = idleTimeSpeech < idleTimeProgression ? idleTimeSpeech : idleTimeProgression;
    
    int currentPauseSeconds = idleMs ~/ 1000;
    if (currentPauseSeconds > 0 && currentPauseSeconds != _lastLoggedPauseSecond && !_isSpeaking) {
        _lastLoggedPauseSecond = currentPauseSeconds;
        _addLog('[PAUSE] timer = $currentPauseSeconds');
    }
    
    final totalTimeoutMs = _promptTimeout * 1000;

    if (idleMs >= totalTimeoutMs + 3000 + (_promptRepeatCount * _promptRepeatInterval * 1000)) {
      if (_promptRepeatCount >= _promptMaxRepeats) {
         _setPromptState(PromptState.PROMPT_EXPIRED, 'Waiting for recitation (Max repeats reached)');
      } else if (_promptRepeatCount == 0 && _promptState != PromptState.PLAYING_AUDIO) {
         _addLog('[PROMPT] triggered');
         _setPromptState(PromptState.PLAYING_AUDIO, 'PLAYING AUDIO');
         setState(() {
           _assistedAyah = anchorAyah + 1;
         });
         widget.engine.sendAssistedPrompt(_assistedAyah);
         _playPromptAudio(anchorSurah, anchorAyah + 1);
      } else if (_promptRepeatCount > 0 && _promptRepeatCount < _promptMaxRepeats) {
         if (_promptState != PromptState.PROMPT_REPEAT || _promptStateMessage != 'PLAYING AUDIO (Repeat $_promptRepeatCount)') {
            _addLog('[PROMPT] triggered (Repeat $_promptRepeatCount)');
            _setPromptState(PromptState.PROMPT_REPEAT, 'PLAYING AUDIO (Repeat $_promptRepeatCount)');
            _playPromptAudio(anchorSurah, anchorAyah + 1, forceRepeat: true);
         }
      }
      
      if (_promptRepeatCount < _promptMaxRepeats && idleMs >= totalTimeoutMs + 3000 + ((_promptRepeatCount + 1) * _promptRepeatInterval * 1000)) {
         _promptRepeatCount++;
      }
    } else if (idleMs >= totalTimeoutMs) {
      final countdown = 3 - ((idleMs - totalTimeoutMs) ~/ 1000);
      final cdText = countdown > 0 ? 'COUNTDOWN: $countdown' : 'COUNTDOWN: 1';
      _setPromptState(PromptState.PAUSE_TIMER_RUNNING, cdText);
    } else {
      _setPromptState(PromptState.PAUSE_TIMER_RUNNING, 'Silence >= ${currentPauseSeconds}s (Timeout in ${(_promptTimeout - currentPauseSeconds)}s)');
    }
  }

  void _cancelPrompt() {
     _audioPlayer.stop();
     _audioPlayedForCurrentPause = false;
     _promptRepeatCount = 0;
     if (_assistedAyah > 0) {
        _assistedAyah = 0;
        widget.engine.clearAssistedPrompt();
     }
  }
  
  void _playPromptAudio(int surah, int ayah, {bool forceRepeat = false}) {
      if (_audioPlayedForCurrentPause && !forceRepeat) return;
      _audioPlayedForCurrentPause = true;
      String s = surah.toString().padLeft(3, '0');
      String a = ayah.toString().padLeft(3, '0');
      String audioUrl = "https://everyayah.com/data/Alafasy_128kbps/$s$a.mp3";
      _addLog('[PROMPT] Audio Download Started: $audioUrl');
      _audioPlayer.setUrl(audioUrl).then((_) {
         _addLog('[PROMPT] Audio Playback Start');
         _audioPlayer.play();
      }).catchError((e) {
         debugPrint("Audio play error: $e");
         _addLog('[PROMPT] Audio Playback Error: $e');
      });
  }

  void _handleEvent(Map<String, dynamic> event) {
    final type = event['type'];
    globalLogger.log('SERVER RESPONSE: $type');
    if (type == 'status') {
      final msg = event['message'] ?? '';
      _trackingMode = event['tracking_mode'] ?? _trackingMode;
      _searchWindow = event['search_window'] ?? _searchWindow;
      _fallbackCount = event['fallback_count'] ?? _fallbackCount;
      _addLog(msg);
      if (msg.contains('Connecting')) {
        _connectionStatus = 'Connecting...';
      } else if (msg.contains('Connected')) {
        _connectionStatus = 'Connected';
      } else if (msg.contains('closed')) {
        _connectionStatus = 'Disconnected';
        _recognitionStatus = 'WebSocket disconnected';
      }
    } else if (type == 'ready') {
      _connectionStatus = 'Connected';
      _recognitionStatus = 'Ready. Press mic to start.';
      _addLog('WebSocket connected and ready.');
    } else if (type == 'verse_match') {
      int newSurah = event['surah'] ?? 0;
      int newAyah = event['ayah'] ?? 0;
      
      if (newSurah != _currentSurah || newAyah != _currentAyah) {
         _lastProgressionTime = DateTime.now();
         _cancelPrompt();
      }

      if (_assistedAyah > 0 && newAyah == _assistedAyah) {
         setState(() {
           _isPromptConfirmed = true;
         });
         Future.delayed(const Duration(seconds: 2), () {
           if (mounted) setState(() => _isPromptConfirmed = false);
         });
         _cancelPrompt();
      }
      
      setState(() {
         if (_currentSurah > 0 && _currentAyah > 0 && _confidence >= 0.8) {
            _lastSurah = _currentSurah;
            _lastAyah = _currentAyah;
         }
         _currentSurah = newSurah;
         _currentAyah = newAyah;
         _confidence = event['confidence'] ?? 0.0;
         _surahNameEn = event['surah_name_en'] ?? '';
         _surahNameAr = event['surah_name'] ?? '';
         _transcript = event['transcript'] ?? '';
         _trackingMode = event['tracking_mode'] ?? '';
         _searchWindow = event['search_window'] ?? '';
         _fallbackCount = event['fallback_count'] ?? 0;
         _currentAyahText = event['current_ayah_text'] ?? _currentAyahText;
      _nextAyahText = event['next_ayah_text'] ?? _nextAyahText;
      _prevAyahText = event['prev_ayah_text'] ?? _prevAyahText;
      _isMutashabihat = event['is_mutashabihat'] ?? false;
      _wordPosition = 0;
      _totalWords = 0;
      if (_trackingMode == 'REWIND DETECTED (Confirming...)') {
        _recognitionStatus = 'Matched Verse: $_currentSurah:$_currentAyah (Rewind Detected)';
      } else {
        _recognitionStatus = 'Matched Verse: $_currentSurah:$_currentAyah';
      }
      _addLog('Match - Surah: $_surahNameEn ($_currentSurah), Ayah: $_currentAyah, Conf: ${(_confidence * 100).toStringAsFixed(0)}%');
      });
    } else if (type == 'word_progress') {
      int newSurah = event['surah'] ?? _currentSurah;
      int newAyah = event['ayah'] ?? _currentAyah;
      int newWord = event['word_index'] ?? 0;
      
      if (newSurah != _currentSurah || newAyah != _currentAyah || newWord != _wordPosition) {
         _lastProgressionTime = DateTime.now();
      }

      _currentSurah = newSurah;
      _currentAyah = newAyah;
      _wordPosition = newWord;
      _totalWords = event['total_words'] ?? 0;
      _confidence = event['confidence'] ?? _confidence;
      _trackingMode = event['tracking_mode'] ?? _trackingMode;
      _searchWindow = event['search_window'] ?? _searchWindow;
      _fallbackCount = event['fallback_count'] ?? _fallbackCount;
      _currentAyahText = event['current_ayah_text'] ?? _currentAyahText;
      _nextAyahText = event['next_ayah_text'] ?? _nextAyahText;
      _prevAyahText = event['prev_ayah_text'] ?? _prevAyahText;
      _isMutashabihat = event['is_mutashabihat'] ?? false;
      _recognitionStatus = 'Tracking... Ayah: $_currentAyah | Word: $_wordPosition/$_totalWords';
    } else if (type == 'error') {
      _connectionStatus = 'Failed';
      _recognitionStatus = "Error: ${event['message']}";
      _addLog("[ERROR] ${event['message']}");
    }
  }

  void _listenToEngine() {
    for (final event in widget.engine.eventHistory) {
      _handleEvent(event);
    }
    widget.engine.onEvent.listen((event) {
      if (!mounted) return;
      setState(() {
        _handleEvent(event);
      });
    });
  }

  Future<void> _toggleListening() async {
    globalLogger.log('MIC BUTTON PRESSED');
    try {
      if (_isListening) {
        _addLog('Stopping microphone...');
        _cancelPrompt();
        await _audioSub?.cancel();
        _audioSub = null;
        await _ampSub?.cancel();
        _ampSub = null;
        _promptLoopTimer?.cancel();
        _promptLoopTimer = null;
        await widget.audioService.stop();
        setState(() {
          _isListening = false;
          _recognitionStatus = 'Ready. Press mic to start.';
        });
        _addLog('Microphone stopped.');
      } else {
        final hasPermission = await widget.audioService.requestPermission();
        if (!hasPermission) {
          _addLog('[ERROR] Microphone permission denied');
          setState(() {
            _recognitionStatus = 'Microphone permission denied';
          });
          return;
        }
        
        await widget.audioService.start();
        _audioSub = widget.audioService.onAudioChunk.listen((chunk) {
          if (_isListening) {
            bool isSpeakingWithHangover = _isSpeaking || DateTime.now().difference(_lastSpeechTime).inMilliseconds < 1000;
            if (isSpeakingWithHangover) {
              try {
                 widget.engine.processAudioChunk(chunk);
              } catch (e, stack) {
                 globalLogger.logError('Exception during chunk forwarding: [${e.runtimeType}] $e', stack);
              }
            }
          }
        });
        
        _ampSub = widget.audioService.onAmplitude.listen((amp) {
           _currentRMS = amp.current;
           if (amp.current > _vadThreshold) { // Dynamic VAD Threshold
             if (!_isSpeaking) {
               _addLog('[SPEECH] detected');
               _lastLoggedPauseSecond = -1;
             }
             _isSpeaking = true;
             _lastSpeechTime = DateTime.now();
           } else {
             if (_isSpeaking) {
               _addLog('[SPEECH] ended');
               _addLog('[PAUSE] timer started');
             }
             _isSpeaking = false;
           }
        });

        _promptLoopTimer = Timer.periodic(const Duration(milliseconds: 100), (_) {
           _evaluatePromptState();
        });
        
        _lastProgressionTime = DateTime.now();

        setState(() {
          _isListening = true;
          _recognitionStatus = 'Listening & Reciting...';
        });
      }
    } catch (e, stack) {
      globalLogger.logError('Exception in _toggleListening: [${e.runtimeType}] $e', stack);
      setState(() {
        _recognitionStatus = 'Hardware/System Error';
        _isListening = false;
      });
    }
  }

  Future<void> _promptTaraweehConfig() async {
    int surah = 1;
    int ayah = 1;
    final result = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Start Taraweeh Mode'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                decoration: const InputDecoration(labelText: 'Starting Surah Number (1-114)'),
                keyboardType: TextInputType.number,
                onChanged: (val) => surah = int.tryParse(val) ?? 1,
              ),
              TextField(
                decoration: const InputDecoration(labelText: 'Starting Ayah Number'),
                keyboardType: TextInputType.number,
                onChanged: (val) => ayah = int.tryParse(val) ?? 1,
              ),
            ],
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
            ElevatedButton(
              onPressed: () {
                _lastSurah = surah;
                _lastAyah = ayah;
                Navigator.pop(context, true);
              },
              child: const Text('Start'),
            ),
          ],);
      }
    );

    if (result == true) {
      widget.engine.startTaraweeh(surah, ayah);
      setState(() {
        _taraweehModeEnabled = true;
        _currentSurah = surah;
        _currentAyah = ayah;
      });
      _addLog('Taraweeh Mode configured for $surah:$ayah');
      if (!_isListening) {
        _toggleListening();
      }
    }
  }

  void _openSettings() {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => SettingsScreen(
          engine: widget.engine,
          audioService: widget.audioService,
          onSaved: () async {
            _addLog('Settings updated. Reconnecting...');
            await widget.engine.disconnect();
            await _initEngine();
            await _loadSettings();
          },
        ),
      ),
    );
  }

  Widget _buildDebugScreen() {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Diagnostics & Debug', style: TextStyle(color: Colors.yellow)),
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
        leading: IconButton(
          icon: const Icon(Icons.close),
          onPressed: () => setState(() => _showDebugScreen = false),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text('--- STATE ---', style: TextStyle(color: Colors.yellow, fontWeight: FontWeight.bold, fontSize: 18)),
            const SizedBox(height: 8),
            Text('Recording: ${_isListening ? "YES" : "NO"}', style: const TextStyle(color: Colors.white, fontSize: 16)),
            Text('Speech Detected: ${_isSpeaking ? "TRUE" : "FALSE"}', style: const TextStyle(color: Colors.cyanAccent, fontSize: 16, fontWeight: FontWeight.bold)),
            Text('Audio RMS: ${_currentRMS.toStringAsFixed(2)} dB (Threshold: ${_vadThreshold.toStringAsFixed(1)})', style: const TextStyle(color: Colors.white, fontSize: 16)),
            Text('Tracking State: $_trackingMode', style: const TextStyle(color: Colors.white, fontSize: 16)),
            Text('Tracking Locked: ${_trackingMode.contains("LOCKED") || _trackingMode == "NORMAL"}', style: const TextStyle(color: Colors.white, fontSize: 16)),
            Text('Confidence: ${(_confidence * 100).toStringAsFixed(0)}%', style: const TextStyle(color: Colors.white, fontSize: 16)),
            const SizedBox(height: 16),
            const Text('--- PROMPT MACHINE ---', style: TextStyle(color: Colors.yellow, fontWeight: FontWeight.bold, fontSize: 18)),
            const SizedBox(height: 8),
            Text('Prompt State: ${_promptState.name}', style: const TextStyle(color: Colors.orangeAccent, fontSize: 16, fontWeight: FontWeight.bold)),
            Text('Prompt Message: $_promptStateMessage', style: const TextStyle(color: Colors.white70, fontSize: 14)),
            Text('Pause Timer: ${(_isListening ? DateTime.now().difference(_lastSpeechTime).inMilliseconds / 1000 : 0.0).toStringAsFixed(1)} s', style: const TextStyle(color: Colors.white, fontSize: 16)),
            Text('Anchor Surah: ${_currentSurah > 0 ? _currentSurah : _lastSurah}', style: const TextStyle(color: Colors.white, fontSize: 16)),
            Text('Anchor Ayah: ${_currentAyah > 0 ? _currentAyah : _lastAyah}', style: const TextStyle(color: Colors.white, fontSize: 16)),
            Text('Repeat Count: $_promptRepeatCount / $_promptMaxRepeats', style: const TextStyle(color: Colors.white, fontSize: 16)),
            const SizedBox(height: 24),
            const Text('--- DEVELOPER CONTROLS ---', style: TextStyle(color: Colors.yellow, fontWeight: FontWeight.bold, fontSize: 18)),
            const SizedBox(height: 8),
            ElevatedButton(
              style: ElevatedButton.styleFrom(backgroundColor: Colors.redAccent),
              onPressed: () {
                _setPromptState(PromptState.PROMPT_DISPLAYED, 'DEV FORCE TRIGGER');
                int anchorSurah = _currentSurah > 0 ? _currentSurah : _lastSurah;
                int anchorAyah = _currentAyah > 0 ? _currentAyah : _lastAyah;
                if (anchorSurah > 0 && anchorAyah > 0) {
                  setState(() => _assistedAyah = anchorAyah + 1);
                  widget.engine.sendAssistedPrompt(_assistedAyah);
                  _playPromptAudio(anchorSurah, anchorAyah + 1, forceRepeat: true);
                }
              },
              child: const Text('Trigger Prompt Now', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            ),
            const SizedBox(height: 8),
            ElevatedButton(
              style: ElevatedButton.styleFrom(backgroundColor: Colors.blueAccent),
              onPressed: () {
                setState(() {
                  _lastSurah = _currentSurah > 0 ? _currentSurah : 1;
                  _lastAyah = _currentAyah > 0 ? _currentAyah : 1;
                  _trackingMode = 'LOCKED (DEV)';
                  _confidence = 0.99;
                });
              },
              child: const Text('Force Lock Current Position', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            ),
            const SizedBox(height: 8),
            ElevatedButton(
              style: ElevatedButton.styleFrom(backgroundColor: Colors.grey),
              onPressed: () {
                setState(() {
                  _lastSurah = 0;
                  _lastAyah = 0;
                  _currentSurah = 0;
                  _currentAyah = 0;
                  _trackingMode = 'IDLE';
                  _confidence = 0.0;
                });
              },
              child: const Text('Reset Tracking', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            ),
            const SizedBox(height: 24),
            const Text('--- LOGS ---', style: TextStyle(color: Colors.yellow, fontWeight: FontWeight.bold, fontSize: 18)),
            const SizedBox(height: 8),
            Container(
              height: 300,
              color: Colors.grey[900],
              child: ListView.builder(
                padding: const EdgeInsets.all(8),
                itemCount: _logs.length,
                itemBuilder: (context, index) {
                  return Text(
                    _logs[_logs.length - 1 - index],
                    style: const TextStyle(color: Colors.greenAccent, fontFamily: 'monospace', fontSize: 12),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_showDebugScreen) {
      return _buildDebugScreen();
    }
    
    return Scaffold(
      appBar: AppBar(
        title: Text(_debugModeEnabled ? 'Live Recitation (Debug)' : 'Live Recitation'),
        actions: [
          if (_debugModeEnabled)
            IconButton(
              icon: const Icon(Icons.bug_report, color: Colors.orange),
              onPressed: () => setState(() => _showDebugScreen = true),
              tooltip: 'Open Debug Screen',
            ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: _openSettings,
            tooltip: 'Connection Settings',
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              child: Card(
                elevation: 2,
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      if (_debugModeEnabled) ...[
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text('Target URL:', style: TextStyle(fontWeight: FontWeight.bold)),
                            Expanded(
                              child: Text(
                                widget.engine.connectionUrl,
                                textAlign: TextAlign.end,
                                style: const TextStyle(color: Colors.blueGrey, fontFamily: 'monospace', fontSize: 11),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                      ],
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Icon(Icons.mic, color: _isListening ? Colors.red : Colors.grey, size: 20),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              _recognitionStatus,
                              style: TextStyle(color: _isListening ? Colors.red : Colors.grey),
                              overflow: TextOverflow.ellipsis,
                              textAlign: TextAlign.end,
                            ),
                          ),
                          if (_isMutashabihat)
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                              decoration: BoxDecoration(
                                color: Colors.orange.withOpacity(0.2),
                                borderRadius: BorderRadius.circular(12),
                                border: Border.all(color: Colors.orange),
                              ),
                              child: const Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(Icons.warning_amber_rounded, color: Colors.orange, size: 16),
                                  SizedBox(width: 4),
                                  Text("Similar Verse", style: TextStyle(color: Colors.orange, fontSize: 12, fontWeight: FontWeight.bold)),
                                ],
                              ),
                            ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text('Taraweeh Mode:', style: TextStyle(fontWeight: FontWeight.bold)),
                          Switch(
                            value: _taraweehModeEnabled,
                            onChanged: (val) {
                              if (val) {
                                _promptTaraweehConfig();
                              } else {
                                setState(() {
                                  _taraweehModeEnabled = false;
                                });
                              }
                            },
                          )
                        ],
                      ),
                      const SizedBox(height: 8),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text('Prompt Mode:', style: TextStyle(fontWeight: FontWeight.bold)),
                          Switch(
                            value: _promptModeEnabled,
                            onChanged: (val) {
                              setState(() {
                                _promptModeEnabled = val;
                                if (val) _loadSettings();
                              });
                            },
                          )
                        ],
                      ),
                      const SizedBox(height: 8),
                      ElevatedButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(builder: (context) => const OpenSourceAcknowledgementsScreen())
                          );
                        },
                        child: const Text('Open Source Acknowledgements'),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            Expanded(
              child: Center(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24.0),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      if (_isPromptConfirmed) ...[
                        Container(
                          padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 24),
                          decoration: BoxDecoration(
                            color: Colors.green,
                            borderRadius: BorderRadius.circular(30),
                          ),
                          child: const Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.check_circle, color: Colors.white, size: 28),
                              SizedBox(width: 8),
                              Text(
                                'Recovery Successful',
                                style: TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 24),
                      ],
                      if (_currentSurah > 0) ...[
                        Text(
                          'Surah: $_surahNameEn',
                          style: const TextStyle(fontSize: 26, fontWeight: FontWeight.bold, color: Colors.blue),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 4),
                        Text(
                          '($_surahNameAr)',
                          style: const TextStyle(fontSize: 20, color: Colors.grey),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'Ayah: $_currentAyah',
                          style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w500),
                          textAlign: TextAlign.center,
                        ),
                        if (_totalWords > 0) ...[
                          const SizedBox(height: 8),
                          Text(
                            'Word Position: $_wordPosition / $_totalWords',
                            style: const TextStyle(fontSize: 20, color: Colors.blueGrey, fontWeight: FontWeight.bold),
                            textAlign: TextAlign.center,
                          ),
                        ],
                        if (_debugModeEnabled) ...[
                          const SizedBox(height: 8),
                          Text(
                            'Confidence: ${(_confidence * 100).toStringAsFixed(0)}%',
                            style: const TextStyle(fontSize: 18, color: Colors.green, fontWeight: FontWeight.bold),
                            textAlign: TextAlign.center,
                          ),
                          if (_transcript.isNotEmpty) ...[
                            const SizedBox(height: 16),
                            Text(
                              'Transcript: "$_transcript"',
                              style: const TextStyle(fontSize: 14, fontStyle: FontStyle.italic, color: Colors.black54),
                              textAlign: TextAlign.center,
                            ),
                          ],
                        ],
                      ] else ...[
                        const Text(
                          'No verse detected yet.',
                          style: TextStyle(fontSize: 18, color: Colors.black45),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),
            if (_debugModeEnabled && _trackingMode.isNotEmpty) ...[
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.grey[200],
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.grey[400]!)
                  ),
                  child: Column(
                    children: [
                      Text('Tracking: $_trackingMode', style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
                      Text('Search Window: $_searchWindow', style: const TextStyle(fontSize: 12, color: Colors.blueGrey)),
                      Text('Fallback Count: $_fallbackCount', style: const TextStyle(fontSize: 12, color: Colors.blueGrey)),
                      if (_promptModeEnabled)
                         Text('Prompt State: ${_promptState.name}', style: const TextStyle(fontSize: 12, color: Colors.deepPurple, fontWeight: FontWeight.bold)),
                      if (_assistedAyah > 0)
                         Text('Prompted Ayah: $_assistedAyah', style: const TextStyle(fontSize: 12, color: Colors.orange, fontWeight: FontWeight.bold)),
                      if (_promptRepeatCount > 0)
                         Text('Repeat Count: $_promptRepeatCount/$_promptMaxRepeats', style: const TextStyle(fontSize: 12, color: Colors.orange)),
                      Text('Current Ayah: $_currentAyah | Expected Next: ${_currentAyah + 1}', style: const TextStyle(fontSize: 12, color: Colors.blueGrey)),
                    ]
                  )
                ),
              ),
            ],
            const SizedBox(height: 10),
            if (_currentAyahText.isNotEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                    boxShadow: const [
                      BoxShadow(color: Colors.black12, blurRadius: 4, offset: Offset(0, 2))
                    ],
                  ),
                  child: Column(
                    children: [
                      Text(
                        '$_surahNameEn:$_currentAyah',
                        style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.blueGrey),
                      ),
                      const SizedBox(height: 10),
                      _buildCurrentAyahText(),
                    ],
                  ),
                ),
              ),
            if (_promptModeEnabled && _promptState != PromptState.DISABLED && _promptState != PromptState.WAITING_FOR_LOCK && _promptState != PromptState.PROMPT_READY && _promptState != PromptState.IDLE)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.amber[100],
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.orange, width: 2),
                  ),
                  child: Column(
                    children: [
                      Text(
                        _promptState == PromptState.PROMPT_DISPLAYED ? 'Need Help? Next Word:' : 'Need Help?',
                        style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.orange, fontSize: 16),
                      ),
                      const SizedBox(height: 10),
                      if (_promptState == PromptState.PROMPT_DISPLAYED) ...[
                         Text(
                           _getNextWordHint(),
                           textAlign: TextAlign.center,
                           textDirection: TextDirection.rtl,
                           style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, fontFamily: 'Amiri'),
                         )
                      ] else if (_promptState == PromptState.PAUSE_TIMER_RUNNING) ...[
                         Text(
                           _promptStateMessage,
                           textAlign: TextAlign.center,
                           style: const TextStyle(fontSize: 18, color: Colors.black54),
                         )
                      ] else if (_promptState == PromptState.PLAYING_AUDIO || _promptState == PromptState.PROMPT_REPEAT) ...[
                         const Icon(Icons.volume_up, color: Colors.orange, size: 48),
                         const SizedBox(height: 8),
                         Text(
                           _nextAyahText.isNotEmpty ? _nextAyahText : _currentAyahText,
                           textAlign: TextAlign.center,
                           textDirection: TextDirection.rtl,
                           style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, fontFamily: 'Amiri'),
                         )
                      ]
                    ],
                  ),
                ),
              ),
            const SizedBox(height: 10),
            FloatingActionButton(
              onPressed: !widget.engine.isReady ? null : _toggleListening,
              backgroundColor: _isListening ? Colors.red : (widget.engine.isReady ? Colors.blue : Colors.grey),
              child: Icon(_isListening ? Icons.stop : Icons.mic),
            ),
            const SizedBox(height: 20),
            if (_debugModeEnabled) ...[
              const Divider(),
              const Text('Debug Logs', style: TextStyle(fontWeight: FontWeight.bold)),
              SizedBox(
                height: 200,
                child: ListView.builder(
                  padding: const EdgeInsets.all(8.0),
                  itemCount: _logs.length,
                  itemBuilder: (context, index) {
                    return Text(
                      _logs[index],
                      style: TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 12,
                        color: _logs[index].contains('[ERROR]') ? Colors.red : Colors.black87,
                      ),
                    );
                  },
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  String _getNextWordHint() {
    if (_currentAyahText.isEmpty && _nextAyahText.isEmpty) return "";
    if (_wordPosition < _totalWords && _currentAyahText.isNotEmpty) {
       List<String> words = _currentAyahText.split(' ');
       if (_wordPosition < words.length) {
          return words[_wordPosition]; // next word in current ayah
       }
    }
    if (_nextAyahText.isNotEmpty) {
       List<String> words = _nextAyahText.split(' ');
       if (words.isNotEmpty) return words[0];
    }
    return "";
  }

  Widget _buildCurrentAyahText() {
    if (_currentAyahText.isEmpty) return const SizedBox();
    
    List<String> words = _currentAyahText.split(' ');
    return Wrap(
      alignment: WrapAlignment.center,
      textDirection: TextDirection.rtl,
      spacing: 8.0,
      runSpacing: 4.0,
      children: List.generate(words.length, (index) {
        int targetIndex = _wordPosition - 1;
        if (targetIndex >= words.length) targetIndex = words.length - 1;
        if (targetIndex < 0) targetIndex = 0;

        Color color = Colors.black87;
        FontWeight weight = FontWeight.normal;
        
        if (_wordPosition == 0) {
           // Not started
           color = Colors.black87;
        } else if (index < targetIndex) {
          color = Colors.green; // Spoken
        } else if (index == targetIndex) {
          color = Colors.blue; // Active Word
          weight = FontWeight.bold;
        }

        return Text(
          words[index],
          textDirection: TextDirection.rtl,
          style: TextStyle(
            fontSize: 28,
            fontFamily: 'Amiri',
            color: color,
            fontWeight: weight,
          ),
        );
      }),
    );
  }
}
