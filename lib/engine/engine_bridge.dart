import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';
import '../core/global_logger.dart';
import 'recitation_engine.dart';

class TarteelEngineBridge implements RecitationEngine {
  HeadlessInAppWebView? _headlessWebView;
  final InAppLocalhostServer _localhostServer = InAppLocalhostServer(
    port: 8080,
    documentRoot: 'assets',
  );

  final _eventsController = StreamController<Map<String, dynamic>>.broadcast();
  Stream<Map<String, dynamic>> get onEvent => _eventsController.stream;

  final List<Map<String, dynamic>> _eventHistory = [];
  List<Map<String, dynamic>> get eventHistory =>
      List.unmodifiable(_eventHistory);

  bool _isReady = false;
  bool _isDisposed = false;
  bool _isServerStarted = false;
  bool get isReady => _isReady;
  @override
  bool get supportsRemoteBackend => false;
  @override
  String get serverIp => 'standalone';
  @override
  String get serverPort => 'local';

  List<Map<String, dynamic>> _quran = [];
  final Map<int, List<Map<String, dynamic>>> _bySurah = {};
  bool _isTaraweehActive = false;
  int _taraweehSurah = 1;
  int _taraweehAyah = 1;
  int _currentSurah = 0;
  int _currentAyah = 0;
  double _wordCoverage = 0.0;
  String? _pendingMistakeKey;
  int _pendingMistakeCount = 0;
  String? _pendingTaraweehCandidateKey;
  int _pendingTaraweehCandidateCount = 0;
  DateTime _mistakeCooldownUntil = DateTime.fromMillisecondsSinceEpoch(0);
  final List<List<int>> _pendingAudioChunks = [];
  int _pendingAudioBytes = 0;
  final BytesBuilder _audioBatch = BytesBuilder(copy: false);
  Timer? _audioFlushTimer;
  Timer? _startupWatchdogTimer;
  static const int _maxPendingAudioBytes = 16000 * 2 * 45; // 45 seconds.
  static const int _audioFlushIntervalMs = 250;
  static const int _maxBatchBytes = 32000; // 1 second of 16 kHz PCM16 mono.

  void _emitEvent(Map<String, dynamic> event) {
    _eventHistory.add(event);
    if (_eventHistory.length > 200) {
      _eventHistory.removeAt(0);
    }
    _eventsController.add(event);
  }

  void _handleEngineEvent(Map<String, dynamic> event) {
    final type = event['type'];
    if (type == 'verse_candidate' && _handleTaraweehCandidate(event)) {
      return;
    }
    if (_isTaraweehEventBlocked(event)) {
      return;
    }
    if (type == 'word_progress') {
      final surah = (event['surah'] as num?)?.toInt();
      final ayah = (event['ayah'] as num?)?.toInt();
      final wordIndex = (event['word_index'] as num?)?.toInt() ?? 0;
      final totalWords = (event['total_words'] as num?)?.toInt() ?? 0;
      if (surah != null && ayah != null) {
        _currentSurah = surah;
        _currentAyah = ayah;
      }
      _wordCoverage = totalWords > 0 ? wordIndex / totalWords : _wordCoverage;
      event['word_coverage'] = _wordCoverage;
    } else if (type == 'verse_match') {
      final surah = (event['surah'] as num?)?.toInt();
      final ayah = (event['ayah'] as num?)?.toInt();
      if (surah != null && ayah != null) {
        _currentSurah = surah;
        _currentAyah = ayah;
        _wordCoverage = 0.0;
      }
      _pendingTaraweehCandidateKey = null;
      _pendingTaraweehCandidateCount = 0;
    }
    _emitEvent(event);
  }

  void _flushPendingAudioChunks() {
    if (!_isReady || _headlessWebView == null || _pendingAudioChunks.isEmpty) {
      return;
    }
    final chunks = List<List<int>>.from(_pendingAudioChunks);
    _pendingAudioChunks.clear();
    _pendingAudioBytes = 0;
    _emitEvent({
      'type': 'status',
      'message':
          'Standalone engine ready; replaying ${chunks.length} buffered audio chunks.',
    });
    for (final chunk in chunks) {
      processAudioChunk(chunk);
    }
  }

  void _scheduleAudioFlush() {
    if (_audioFlushTimer?.isActive ?? false) return;
    _audioFlushTimer = Timer(
      const Duration(milliseconds: _audioFlushIntervalMs),
      _flushAudioBatch,
    );
  }

  void _flushAudioBatch() {
    _audioFlushTimer?.cancel();
    _audioFlushTimer = null;
    if (_isDisposed || !_isReady || _headlessWebView == null) {
      return;
    }
    if (_audioBatch.length == 0) return;

    final pcmBytes = _audioBatch.takeBytes();
    final floatBytes = _pcm16ToFloat32Bytes(pcmBytes);
    if (floatBytes.isEmpty) return;
    final base64Data = base64Encode(floatBytes);
    unawaited(
      _headlessWebView?.webViewController
          ?.evaluateJavascript(
            source: 'window.processAudioChunk("$base64Data");',
          )
          .catchError((Object error, StackTrace stack) {
            globalLogger.logError(
              'Failed to forward audio to standalone engine: $error',
              stack,
            );
          }),
    );
  }

  bool _isTaraweehEventBlocked(Map<String, dynamic> event) {
    if (!_isTaraweehActive) return false;
    final type = event['type'];
    if (type != 'verse_match' && type != 'word_progress') return false;

    final surah = (event['surah'] as num?)?.toInt();
    final ayah = (event['ayah'] as num?)?.toInt();
    if (surah == null || ayah == null) return false;

    final activeSurah = _currentSurah > 0 ? _currentSurah : _taraweehSurah;
    final activeAyah = _currentAyah > 0 ? _currentAyah : _taraweehAyah;
    final confidence = ((event['confidence'] as num?)?.toDouble() ?? 0.0);
    final lastAyah = _lastAyahInSurah(activeSurah);
    final allowedTransition =
        surah == activeSurah + 1 &&
        activeAyah >= lastAyah &&
        ayah <= 2 &&
        confidence >= 0.82;

    if (surah != activeSurah && !allowedTransition) {
      _handleSuppressedMistake(
        activeSurah,
        activeAyah,
        surah,
        ayah,
        confidence,
      );
      _emitEvent({
        'type': 'status',
        'message':
            'Suppressed cross-surah prediction $surah:$ayah during Taraweeh.',
        'tracking_mode': 'TARAWIH_SURAH_LOCK',
        'search_window': 'Surah $activeSurah only',
      });
      return true;
    }

    final forwardSections = _forwardSectionDistance(
      activeSurah,
      activeAyah,
      ayah,
    );
    if (surah == activeSurah && forwardSections > 8 && confidence < 0.92) {
      _emitEvent({
        'type': 'status',
        'message':
            'Confirming forward jump to Ayah $ayah ($forwardSections sections ahead)...',
        'tracking_mode': 'TARAWIH_FORWARD_JUMP_CHECK',
        'search_window': 'Surah $activeSurah only',
      });
      return true;
    }

    _pendingMistakeKey = null;
    _pendingMistakeCount = 0;
    return false;
  }

  void _handleSuppressedMistake(
    int activeSurah,
    int activeAyah,
    int detectedSurah,
    int detectedAyah,
    double confidence,
  ) {
    final hasConfirmedProgress = _wordCoverage >= 0.08;
    final minConfidence = hasConfirmedProgress ? 0.90 : 0.94;
    if (confidence < minConfidence ||
        DateTime.now().isBefore(_mistakeCooldownUntil)) {
      return;
    }
    final key = '$activeSurah:$activeAyah->$detectedSurah:$detectedAyah';
    if (_pendingMistakeKey == key) {
      _pendingMistakeCount++;
    } else {
      _pendingMistakeKey = key;
      _pendingMistakeCount = 1;
    }
    final requiredRepeats = hasConfirmedProgress ? 3 : 4;
    if (_pendingMistakeCount < requiredRepeats) return;

    final expectedAyah = _wordCoverage >= 0.90 ? activeAyah + 1 : activeAyah;
    final expected = _normalizeAyahPosition(activeSurah, expectedAyah);
    _emitEvent({
      'type': 'mistake_detected',
      'expected_surah': expected.$1,
      'expected_ayah': expected.$2,
      'detected_surah': detectedSurah,
      'detected_ayah': detectedAyah,
      'score': confidence,
    });
    _pendingMistakeKey = null;
    _pendingMistakeCount = 0;
    _mistakeCooldownUntil = DateTime.now().add(const Duration(seconds: 3));
  }

  bool _handleTaraweehCandidate(Map<String, dynamic> event) {
    if (!_isTaraweehActive) return false;
    final candidates = event['candidates'];
    if (candidates is! List || candidates.isEmpty) return false;

    final activeSurah = _currentSurah > 0 ? _currentSurah : _taraweehSurah;
    final activeAyah = _currentAyah > 0 ? _currentAyah : _taraweehAyah;
    Map<String, dynamic>? selected;

    for (final rawCandidate in candidates) {
      if (rawCandidate is! Map) continue;
      final candidate = Map<String, dynamic>.from(rawCandidate);
      final surah = (candidate['surah'] as num?)?.toInt();
      final ayah = (candidate['ayah'] as num?)?.toInt();
      final confidence = (candidate['confidence'] as num?)?.toDouble() ?? 0.0;
      if (surah != activeSurah || ayah == null || confidence < 0.50) {
        continue;
      }

      final forwardSections = _forwardSectionDistance(
        activeSurah,
        activeAyah,
        ayah,
      );
      final isNearbyRewind = ayah < activeAyah && activeAyah - ayah <= 2;
      final isCurrentOrNext = ayah == activeAyah || ayah == activeAyah + 1;
      final isPatientForward = ayah > activeAyah && forwardSections <= 8;
      if (!isNearbyRewind && !isCurrentOrNext && !isPatientForward) {
        continue;
      }

      selected = candidate;
      break;
    }

    if (selected == null) return false;

    final surah = (selected['surah'] as num).toInt();
    final ayah = (selected['ayah'] as num).toInt();
    final confidence = (selected['confidence'] as num?)?.toDouble() ?? 0.0;
    final key = '$surah:$ayah';
    if (_pendingTaraweehCandidateKey == key) {
      _pendingTaraweehCandidateCount++;
    } else {
      _pendingTaraweehCandidateKey = key;
      _pendingTaraweehCandidateCount = 1;
    }

    final stable = event['stable'] == true || event['final_flush'] == true;
    final isForward = ayah > activeAyah;
    final isNextAyah = ayah == activeAyah + 1;
    final minConfidence = isForward ? (isNextAyah ? 0.58 : 0.72) : 0.52;
    final strongNextAyah = isNextAyah && confidence >= 0.76;
    final requiredRepeats = stable || strongNextAyah
        ? 1
        : (isForward && _wordCoverage < 0.45 ? 2 : 2);

    if (confidence < minConfidence ||
        _pendingTaraweehCandidateCount < requiredRepeats) {
      _emitEvent({
        'type': 'status',
        'message':
            'Considering Taraweeh candidate $surah:$ayah (${(confidence * 100).toStringAsFixed(0)}%).',
        'tracking_mode': 'TARAWIH_CANDIDATE_ASSIST',
        'search_window': 'Surah $activeSurah only',
      });
      return true;
    }

    final verseEvent = _buildVerseMatchEvent(
      surah,
      ayah,
      confidence,
      trackingMode: 'TARAWIH_CANDIDATE_ASSIST',
    );
    if (verseEvent == null) return false;

    _currentSurah = surah;
    _currentAyah = ayah;
    _wordCoverage = 0.0;
    _pendingMistakeKey = null;
    _pendingMistakeCount = 0;
    _pendingTaraweehCandidateKey = null;
    _pendingTaraweehCandidateCount = 0;
    _emitEvent({
      'type': 'status',
      'message':
          'Accepted Taraweeh candidate $surah:$ayah (${(confidence * 100).toStringAsFixed(0)}%).',
      'tracking_mode': 'TARAWIH_CANDIDATE_ASSIST',
      'search_window': 'Surah $activeSurah only',
    });
    _emitEvent(verseEvent);
    return true;
  }

  Future<void> initialize() async {
    if (_isReady) return;
    await _loadQuranData();
    if (_headlessWebView != null) {
      if (_startupWatchdogTimer?.isActive ?? false) {
        _emitEvent({
          'type': 'status',
          'message': 'On-device recognition is still loading...',
        });
        return;
      }
      _emitEvent({
        'type': 'status',
        'message': 'Restarting stale on-device recognition engine...',
      });
      try {
        await _headlessWebView?.dispose();
      } catch (e, stack) {
        globalLogger.logError('Failed to dispose stale WebView: $e', stack);
      }
      _headlessWebView = null;
      _startupWatchdogTimer?.cancel();
      _startupWatchdogTimer = null;
      _isReady = false;
    }
    try {
      if (!_isServerStarted) {
        _emitEvent({'type': 'status', 'message': 'Starting local server...'});
        await _localhostServer.start();
        _isServerStarted = true;
        _emitEvent({
          'type': 'status',
          'message': 'Local server started successfully.',
        });
      }
    } catch (e, stack) {
      _emitEvent({
        'type': 'error',
        'message': 'Failed to start local server: $e\nStacktrace: $stack',
      });
      return;
    }

    try {
      _emitEvent({'type': 'status', 'message': 'Starting Headless WebView...'});
      _headlessWebView = HeadlessInAppWebView(
        initialUrlRequest: URLRequest(
          url: WebUri("http://localhost:8080/web/index.html"),
        ),
        initialSettings: InAppWebViewSettings(
          javaScriptEnabled: true,
          allowFileAccessFromFileURLs: true,
          allowUniversalAccessFromFileURLs: true,
        ),
        onWebViewCreated: (controller) {
          _emitEvent({
            'type': 'status',
            'message': 'WebView created, adding JS handlers...',
          });
          controller.addJavaScriptHandler(
            handlerName: 'onEngineEvent',
            callback: (args) {
              if (args.isNotEmpty) {
                final rawEvent = args[0];
                if (rawEvent is! Map) return;
                final event = Map<String, dynamic>.from(rawEvent);
                if (event['type'] == 'ready') {
                  _isReady = true;
                  _startupWatchdogTimer?.cancel();
                  _startupWatchdogTimer = null;
                  _flushPendingAudioChunks();
                  _syncTaraweehLockToWebView();
                }
                _handleEngineEvent(event);
              }
            },
          );
        },
        onLoadStart: (controller, url) {
          _emitEvent({
            'type': 'status',
            'message': 'WebView load started: $url',
          });
        },
        onLoadStop: (controller, url) async {
          _emitEvent({
            'type': 'status',
            'message': 'WebView load stopped: $url',
          });
          unawaited(
            controller
                .evaluateJavascript(
                  source: '''
                    (function bootStandaloneEngine() {
                      if (window._engineInitialized) return;
                      var attempts = 0;
                      function boot() {
                        if (window._engineInitialized) return;
                        if (window.initEngine) {
                          window._engineInitialized = true;
                          window.initEngine("http://localhost:8080/web");
                          return;
                        }
                        attempts++;
                        if (attempts <= 120) {
                          setTimeout(boot, 500);
                        } else {
                          console.error("window.initEngine is undefined after waiting for engine bundle startup.");
                        }
                      }
                      boot();
                    })();
                  ''',
                )
                .catchError((Object error, StackTrace stack) {
                  globalLogger.logError(
                    'Failed to start standalone JS engine: $error',
                    stack,
                  );
                }),
          );
        },
        onConsoleMessage: (controller, consoleMessage) {
          if (kDebugMode) {
            print(
              '[WebEngine] ${consoleMessage.messageLevel}: ${consoleMessage.message}',
            );
          }
          final msg = consoleMessage.message;
          if (consoleMessage.messageLevel == ConsoleMessageLevel.ERROR) {
            _emitEvent({'type': 'error', 'message': 'JS Error: $msg'});
          } else {
            _emitEvent({'type': 'status', 'message': 'JS Log: $msg'});
          }
        },
        onReceivedError: (controller, request, error) {
          _emitEvent({
            'type': 'error',
            'message': 'WebView Error: ${error.description}',
          });
        },
        onReceivedHttpError: (controller, request, errorResponse) {
          _emitEvent({
            'type': 'error',
            'message':
                'HTTP Error: ${request.url} - ${errorResponse.statusCode}',
          });
        },
      );

      await _headlessWebView?.run();
      _emitEvent({'type': 'status', 'message': 'Headless WebView running.'});
      _startupWatchdogTimer?.cancel();
      _startupWatchdogTimer = Timer(const Duration(seconds: 45), () {
        if (_isDisposed || _isReady) return;
        _startupWatchdogTimer = null;
        _emitEvent({
          'type': 'status',
          'message':
              'On-device recognition is still loading. Keep the mic open or switch to PC backend for this test.',
        });
      });
    } catch (e, stack) {
      _emitEvent({
        'type': 'error',
        'message': 'Failed to launch Headless WebView: $e\nStacktrace: $stack',
      });
    }
  }

  Future<void> _loadQuranData() async {
    if (_quran.isNotEmpty) return;
    try {
      final raw = await rootBundle.loadString('assets/web/quran_phonemes.json');
      final decoded = jsonDecode(raw);
      if (decoded is! List) return;
      _quran = decoded.cast<Map<String, dynamic>>();
      _bySurah.clear();
      for (final verse in _quran) {
        final surah = (verse['surah'] as num?)?.toInt();
        if (surah == null) continue;
        _bySurah.putIfAbsent(surah, () => []).add(verse);
      }
      for (final verses in _bySurah.values) {
        verses.sort(
          (a, b) => ((a['ayah'] as num?)?.toInt() ?? 0).compareTo(
            (b['ayah'] as num?)?.toInt() ?? 0,
          ),
        );
      }
    } catch (e, stack) {
      globalLogger.logError('Failed to load bundled Quran data: $e', stack);
    }
  }

  (int, int) _normalizeAyahPosition(int surah, int ayah) {
    var normalizedSurah = surah.clamp(1, 114).toInt();
    var normalizedAyah = ayah < 1 ? 1 : ayah;
    while (normalizedSurah < 114) {
      final verses = _bySurah[normalizedSurah] ?? const [];
      final lastAyah = verses.isNotEmpty
          ? ((verses.last['ayah'] as num?)?.toInt() ?? 1)
          : 1;
      if (normalizedAyah <= lastAyah) break;
      normalizedAyah -= lastAyah;
      normalizedSurah++;
    }
    final verses = _bySurah[normalizedSurah] ?? const [];
    if (verses.isNotEmpty) {
      final lastAyah = (verses.last['ayah'] as num?)?.toInt() ?? normalizedAyah;
      if (normalizedAyah > lastAyah) normalizedAyah = lastAyah;
    }
    return (normalizedSurah, normalizedAyah);
  }

  Map<String, dynamic>? _getVerse(int surah, int ayah) {
    for (final verse in _bySurah[surah] ?? const []) {
      if ((verse['ayah'] as num?)?.toInt() == ayah) return verse;
    }
    return null;
  }

  int _lastAyahInSurah(int surah) {
    final verses = _bySurah[surah] ?? const [];
    if (verses.isEmpty) return 1;
    return (verses.last['ayah'] as num?)?.toInt() ?? 1;
  }

  int _targetSectionCount(int totalWords) {
    if (totalWords <= 0) return 0;
    if (totalWords <= 5) return 2;
    if (totalWords <= 11) return 3;
    if (totalWords <= 22) return 4;
    if (totalWords <= 32) return 5;
    if (totalWords <= 45) return 6;
    if (totalWords <= 70) return 8;
    return (totalWords / 9).ceil().clamp(9, 18).toInt();
  }

  int _verseSectionCount(int surah, int ayah) {
    final verse = _getVerse(surah, ayah);
    final text = verse?['text_uthmani']?.toString() ?? '';
    final totalWords = text
        .split(RegExp(r'\s+'))
        .where((w) => w.trim().isNotEmpty)
        .length;
    return _targetSectionCount(totalWords);
  }

  int _forwardSectionDistance(int surah, int activeAyah, int matchedAyah) {
    if (surah <= 0 || matchedAyah <= activeAyah) return 0;
    var distance = 0;
    final currentTotal = _verseSectionCount(surah, activeAyah);
    final completed = (_wordCoverage.clamp(0.0, 1.0) * currentTotal).round();
    distance += (currentTotal - completed).clamp(0, currentTotal).toInt();
    for (var ayah = activeAyah + 1; ayah < matchedAyah; ayah++) {
      distance += _verseSectionCount(surah, ayah);
    }
    distance += 1;
    return distance;
  }

  Map<String, dynamic>? _buildVerseMatchEvent(
    int surah,
    int ayah,
    double confidence, {
    String trackingMode = '',
  }) {
    final verse = _getVerse(surah, ayah);
    if (verse == null) return null;
    final previous = ayah > 1 ? _getVerse(surah, ayah - 1) : null;
    final next = _getVerse(surah, ayah + 1);
    final text = verse['text_uthmani']?.toString() ?? '';
    final totalWords = text
        .split(RegExp(r'\s+'))
        .where((word) => word.trim().isNotEmpty)
        .length;
    return {
      'type': 'verse_match',
      'surah': surah,
      'ayah': ayah,
      'verse_text': text,
      'current_ayah_text': text,
      'next_ayah_text': next?['text_uthmani'] ?? '',
      'prev_ayah_text': previous?['text_uthmani'] ?? '',
      'surah_name': verse['surah_name'] ?? '',
      'surah_name_en': verse['surah_name_en'] ?? '',
      'confidence': double.parse(confidence.clamp(0.0, 1.0).toStringAsFixed(2)),
      'tracking_mode': trackingMode,
      'search_window': 'Surah $surah only',
      'word_index': 0,
      'total_words': totalWords,
      'section_index': 0,
      'total_sections': _verseSectionCount(surah, ayah),
      'section_coverage': 0.0,
    };
  }

  List<Map<String, dynamic>> _buildLuqmahVerses(
    int surah,
    int ayah,
    int count,
  ) {
    final normalized = _normalizeAyahPosition(surah, ayah);
    final selected = <Map<String, dynamic>>[];
    final seen = <String>{};
    final safeCount = count.clamp(1, 3).toInt();
    for (var offset = 0; offset < safeCount; offset++) {
      final position = _normalizeAyahPosition(
        normalized.$1,
        normalized.$2 + offset,
      );
      final key = '${position.$1}:${position.$2}';
      if (!seen.add(key)) break;
      final verse = _getVerse(position.$1, position.$2);
      if (verse == null) break;
      selected.add({
        'surah': position.$1,
        'ayah': position.$2,
        'ayah_text': verse['text_uthmani'] ?? '',
        'surah_name': verse['surah_name'] ?? '',
        'surah_name_en': verse['surah_name_en'] ?? '',
      });
    }
    return selected;
  }

  @override
  Future<void> saveSettings(String ip, String port) async {}

  @override
  Future<bool> testConnection(String ip, String port) async => _isReady;

  @override
  void processAudioChunk(List<int> pcm16Bytes) {
    if (_isDisposed) return;
    if (!_isReady || _headlessWebView == null) {
      final copy = List<int>.from(pcm16Bytes);
      _pendingAudioChunks.add(copy);
      _pendingAudioBytes += copy.length;
      while (_pendingAudioBytes > _maxPendingAudioBytes &&
          _pendingAudioChunks.isNotEmpty) {
        final removed = _pendingAudioChunks.removeAt(0);
        _pendingAudioBytes -= removed.length;
      }
      return;
    }
    _audioBatch.add(pcm16Bytes);
    if (_audioBatch.length >= _maxBatchBytes) {
      _flushAudioBatch();
    } else {
      _scheduleAudioFlush();
    }
  }

  Uint8List _pcm16ToFloat32Bytes(List<int> pcm16Bytes) {
    final input = Uint8List.fromList(pcm16Bytes);
    if (input.lengthInBytes < 2) return Uint8List(0);
    final sampleCount = input.lengthInBytes ~/ 2;
    final floats = Float32List(sampleCount);
    final bytes = ByteData.sublistView(input);
    for (var i = 0; i < sampleCount; i++) {
      floats[i] = bytes.getInt16(i * 2, Endian.little) / 32768.0;
    }
    return Uint8List.view(floats.buffer);
  }

  void _syncTaraweehLockToWebView() {
    if (!_isReady || _headlessWebView == null) return;
    final source = _isTaraweehActive
        ? 'window.startTaraweeh && window.startTaraweeh($_taraweehSurah, $_taraweehAyah);'
        : 'window.stopTaraweeh && window.stopTaraweeh();';
    unawaited(
      _headlessWebView?.webViewController
          ?.evaluateJavascript(source: source)
          .catchError((Object error, StackTrace stack) {
            globalLogger.logError(
              'Failed to sync Taraweeh lock to standalone engine: $error',
              stack,
            );
          }),
    );
  }

  @override
  void startTaraweeh(int surah, int ayah) {
    final normalized = _normalizeAyahPosition(surah, ayah);
    _isTaraweehActive = true;
    _taraweehSurah = normalized.$1;
    _taraweehAyah = normalized.$2;
    _currentSurah = normalized.$1;
    _currentAyah = normalized.$2;
    _wordCoverage = 0.0;
    _pendingMistakeKey = null;
    _pendingMistakeCount = 0;
    final verse = _getVerse(_taraweehSurah, _taraweehAyah);
    if (verse != null) {
      _emitEvent({
        'type': 'assisted_verse_text',
        'ayah_text': verse['text_uthmani'] ?? '',
        'surah': _taraweehSurah,
        'ayah': _taraweehAyah,
        'surah_name': verse['surah_name'] ?? '',
        'surah_name_en': verse['surah_name_en'] ?? '',
      });
    }
    _syncTaraweehLockToWebView();
  }

  @override
  void stopTaraweeh() {
    _isTaraweehActive = false;
    _pendingMistakeKey = null;
    _pendingMistakeCount = 0;
    _pendingTaraweehCandidateKey = null;
    _pendingTaraweehCandidateCount = 0;
    _syncTaraweehLockToWebView();
    discardPendingAudio();
  }

  @override
  void sendAssistedPrompt(
    int surah,
    int ayah, {
    int count = 1,
    int wordIndex = 0,
  }) {
    if (_isReady) {
      discardPendingAudio();
    }
    final promptVerses = _buildLuqmahVerses(surah, ayah, count);
    if (promptVerses.isEmpty) return;
    final first = promptVerses.first;
    _emitEvent({
      'type': 'assisted_verse_text',
      'ayah_text': promptVerses.map((v) => v['ayah_text']).join('\n'),
      'surah': first['surah'],
      'ayah': first['ayah'],
      'surah_name': first['surah_name'] ?? '',
      'surah_name_en': first['surah_name_en'] ?? '',
      'prompt_verses': promptVerses,
    });
  }

  @override
  void discardPendingAudio() {
    _pendingAudioChunks.clear();
    _pendingAudioBytes = 0;
    _audioFlushTimer?.cancel();
    _audioFlushTimer = null;
    if (_audioBatch.length > 0) {
      _audioBatch.takeBytes();
    }
    unawaited(
      _headlessWebView?.webViewController?.evaluateJavascript(
        source: 'window.resetAudioStream && window.resetAudioStream();',
      ),
    );
  }

  @override
  void clearAssistedPrompt() {}

  @override
  Future<void> disconnect() async {
    discardPendingAudio();
  }

  @override
  Future<void> dispose() async {
    _isDisposed = true;
    _audioFlushTimer?.cancel();
    _startupWatchdogTimer?.cancel();
    await _headlessWebView?.dispose();
    await _localhostServer.close();
    _isServerStarted = false;
    await _eventsController.close();
  }
}
