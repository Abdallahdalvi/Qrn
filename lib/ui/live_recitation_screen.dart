import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import 'package:record/record.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../engine/recitation_engine.dart';
import '../audio/audio_capture.dart';
import '../audio/luqmah_reciters.dart';
import '../core/global_logger.dart';
import 'settings_screen.dart';

class LiveRecitationScreen extends StatefulWidget {
  final RecitationEngine engine;
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
  PROMPT_COOLDOWN,
  RECOVERY_MODE,
}

class _LiveRecitationScreenState extends State<LiveRecitationScreen>
    with TickerProviderStateMixin, WidgetsBindingObserver {
  bool _isListening = false;
  bool _isStarting = false; // Guard against double-start race condition
  StreamSubscription<Uint8List>? _audioSub;
  StreamSubscription<AudioCaptureState>? _captureStateSub;
  StreamSubscription<PlayerState>? _playerStateSub;
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
  double _wordCoverage = 0.0;
  int _sectionPosition = 0;
  int _totalSections = 0;
  double _sectionCoverage = 0.0;
  double _progressCoverage = 0.0;
  double _lastWordConfidence = 0.0;
  bool _needsCurrentAyahCorrection = false;
  String _transcript = '';

  String _trackingMode = '';
  String _searchWindow = '';
  int _fallbackCount = 0;
  bool _taraweehModeEnabled = false;
  bool _debugModeEnabled = false;
  bool _showDebugScreen = false;
  bool _taraweehJustStarted = false;
  bool _hasRecitedSinceTaraweehStart = false;
  bool _taraweehSurahComplete = false;

  bool _promptModeEnabled = false;
  double _vadThreshold = -48.0;

  String _currentAyahText = '';
  String _nextAyahText = '';
  String _prevAyahText = '';
  String _promptAyahText = '';

  int _promptTimeout = 15;
  String _promptAggressiveness = 'Normal';
  bool _predictiveLuqmahEnabled = false;
  int _promptRepeatInterval = 3;
  int _promptMaxRepeats = 3;
  int _promptAyahCount = 1;
  bool _isSpeaking = false;
  DateTime _lastSpeechTime = DateTime.now();
  DateTime _lastProgressionTime = DateTime.now();
  double _currentRMS = -100.0;
  double _noiseFloor = -60.0;
  DateTime _lastVoiceLevelTime = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _listeningStartedAt = DateTime.fromMillisecondsSinceEpoch(0);
  PromptState _promptState = PromptState.DISABLED;
  String _promptStateMessage = 'PROMPT DISABLED';
  int _promptRepeatCount = 0;
  bool _isPromptConfirmed = false;
  int _assistedSurah = 0;
  int _assistedAyah = 0;
  int _assistedWordIndex = 0;
  int _promptAudioSurah = 0;
  int _promptAudioAyah = 0;
  List<Map<String, dynamic>> _promptAudioVerses = [];
  Timer? _promptLoopTimer;
  Timer? _promptCooldownTimer;
  StreamSubscription<Amplitude>? _ampSub;
  int _lastLoggedPauseSecond = -1;

  bool _isMutashabihat = false;
  final AudioPlayer _audioPlayer = AudioPlayer();
  static const int _minimumInterruptiblePromptMs = 3000;
  static const int _promptSpeechConfirmationMs = 2200;
  static const int _taraweehBasePromptFloorMs = 7000;
  static const int _taraweehStartupPromptFloorMs = 12000;
  static const int _taraweehMuqattaatPromptFloorMs = 12000;
  static const int _taraweehEarlyCorrectionPromptFloorMs = 10000;
  static const int _taraweehRepeatedRefrainPromptFloorMs = 11000;
  bool _audioPlayedForCurrentPause = false;
  int _promptPlaybackGeneration = 0;
  bool _promptCompletionHandled = true;
  DateTime _ignorePromptAudioUntil = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _promptPlaybackStartedAt = DateTime.fromMillisecondsSinceEpoch(0);
  double _promptPlaybackLeakFloor = -100.0;
  double _promptPlaybackLastRms = -100.0;
  DateTime? _promptSpeechCandidateSince;
  String _luqmahReciterFolder = LuqmahReciters.defaultFolder;
  String _promptPinnedReciterFolder = '';
  String _promptPinnedReciterKey = '';
  DateTime _lastMistakeCorrectionAt = DateTime.fromMillisecondsSinceEpoch(0);
  String _lastMistakeCorrectionKey = '';
  DateTime _lastWordCorrectionAt = DateTime.fromMillisecondsSinceEpoch(0);
  String _lastWordCorrectionKey = '';

  StreamSubscription<String>? _logSub;
  StreamSubscription<Map<String, dynamic>>? _engineSub;

  // Animation controllers
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    )..repeat(reverse: true);
    _pulseAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _logs.addAll(globalLogger.history);
    _logSub = globalLogger.onLog.listen((logMessage) {
      if (!mounted) return;
      setState(() {
        _logs.insert(0, logMessage);
        if (_logs.length > 200) _logs.removeLast();
      });
    });

    _playerStateSub = _audioPlayer.playerStateStream.listen(
      (state) {
        if (!mounted) return;
        if (state.processingState == ProcessingState.completed) {
          if (_promptState == PromptState.PLAYING_AUDIO ||
              _promptState == PromptState.PROMPT_REPEAT) {
            _handlePromptPlaybackCompleted();
          }
        }
      },
      onError: (Object error, StackTrace stack) {
        globalLogger.logError('Prompt audio player error: $error', stack);
      },
    );

    _listenToEngine();
    _initEngine();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.reload();
    if (!mounted) return;
    final savedVadThreshold = prefs.getDouble('vad_threshold');
    setState(() {
      _promptTimeout = prefs.getInt('prompt_timeout') ?? 15;
      _promptAggressiveness =
          prefs.getString('prompt_aggressiveness') ?? 'Normal';
      _predictiveLuqmahEnabled = prefs.getBool('predictive_luqmah') ?? false;
      final savedRepeatInterval = prefs.getInt('prompt_repeat_interval');
      _promptRepeatInterval = const [2, 3, 4].contains(savedRepeatInterval)
          ? savedRepeatInterval!
          : 3;
      _promptMaxRepeats = prefs.getInt('prompt_max_repeats') ?? 3;
      final savedPromptAyahCount = prefs.getInt('prompt_ayah_count');
      _promptAyahCount = const [1, 2, 3].contains(savedPromptAyahCount)
          ? savedPromptAyahCount!
          : 1;
      _luqmahReciterFolder = LuqmahReciters.fromFolder(
        prefs.getString('luqmah_reciter_folder'),
      ).folder;
      _debugModeEnabled = prefs.getBool('debug_mode') ?? false;
      _promptModeEnabled = _promptAggressiveness != 'Off';
      _vadThreshold =
          savedVadThreshold == null ||
              savedVadThreshold < -70.0 ||
              savedVadThreshold > -10.0
          ? -48.0
          : savedVadThreshold;
      widget.audioService.enhancementEnabled =
          prefs.getBool('audio_enhancement') ?? true;
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _pulseController.dispose();
    _logSub?.cancel();
    _engineSub?.cancel();
    _audioSub?.cancel();
    _ampSub?.cancel();
    _captureStateSub?.cancel();
    _playerStateSub?.cancel();
    _promptLoopTimer?.cancel();
    _promptCooldownTimer?.cancel();
    _audioPlayer.stop();
    _audioPlayer.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused ||
        state == AppLifecycleState.detached) {
      unawaited(_stopListening(status: 'Paused while app is in background'));
    }
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
      if (mounted) {
        setState(() {
          _connectionStatus = 'Failed';
          _recognitionStatus = 'Connection failure';
        });
      }
    }
  }

  final List<String> _logs = [];

  void _addLog(String logMsg) {
    globalLogger.log(logMsg);
  }

  void _setPromptState(PromptState newState, String message) {
    if (!mounted) return;
    if (_promptState != newState || _promptStateMessage != message) {
      if (_promptState != newState) {
        _addLog(
          '[Prompt] Transition: ${_promptState.name} -> ${newState.name} ($message)',
        );
      }
      setState(() {
        _promptState = newState;
        _promptStateMessage = message;
      });
    }
  }

  void _evaluatePromptState() {
    if (!mounted || !_isListening || !_promptModeEnabled) {
      if (_promptState != PromptState.DISABLED) {
        _cancelPrompt();
        _setPromptState(PromptState.DISABLED, 'Prompt audio off');
      }
      return;
    }

    if (_promptAggressiveness == 'Off') {
      if (_promptState != PromptState.DISABLED) {
        _cancelPrompt();
        _setPromptState(PromptState.DISABLED, 'Prompt audio off');
      }
      return;
    }

    if (!widget.engine.isReady) {
      _setPromptState(
        PromptState.WAITING_FOR_LOCK,
        'Recognition engine loading',
      );
      return;
    }

    if (_promptState == PromptState.PLAYING_AUDIO ||
        _promptState == PromptState.PROMPT_REPEAT) {
      return;
    }

    if (_isSpeaking) {
      if (_audioPlayedForCurrentPause ||
          _assistedAyah > 0 ||
          _promptState == PromptState.RECOVERY_MODE) {
        _promptCooldownTimer?.cancel();
        _promptCooldownTimer = null;
        _promptRepeatCount = 0;
        if (_promptState != PromptState.RECOVERY_MODE) {
          _setPromptState(
            PromptState.RECOVERY_MODE,
            'Listening for corrected recitation',
          );
        }
        return;
      }
      _setPromptState(PromptState.PROMPT_READY, 'PROMPT READY (Speech Active)');
      return;
    }

    // A repeat session owns one pinned target until the user resumes. The
    // single cooldown timer below is the only code allowed to request repeats.
    if (_assistedAyah > 0 ||
        _promptState == PromptState.PROMPT_COOLDOWN ||
        _promptState == PromptState.PROMPT_EXPIRED) {
      return;
    }

    // Determine Anchor Position
    int anchorSurah = _currentSurah > 0 ? _currentSurah : _lastSurah;
    int anchorAyah = _currentAyah > 0 ? _currentAyah : _lastAyah;
    bool hasValidAnchor = anchorSurah > 0 && anchorAyah > 0;

    if (!hasValidAnchor) {
      _setPromptState(
        PromptState.WAITING_FOR_LOCK,
        'WAITING FOR FIRST LOCK OR TARAWEEH CONFIG',
      );
      return;
    }

    if (_taraweehModeEnabled &&
        _taraweehSurahComplete &&
        !_shouldCorrectCurrentAyah(anchorAyah)) {
      _setPromptState(PromptState.PROMPT_READY, 'Surah complete');
      return;
    }

    if (_taraweehModeEnabled && !_hasRecitedSinceTaraweehStart) {
      _setPromptState(PromptState.PROMPT_READY, 'Waiting for recitation');
      return;
    }

    final now = DateTime.now();
    final idleTimeSpeech = now.difference(_lastSpeechTime).inMilliseconds;
    final idleTimeProgression = now
        .difference(_lastProgressionTime)
        .inMilliseconds;

    final idleMs = idleTimeSpeech < idleTimeProgression
        ? idleTimeSpeech
        : idleTimeProgression;

    int currentPauseSeconds = idleMs ~/ 1000;
    if (currentPauseSeconds > 0 &&
        currentPauseSeconds != _lastLoggedPauseSecond &&
        !_isSpeaking) {
      _lastLoggedPauseSecond = currentPauseSeconds;
      _addLog('[PAUSE] timer = $currentPauseSeconds');
    }

    final totalTimeoutMs = _promptTimeout * 1000;
    final predictiveTimeoutMs = _predictiveLuqmahEnabled
        ? math.max(2500, totalTimeoutMs - 2500)
        : totalTimeoutMs;

    final needsCurrentAyahCorrection = _shouldCorrectCurrentAyah(anchorAyah);
    int targetAyahToPrompt = needsCurrentAyahCorrection
        ? anchorAyah
        : anchorAyah + 1;
    if (_taraweehJustStarted) {
      targetAyahToPrompt = anchorAyah;
    }
    final effectiveTimeoutMs = math.max(
      predictiveTimeoutMs,
      _dynamicPromptTimeoutFloorMs(anchorSurah, targetAyahToPrompt),
    );

    if (idleMs >= effectiveTimeoutMs) {
      _addLog('[PROMPT] triggered for $anchorSurah:$targetAyahToPrompt');
      _setPromptState(PromptState.PLAYING_AUDIO, 'PLAYING AUDIO');
      setState(() {
        _assistedSurah = anchorSurah;
        _assistedAyah = targetAyahToPrompt;
        _assistedWordIndex = needsCurrentAyahCorrection ? _wordPosition : 0;
        _promptRepeatCount = 1;
      });
      widget.engine.sendAssistedPrompt(
        _assistedSurah,
        _assistedAyah,
        count: _promptAyahCount,
        wordIndex: _assistedWordIndex,
      );
    } else if (idleMs >= effectiveTimeoutMs - 3000 &&
        effectiveTimeoutMs >= 3000) {
      final countdown = ((effectiveTimeoutMs - idleMs) ~/ 1000) + 1;
      final cdText = countdown > 0 ? 'COUNTDOWN: $countdown' : 'COUNTDOWN: 1';
      _setPromptState(PromptState.PAUSE_TIMER_RUNNING, cdText);
    } else {
      _setPromptState(PromptState.PROMPT_READY, 'Listening for a pause');
    }
  }

  void _triggerMistakeCorrection(
    int expectedSurah,
    int expectedAyah,
    int detectedSurah,
    int detectedAyah,
  ) {
    if (!mounted || !_isListening || !_promptModeEnabled) return;
    final now = DateTime.now();
    final key = '$expectedSurah:$expectedAyah->$detectedSurah:$detectedAyah';
    if (_lastMistakeCorrectionKey == key &&
        now.difference(_lastMistakeCorrectionAt).inSeconds < 3) {
      _addLog(
        '[MISTAKE] Correction already in progress for $expectedSurah:$expectedAyah',
      );
      return;
    }
    _lastMistakeCorrectionKey = key;
    _lastMistakeCorrectionAt = now;

    _addLog(
      '[MISTAKE] Detected recitation of $detectedSurah:$detectedAyah instead of $expectedSurah:$expectedAyah',
    );
    _setPromptState(
      PromptState.PLAYING_AUDIO,
      'MISTAKE DETECTED - Correcting...',
    );

    setState(() {
      _assistedSurah = expectedSurah;
      _assistedAyah = expectedAyah;
      _assistedWordIndex =
          expectedSurah == _currentSurah && expectedAyah == _currentAyah
          ? _wordPosition
          : 0;
      _promptRepeatCount = 1;
      _promptCompletionHandled = false;
      _audioPlayedForCurrentPause = false;
    });

    widget.engine.sendAssistedPrompt(
      _assistedSurah,
      _assistedAyah,
      count: _promptAyahCount,
      wordIndex: _assistedWordIndex,
    );
  }

  void _deferUncertainMistakeCorrection(
    int expectedSurah,
    int expectedAyah,
    String reason,
  ) {
    final now = DateTime.now();
    _lastProgressionTime = now;
    _needsCurrentAyahCorrection = true;
    _addLog(
      '[MISTAKE] Deferred uncertain correction for $expectedSurah:$expectedAyah'
      ' ($reason)',
    );
    if (mounted) {
      setState(() {
        _recognitionStatus = 'Waiting for clearer correction evidence';
      });
    }
  }

  double _resolveProgressCoverage(Map<String, dynamic> event) {
    final progress = event['progress_coverage'];
    if (progress is num) return progress.toDouble();
    final word = event['word_coverage'];
    final section = event['section_coverage'];
    final wordCoverage = word is num ? word.toDouble() : _wordCoverage;
    final sectionCoverage = section is num
        ? section.toDouble()
        : _sectionCoverage;
    return math.max(wordCoverage, sectionCoverage);
  }

  bool _didRecognitionAdvance({
    required int surah,
    required int ayah,
    int? wordIndex,
    int? sectionIndex,
    double? progressCoverage,
    double? wordCoverage,
    double? sectionCoverage,
  }) {
    if (surah != _currentSurah || ayah != _currentAyah) {
      return true;
    }
    if ((wordIndex ?? _wordPosition) > _wordPosition) {
      return true;
    }
    if ((sectionIndex ?? _sectionPosition) > _sectionPosition) {
      return true;
    }
    final incomingProgress =
        progressCoverage ??
        math.max(
          wordCoverage ?? _wordCoverage,
          sectionCoverage ?? _sectionCoverage,
        );
    if (incomingProgress > _progressCoverage + 0.03) {
      return true;
    }
    if ((wordCoverage ?? _wordCoverage) > _wordCoverage + 0.04) {
      return true;
    }
    if ((sectionCoverage ?? _sectionCoverage) > _sectionCoverage + 0.04) {
      return true;
    }
    return false;
  }

  int _countWords(String text) {
    return text
        .split(RegExp(r'\s+'))
        .where((word) => word.trim().isNotEmpty)
        .length;
  }

  String _normalizePromptArabic(String text) {
    return text
        .replaceFirst('\uFEFF', '')
        .replaceAll(
          RegExp(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]'),
          '',
        )
        .replaceAll(RegExp(r'[ۖۗۘۙۚۛۜ۝]'), '')
        .replaceAll('ٱ', 'ا')
        .replaceAll('أ', 'ا')
        .replaceAll('إ', 'ا')
        .replaceAll('آ', 'ا')
        .replaceAll('ى', 'ي')
        .replaceAll('ؤ', 'و')
        .replaceAll('ئ', 'ي')
        .replaceAll('ة', 'ه')
        .replaceAll(RegExp(r'[^\u0621-\u064A\s]'), ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }

  String _trimBismillahPrefix(String text) {
    const bismillah = 'بسم الله الرحمن الرحيم';
    final normalized = _normalizePromptArabic(text);
    if (!normalized.startsWith(bismillah)) {
      return normalized;
    }
    return normalized.substring(bismillah.length).trim();
  }

  bool _isMuqattaatOpeningText(String text) {
    final trimmed = _trimBismillahPrefix(text);
    return const {
      'الم',
      'المص',
      'المر',
      'الر',
      'كهيعص',
      'طه',
      'طسم',
      'طس',
      'يس',
      'ص',
      'حم',
      'حم عسق',
      'ق',
      'ن',
    }.contains(trimmed);
  }

  bool _isRahmanRepeatedRefrain(int surah, int ayah, String text) {
    if (surah != 55) return false;
    if (!const {
      13,
      16,
      18,
      21,
      23,
      25,
      28,
      30,
      32,
      34,
      36,
      38,
      40,
      42,
      45,
      47,
      49,
      51,
      53,
      55,
      57,
      59,
      61,
      63,
      65,
      67,
      69,
      71,
      73,
      75,
      77,
    }.contains(ayah)) {
      return false;
    }
    if (text.trim().isEmpty) return true;
    return _trimBismillahPrefix(text) == 'فباي الاء ربكما تكذبان';
  }

  String _promptAnchorText(int surah, int ayah) {
    if (surah == _currentSurah &&
        ayah == _currentAyah &&
        _currentAyahText.isNotEmpty) {
      return _currentAyahText;
    }
    if (surah == _currentSurah &&
        ayah == _currentAyah + 1 &&
        _nextAyahText.isNotEmpty) {
      return _nextAyahText;
    }
    if (surah == _currentSurah &&
        ayah == _currentAyah - 1 &&
        _prevAyahText.isNotEmpty) {
      return _prevAyahText;
    }
    return _promptAyahText;
  }

  int _dynamicPromptTimeoutFloorMs(int surah, int ayah) {
    final anchorText = _promptAnchorText(surah, ayah);
    final hasAnchorText = anchorText.trim().isNotEmpty;
    final words = _countWords(anchorText);
    final trimmedAnchor = _trimBismillahPrefix(anchorText);
    final normalizedCharCount = trimmedAnchor.replaceAll(' ', '').length;
    final correctingCurrentAyah = _shouldCorrectCurrentAyah(ayah);
    var floorMs = 0;
    if (_taraweehModeEnabled) {
      floorMs = _taraweehBasePromptFloorMs;
      if (!hasAnchorText || !_hasRecitedSinceTaraweehStart) {
        floorMs = math.max(floorMs, _taraweehStartupPromptFloorMs);
      }
    }
    if (_isMuqattaatOpeningText(anchorText)) {
      floorMs = math.max(
        floorMs,
        _taraweehModeEnabled ? _taraweehMuqattaatPromptFloorMs : 7000,
      );
    } else if (_isRahmanRepeatedRefrain(surah, ayah, anchorText)) {
      floorMs = math.max(
        floorMs,
        _taraweehModeEnabled ? _taraweehRepeatedRefrainPromptFloorMs : 9000,
      );
    } else if (normalizedCharCount >= 45) {
      floorMs = math.max(floorMs, _taraweehModeEnabled ? 11000 : 7500);
    } else if (normalizedCharCount >= 28) {
      floorMs = math.max(floorMs, _taraweehModeEnabled ? 8500 : 6000);
    }
    if (words >= 12) {
      floorMs = math.max(floorMs, _taraweehModeEnabled ? 11000 : 8000);
    } else if (words >= 7) {
      floorMs = math.max(floorMs, _taraweehModeEnabled ? 8500 : 6000);
    } else if (_taraweehModeEnabled && words >= 4) {
      floorMs = math.max(floorMs, 7500);
    }
    if (correctingCurrentAyah &&
        (_progressCoverage <= 0.45 ||
            _wordPosition <= 1 ||
            _sectionPosition <= 1)) {
      floorMs = math.max(
        floorMs,
        _taraweehModeEnabled ? _taraweehEarlyCorrectionPromptFloorMs : 5500,
      );
    }
    return floorMs;
  }

  bool _isActionableWordCorrection(Map<dynamic, dynamic> correction) {
    final confidence = correction['confidence'] is num
        ? (correction['confidence'] as num).toDouble()
        : 1.0;
    final expected = correction['expected']?.toString() ?? '';
    final got = correction['got']?.toString() ?? '';
    final expectedLen = expected.replaceAll(RegExp(r'\s+'), '').length;
    final gotLen = got.replaceAll(RegExp(r'\s+'), '').length;
    if (confidence > 0.35) return false;
    if (expectedLen <= 3) return false;
    if (gotLen < math.max(4, (expectedLen * 0.55).ceil())) return false;
    return true;
  }

  bool _shouldCorrectCurrentAyah(int anchorAyah) {
    if (_currentAyah <= 0 || anchorAyah != _currentAyah) return false;
    if (_needsCurrentAyahCorrection) return true;
    if (_wordPosition <= 0 || _totalWords <= 0) return false;
    final liveCoverage = _progressCoverage > 0
        ? _progressCoverage
        : _wordCoverage;
    return liveCoverage > 0 && liveCoverage < 0.60;
  }

  void _interruptPromptPlaybackDueToSpeech() {
    if (_promptState != PromptState.PLAYING_AUDIO &&
        _promptState != PromptState.PROMPT_REPEAT) {
      return;
    }
    _addLog('[PROMPT] Luqmah interrupted by recitation');
    _promptSpeechCandidateSince = null;
    _promptPlaybackLeakFloor = -100.0;
    _promptPlaybackLastRms = -100.0;
    _promptCompletionHandled = true;
    _ignorePromptAudioUntil = DateTime.now().add(
      const Duration(milliseconds: 160),
    );
    _cancelPrompt(clearAssisted: true, resetPauseLatch: false);
    if (mounted) {
      setState(() => _isSpeaking = true);
    }
    _setPromptState(
      PromptState.RECOVERY_MODE,
      'Reciter resumed - listening for correction',
    );
  }

  void _cancelPrompt({bool clearAssisted = true, bool resetPauseLatch = true}) {
    if (!mounted) return;
    _promptCooldownTimer?.cancel();
    _promptCooldownTimer = null;
    _promptPlaybackGeneration++;
    _promptCompletionHandled = true;
    if (_audioPlayer.playing) {
      unawaited(_audioPlayer.stop());
    }
    if (resetPauseLatch) {
      _audioPlayedForCurrentPause = false;
    }
    _promptRepeatCount = 0;
    _promptPlaybackStartedAt = DateTime.fromMillisecondsSinceEpoch(0);
    _promptPlaybackLeakFloor = -100.0;
    _promptPlaybackLastRms = -100.0;
    _promptSpeechCandidateSince = null;
    if (clearAssisted && _assistedAyah > 0) {
      _assistedSurah = 0;
      _assistedAyah = 0;
      _assistedWordIndex = 0;
      _promptAudioSurah = 0;
      _promptAudioAyah = 0;
      _promptAudioVerses = [];
      _promptAyahText = '';
      _promptPinnedReciterFolder = '';
      _promptPinnedReciterKey = '';
      widget.engine.clearAssistedPrompt();
    }
    if (_promptState == PromptState.PLAYING_AUDIO ||
        _promptState == PromptState.PROMPT_REPEAT ||
        _promptState == PromptState.PROMPT_DISPLAYED ||
        _promptState == PromptState.PROMPT_COOLDOWN ||
        _promptState == PromptState.PROMPT_EXPIRED) {
      _setPromptState(PromptState.PROMPT_READY, 'Prompt Cancelled');
    }
  }

  void _handlePromptPlaybackCompleted() {
    if (_promptCompletionHandled || !mounted) return;
    _promptCompletionHandled = true;
    widget.engine.discardPendingAudio();
    _ignorePromptAudioUntil = DateTime.now().add(
      const Duration(milliseconds: 450),
    );
    _promptCooldownTimer?.cancel();

    if (_promptRepeatCount >= _promptMaxRepeats) {
      _addLog(
        '[PROMPT] Maximum repeats reached for $_assistedSurah:$_assistedAyah',
      );
      _setPromptState(PromptState.PROMPT_EXPIRED, 'Waiting for recitation');
      return;
    }

    _addLog(
      '[PROMPT] Luqmah finished; listening for '
      '$_promptRepeatInterval seconds before repeating same target',
    );
    _setPromptState(
      PromptState.PROMPT_COOLDOWN,
      'Listening - repeats in $_promptRepeatInterval seconds',
    );
    _promptCooldownTimer = Timer(Duration(seconds: _promptRepeatInterval), () {
      if (!mounted ||
          !_isListening ||
          _isSpeaking ||
          _promptState != PromptState.PROMPT_COOLDOWN ||
          _assistedSurah <= 0 ||
          _assistedAyah <= 0) {
        return;
      }
      setState(() => _promptRepeatCount++);
      _addLog(
        '[PROMPT] Repeating pinned target '
        '$_assistedSurah:$_assistedAyah ($_promptRepeatCount/$_promptMaxRepeats)',
      );
      _setPromptState(
        PromptState.PROMPT_REPEAT,
        'PLAYING AUDIO (Repeat $_promptRepeatCount)',
      );
      widget.engine.sendAssistedPrompt(
        _assistedSurah,
        _assistedAyah,
        count: _promptAyahCount,
        wordIndex: _assistedWordIndex,
      );
    });
  }

  bool _isPromptPlaybackActive() {
    return _promptState == PromptState.PLAYING_AUDIO ||
        _promptState == PromptState.PROMPT_REPEAT;
  }

  Future<void> _playPromptAudio(
    List<Map<String, dynamic>> verses, {
    bool forceRepeat = false,
  }) async {
    if (verses.isEmpty) return;
    if (_audioPlayedForCurrentPause && !forceRepeat) return;
    _audioPlayedForCurrentPause = true;
    final generation = ++_promptPlaybackGeneration;

    List<AudioSource> sourcesFor(String reciter) {
      return verses.map((verse) {
        final surahNum = verse['surah'] as int;
        final ayahNum = verse['ayah'] as int;
        final surah = surahNum.toString().padLeft(3, '0');
        final ayah = ayahNum.toString().padLeft(3, '0');
        final uriSource = AudioSource.uri(
          Uri.parse('https://everyayah.com/data/$reciter/$surah$ayah.mp3'),
        );
        final startMs = verse['estimated_start_ms'] is int
            ? verse['estimated_start_ms'] as int
            : 0;
        final strategy = verse['prompt_strategy']?.toString() ?? 'whole_ayah';
        if (strategy == 'phrase_boundary' && startMs > 0) {
          return ClippingAudioSource(
            child: uriSource,
            start: Duration(milliseconds: startMs),
          );
        }
        return uriSource;
      }).toList();
    }

    final labels = verses
        .map((verse) => '${verse['surah']}:${verse['ayah']}')
        .join(', ');
    _addLog('[PROMPT] Luqmah queued: $labels');
    if (_promptPinnedReciterKey != labels) {
      _promptPinnedReciterKey = labels;
      _promptPinnedReciterFolder = '';
    }
    final reciterFolders = _promptPinnedReciterFolder.isNotEmpty
        ? <String>[_promptPinnedReciterFolder]
        : <String>{
            _luqmahReciterFolder,
            LuqmahReciters.defaultFolder,
            'Abdul_Basit_Murattal_192kbps',
          }.toList();
    for (final reciter in reciterFolders) {
      try {
        await _audioPlayer.stop();
        await _audioPlayer.setAudioSource(
          ConcatenatingAudioSource(
            useLazyPreparation: false,
            children: sourcesFor(reciter),
          ),
        );
        if (generation != _promptPlaybackGeneration || !mounted) return;
        _promptCompletionHandled = false;
        _promptPlaybackStartedAt = DateTime.now();
        _promptPlaybackLeakFloor = -100.0;
        _promptPlaybackLastRms = -100.0;
        _promptSpeechCandidateSince = null;
        _promptPinnedReciterKey = labels;
        _promptPinnedReciterFolder = reciter;
        _addLog(
          '[PROMPT] Luqmah playback started with $reciter '
          '(${verses.length} ayah)',
        );
        await _audioPlayer.play();
        return;
      } catch (error) {
        if (generation != _promptPlaybackGeneration || !mounted) return;
        _promptCompletionHandled = true;
        _addLog('[PROMPT] Audio source $reciter failed: $error');
      }
    }
    _audioPlayedForCurrentPause = false;
    if (mounted) {
      _setPromptState(
        PromptState.PROMPT_EXPIRED,
        'Audio unavailable. Check internet and tap prompt to retry.',
      );
    }
  }

  void _handleEvent(Map<String, dynamic> event) {
    final type = event['type'];
    globalLogger.log('SERVER RESPONSE: $type');
    final promptOwnAudioMayBeRecognized =
        _promptState == PromptState.PLAYING_AUDIO ||
        _promptState == PromptState.PROMPT_REPEAT ||
        DateTime.now().isBefore(_ignorePromptAudioUntil);
    if (promptOwnAudioMayBeRecognized &&
        (type == 'verse_match' || type == 'word_progress')) {
      _addLog('[PROMPT] Ignored recognition caused by luqmah playback');
      return;
    }
    if (type == 'status') {
      final msg = event['message'] ?? '';
      _trackingMode = event['tracking_mode'] ?? _trackingMode;
      _searchWindow = event['search_window'] ?? _searchWindow;
      _fallbackCount = event['fallback_count'] ?? _fallbackCount;
      final reason = event['reason']?.toString() ?? '';
      final expectedSurah = event['expected_surah'] is num
          ? (event['expected_surah'] as num).toInt()
          : 0;
      final expectedAyah = event['expected_ayah'] is num
          ? (event['expected_ayah'] as num).toInt()
          : 0;
      if (_taraweehModeEnabled &&
          reason == 'no_context_progress' &&
          expectedSurah == _currentSurah &&
          expectedAyah > 0) {
        _needsCurrentAyahCorrection = expectedAyah == _currentAyah;
      }
      _addLog(msg);
      if (msg.contains('Connecting')) {
        _connectionStatus = 'Connecting...';
      } else if (msg.contains('Connected')) {
        _connectionStatus = 'Connected';
      } else if (msg.contains('Standalone engine ready') ||
          msg.contains('On-device recognition session ready') ||
          msg.contains('Local server started') ||
          msg.contains('Headless WebView running')) {
        _connectionStatus = widget.engine.isReady
            ? 'Connected'
            : 'Connecting...';
        _recognitionStatus = msg;
      } else if (msg.contains('Initializing engine') ||
          msg.contains('Loading ') ||
          msg.contains('Creating on-device') ||
          msg.contains('Starting ONNX') ||
          msg.contains('On-device recognition is still loading') ||
          msg.contains('Starting local server') ||
          msg.contains('Starting Headless WebView')) {
        _connectionStatus = 'Connecting...';
        _recognitionStatus = msg;
      } else if (msg.contains('closed')) {
        _connectionStatus = 'Disconnected';
        _recognitionStatus = 'WebSocket disconnected';
      }
    } else if (type == 'ready') {
      _connectionStatus = 'Connected';
      _recognitionStatus = 'Ready. Press mic to start.';
      _addLog('WebSocket connected and ready.');
    } else if (type == 'verse_candidate') {
      final candidates = event['candidates'];
      if (candidates is List && candidates.isNotEmpty) {
        final firstRaw = candidates.first;
        if (firstRaw is Map) {
          final first = Map<String, dynamic>.from(firstRaw);
          final surah = first['surah'];
          final ayah = first['ayah'];
          final confidence = first['confidence'];
          final confidenceText = confidence is num
              ? ' ${(confidence * 100).toStringAsFixed(0)}%'
              : '';
          final stable = event['stable'] == true ? 'stable' : 'checking';
          _recognitionStatus =
              'Candidate $surah:$ayah$confidenceText ($stable)';
          _trackingMode = 'Candidate search';
          if (event['final_flush'] == true || event['stable'] == true) {
            _addLog(
              '[CANDIDATE] $surah:$ayah$confidenceText '
              '(${event['final_flush'] == true ? 'final' : stable})',
            );
          }
        }
      }
    } else if (type == 'raw_transcript') {
      final text = event['text']?.toString() ?? '';
      final confidence = event['confidence'];
      _transcript = text;
      if (text.trim().isNotEmpty) {
        final confidenceText = confidence is num
            ? ' ${(confidence * 100).toStringAsFixed(0)}%'
            : '';
        _recognitionStatus = 'Heard recitation - matching$confidenceText';
      }
    } else if (type == 'assisted_verse_text') {
      final rawPromptVerses = event['prompt_verses'];
      final fallbackSurah = event['surah'] is num
          ? (event['surah'] as num).toInt()
          : _currentSurah;
      final fallbackAyah = event['ayah'] is num
          ? (event['ayah'] as num).toInt()
          : _assistedAyah;
      final parsedPromptVerses = <Map<String, dynamic>>[];
      if (rawPromptVerses is List) {
        for (final rawVerse in rawPromptVerses.whereType<Map>()) {
          final surah = rawVerse['surah'];
          final ayah = rawVerse['ayah'];
          if (surah is num && ayah is num && surah > 0 && ayah > 0) {
            parsedPromptVerses.add({
              'surah': surah.toInt(),
              'ayah': ayah.toInt(),
              'start_word': rawVerse['start_word'] is num
                  ? (rawVerse['start_word'] as num).toInt()
                  : 1,
              'estimated_start_ms': rawVerse['estimated_start_ms'] is num
                  ? (rawVerse['estimated_start_ms'] as num).toInt()
                  : 0,
              'prompt_strategy': rawVerse['prompt_strategy'] ?? 'whole_ayah',
              'total_words': rawVerse['total_words'] is num
                  ? (rawVerse['total_words'] as num).toInt()
                  : 0,
            });
          }
        }
      }
      if (parsedPromptVerses.isEmpty && fallbackSurah > 0 && fallbackAyah > 0) {
        parsedPromptVerses.add({'surah': fallbackSurah, 'ayah': fallbackAyah});
      }
      setState(() {
        _promptAyahText = event['ayah_text'] ?? '';
        _promptAudioSurah = fallbackSurah;
        _promptAudioAyah = fallbackAyah;
        _promptAudioVerses = parsedPromptVerses;
        if (_assistedAyah <= 0 &&
            fallbackSurah > 0 &&
            fallbackAyah > 0 &&
            fallbackSurah == _currentSurah &&
            fallbackAyah == _currentAyah) {
          _currentAyahText = event['ayah_text'] ?? _currentAyahText;
          _surahNameEn = event['surah_name_en'] ?? _surahNameEn;
          _surahNameAr = event['surah_name'] ?? _surahNameAr;
        }
        if (_assistedAyah > 0) {
          // The backend normalizes cross-surah positions. Pin repeats to the
          // normalized first ayah returned, not to an invalid anchor + 1.
          _assistedSurah = fallbackSurah;
          _assistedAyah = fallbackAyah;
        }
      });
      _addLog('Received assisted verse text.');
      if ((_promptState == PromptState.PLAYING_AUDIO ||
              _promptState == PromptState.PROMPT_REPEAT) &&
          _promptAudioSurah > 0 &&
          _promptAudioAyah > 0) {
        unawaited(
          _playPromptAudio(
            _promptAudioVerses,
            forceRepeat: _promptState == PromptState.PROMPT_REPEAT,
          ),
        );
      }
    } else if (type == 'verse_match') {
      int newSurah = event['surah'] ?? 0;
      int newAyah = event['ayah'] ?? 0;
      final sameAyahAsCurrent =
          newSurah == _currentSurah && newAyah == _currentAyah;
      final eventWord = event['word_index'] is num
          ? (event['word_index'] as num).toInt()
          : null;
      final eventSection = event['section_index'] is num
          ? (event['section_index'] as num).toInt()
          : null;
      final incomingWordCoverage = event['word_coverage'] is num
          ? (event['word_coverage'] as num).toDouble()
          : null;
      final incomingSectionCoverage = event['section_coverage'] is num
          ? (event['section_coverage'] as num).toDouble()
          : null;
      final incomingProgressCoverage = _resolveProgressCoverage(event);
      final hadActivePrompt = _assistedSurah > 0 && _assistedAyah > 0;
      final matchedAssistedTarget =
          hadActivePrompt &&
          newSurah == _assistedSurah &&
          newAyah == _assistedAyah;

      if (_taraweehJustStarted) {
        _taraweehJustStarted = false;
      }
      if (widget.engine.isReady &&
          _taraweehModeEnabled &&
          !_hasRecitedSinceTaraweehStart) {
        _hasRecitedSinceTaraweehStart = true;
      }

      if (_didRecognitionAdvance(
        surah: newSurah,
        ayah: newAyah,
        wordIndex: eventWord,
        sectionIndex: eventSection,
        progressCoverage: incomingProgressCoverage,
        wordCoverage: incomingWordCoverage,
        sectionCoverage: incomingSectionCoverage,
      )) {
        _lastProgressionTime = DateTime.now();
      }

      if (matchedAssistedTarget) {
        setState(() {
          _isPromptConfirmed = true;
        });
        Future.delayed(const Duration(seconds: 2), () {
          if (mounted) setState(() => _isPromptConfirmed = false);
        });
      }
      if (hadActivePrompt) _cancelPrompt();

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
        _taraweehSurahComplete = event['surah_complete'] == true;
        final eventTotal = event['total_words'];
        final eventTotalSections = event['total_sections'];
        if (sameAyahAsCurrent) {
          if (eventWord != null && eventWord > _wordPosition) {
            _wordPosition = eventWord;
          }
          if (eventTotal is num && eventTotal > 0) {
            _totalWords = eventTotal.toInt();
          }
          if (eventSection != null && eventSection > _sectionPosition) {
            _sectionPosition = eventSection;
          }
          if (eventTotalSections is num && eventTotalSections > 0) {
            _totalSections = eventTotalSections.toInt();
          }
          _wordCoverage = (event['word_coverage'] ?? _wordCoverage).toDouble();
          _sectionCoverage = (event['section_coverage'] ?? _sectionCoverage)
              .toDouble();
          _lastWordConfidence =
              (event['word_confidence'] ?? _lastWordConfidence).toDouble();
        } else {
          _wordPosition = eventWord ?? 0;
          _totalWords = eventTotal is num ? eventTotal.toInt() : 0;
          _sectionPosition = eventSection ?? 0;
          _totalSections = eventTotalSections is num
              ? eventTotalSections.toInt()
              : 0;
          _wordCoverage = (event['word_coverage'] ?? 0.0).toDouble();
          _sectionCoverage = (event['section_coverage'] ?? _wordCoverage)
              .toDouble();
          _lastWordConfidence = (event['word_confidence'] ?? 0.0).toDouble();
        }
        _progressCoverage = _resolveProgressCoverage(event);
        final progressCoverage = _progressCoverage;
        _needsCurrentAyahCorrection =
            progressCoverage > 0 && progressCoverage < 0.82;
        if (_trackingMode == 'REWIND DETECTED (Confirming...)') {
          _recognitionStatus =
              'Matched Verse: $_currentSurah:$_currentAyah (Rewind Detected)';
        } else {
          _recognitionStatus = 'Matched Verse: $_currentSurah:$_currentAyah';
        }
        _addLog(
          'Match - Surah: $_surahNameEn ($_currentSurah), Ayah: $_currentAyah, Conf: ${(_confidence * 100).toStringAsFixed(0)}%',
        );
      });
    } else if (type == 'word_progress') {
      int newSurah = event['surah'] ?? _currentSurah;
      int newAyah = event['ayah'] ?? _currentAyah;
      int newWord = event['word_index'] ?? 0;
      final incomingWordCoverage = event['word_coverage'] is num
          ? (event['word_coverage'] as num).toDouble()
          : null;
      final incomingSectionCoverage = event['section_coverage'] is num
          ? (event['section_coverage'] as num).toDouble()
          : null;
      final incomingProgressCoverage = _resolveProgressCoverage(event);
      final incomingSection = event['section_index'] is num
          ? (event['section_index'] as num).toInt()
          : null;
      if (_taraweehModeEnabled && !_hasRecitedSinceTaraweehStart) {
        _hasRecitedSinceTaraweehStart = true;
      }

      if (_didRecognitionAdvance(
        surah: newSurah,
        ayah: newAyah,
        wordIndex: newWord,
        sectionIndex: incomingSection,
        progressCoverage: incomingProgressCoverage,
        wordCoverage: incomingWordCoverage,
        sectionCoverage: incomingSectionCoverage,
      )) {
        _lastProgressionTime = DateTime.now();
      }
      if (_assistedAyah > 0) _cancelPrompt();

      _currentSurah = newSurah;
      _currentAyah = newAyah;
      _wordPosition = newWord;
      _totalWords = event['total_words'] ?? 0;
      _confidence = event['confidence'] ?? _confidence;
      _lastWordConfidence = (event['confidence'] ?? _lastWordConfidence)
          .toDouble();
      _wordCoverage = (event['word_coverage'] ?? _wordCoverage).toDouble();
      _sectionPosition = event['section_index'] ?? _sectionPosition;
      _totalSections = event['total_sections'] ?? _totalSections;
      _sectionCoverage = (event['section_coverage'] ?? _sectionCoverage)
          .toDouble();
      _progressCoverage = _resolveProgressCoverage(event);
      _trackingMode = event['tracking_mode'] ?? _trackingMode;
      _searchWindow = event['search_window'] ?? _searchWindow;
      _fallbackCount = event['fallback_count'] ?? _fallbackCount;
      _currentAyahText = event['current_ayah_text'] ?? _currentAyahText;
      _nextAyahText = event['next_ayah_text'] ?? _nextAyahText;
      _prevAyahText = event['prev_ayah_text'] ?? _prevAyahText;
      _isMutashabihat = event['is_mutashabihat'] ?? false;
      _taraweehSurahComplete = event['surah_complete'] == true;
      if (_progressCoverage > 0 && _progressCoverage < 0.82) {
        _needsCurrentAyahCorrection = true;
      } else if (_progressCoverage >= 0.9 && _lastWordConfidence >= 0.62) {
        _needsCurrentAyahCorrection = false;
      }
      _recognitionStatus = _totalSections > 0
          ? 'Tracking... Ayah: $_currentAyah | Section: $_sectionPosition/$_totalSections'
          : 'Tracking... Ayah: $_currentAyah';
    } else if (type == 'word_correction') {
      final corrections = event['corrections'];
      if (corrections is List && corrections.isNotEmpty) {
        final first = corrections.first;
        if (first is Map) {
          final wordIndex = first['word_index'] ?? 0;
          final expected = first['expected'] ?? '';
          final got = first['got'] ?? '';
          final key =
              '${event['surah']}:${event['ayah']}:$wordIndex:$expected:$got';
          final now = DateTime.now();
          if (_lastWordCorrectionKey != key ||
              now.difference(_lastWordCorrectionAt).inSeconds >= 4) {
            _lastWordCorrectionKey = key;
            _lastWordCorrectionAt = now;
            _addLog(
              '[CORRECTION] Word $wordIndex may need attention: expected "$expected", heard "$got"',
            );
          }
          if (_isActionableWordCorrection(first)) {
            _needsCurrentAyahCorrection = true;
            _recognitionStatus = 'Check word $wordIndex';
          } else {
            _recognitionStatus = 'Tracking word-level detail';
          }
        }
      }
    } else if (type == 'mistake_detected') {
      final expectedSurah = event['expected_surah'] ?? 0;
      final expectedAyah = event['expected_ayah'] ?? 0;
      final detectedSurah = event['detected_surah'] ?? 0;
      final detectedAyah = event['detected_ayah'] ?? 0;
      final score = event['score'] is num
          ? (event['score'] as num).toDouble()
          : 0.0;
      final reason = event['reason']?.toString() ?? '';
      if (expectedSurah > 0 && expectedAyah > 0 && _taraweehModeEnabled) {
        if (detectedSurah <= 0 ||
            detectedAyah <= 0 ||
            score <= 0.0 ||
            reason == 'no_context_progress') {
          _deferUncertainMistakeCorrection(expectedSurah, expectedAyah, reason);
          return;
        }
        _triggerMistakeCorrection(
          expectedSurah,
          expectedAyah,
          detectedSurah,
          detectedAyah,
        );
      }
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
    _engineSub = widget.engine.onEvent.listen((event) {
      if (!mounted) return;
      _handleEvent(event);
      if (mounted) setState(() {});
    });
  }

  Future<void> _toggleListening() async {
    if (_isStarting) return; // Guard against race condition
    globalLogger.log('MIC BUTTON PRESSED');
    if (_isListening) {
      await _stopListening();
    } else {
      await _startListening();
    }
  }

  Future<void> _startListening() async {
    try {
      if (mounted) {
        setState(() {
          _isStarting = true;
          _recognitionStatus = 'Starting microphone...';
        });
      }

      final hasPermission = await widget.audioService.requestPermission();
      if (!hasPermission) {
        _addLog('[ERROR] Microphone permission denied');
        if (mounted) {
          setState(() {
            _recognitionStatus =
                'Microphone permission denied. Enable it in Android settings.';
            _isStarting = false;
          });
        }
        return;
      }

      if (!widget.engine.isReady) {
        _addLog('Recognition engine is not ready; requesting startup.');
        if (mounted) {
          setState(() {
            _recognitionStatus = 'Starting recognition engine...';
          });
        }
        await widget.engine.initialize();
      }

      _noiseFloor = -60.0;
      _lastVoiceLevelTime = DateTime.fromMillisecondsSinceEpoch(0);
      _listeningStartedAt = DateTime.now();
      _lastSpeechTime = DateTime.now();
      _lastProgressionTime = DateTime.now();
      _lastLoggedPauseSecond = -1;
      _progressCoverage = 0.0;

      await _audioSub?.cancel();
      await _ampSub?.cancel();
      await _captureStateSub?.cancel();

      // Forward the full enhanced PCM stream. VAD is used for prompt timing,
      // not as a hard gate, so the first sounds of an ayah are never clipped.
      _audioSub = widget.audioService.onAudioChunk.listen((chunk) {
        if (_isPromptPlaybackActive() ||
            DateTime.now().isBefore(_ignorePromptAudioUntil)) {
          return;
        }
        try {
          widget.engine.processAudioChunk(chunk);
        } catch (e, stack) {
          globalLogger.logError(
            'Exception during chunk forwarding: [${e.runtimeType}] $e',
            stack,
          );
        }
      });

      _ampSub = widget.audioService.onAmplitude.listen((amp) {
        if (!mounted) return;
        _handleAmplitude(amp.current);
      });

      _captureStateSub = widget.audioService.onStateChanged.listen((state) {
        if (!mounted || !_isListening) return;
        if (state == AudioCaptureState.failed ||
            state == AudioCaptureState.idle) {
          final error = widget.audioService.lastError;
          unawaited(
            _stopListening(
              status: error == null
                  ? 'Microphone stopped. Tap to restart.'
                  : 'Microphone error. Tap to retry.',
            ),
          );
        }
      });

      final started = await widget.audioService.start();
      if (!started) {
        throw StateError(
          widget.audioService.lastError ??
              'Android could not start the microphone.',
        );
      }

      _promptLoopTimer?.cancel();
      _promptLoopTimer = Timer.periodic(const Duration(milliseconds: 250), (_) {
        _evaluatePromptState();
      });

      if (mounted) {
        setState(() {
          _isListening = true;
          _isStarting = false;
          _recognitionStatus = widget.engine.isReady
              ? 'Listening for recitation...'
              : 'Listening. Waiting for recognition server...';
        });
      }
    } catch (e, stack) {
      globalLogger.logError(
        'Exception while starting microphone: [${e.runtimeType}] $e',
        stack,
      );
      await _stopListening(status: 'Could not start microphone. Tap to retry.');
    }
  }

  void _handlePromptPlaybackAmplitude(DateTime now) {
    if (_promptPlaybackStartedAt.millisecondsSinceEpoch == 0) {
      if (_isSpeaking) setState(() => _isSpeaking = false);
      return;
    }
    final elapsedMs = now.difference(_promptPlaybackStartedAt).inMilliseconds;
    final promptRms = _currentRMS.clamp(-80.0, -10.0);
    void updatePromptEchoLevel() {
      if (_promptPlaybackLeakFloor <= -99.0) {
        _promptPlaybackLeakFloor = promptRms;
      } else if (promptRms > _promptPlaybackLeakFloor) {
        _promptPlaybackLeakFloor =
            (_promptPlaybackLeakFloor * 0.82) + (promptRms * 0.18);
      } else {
        _promptPlaybackLeakFloor =
            (_promptPlaybackLeakFloor * 0.985) + (promptRms * 0.015);
      }
    }

    if (elapsedMs < _minimumInterruptiblePromptMs) {
      updatePromptEchoLevel();
      _promptPlaybackLastRms = promptRms;
      _promptSpeechCandidateSince = null;
      if (_isSpeaking) setState(() => _isSpeaking = false);
      return;
    }

    final interruptThreshold = math.max(
      _vadThreshold + 18.0,
      (_promptPlaybackLeakFloor + 18.0).clamp(-30.0, -8.0),
    );
    final aboveEcho = promptRms - _promptPlaybackLeakFloor;
    final freshRise = promptRms - _promptPlaybackLastRms;
    final hasVoiceLikeRise =
        aboveEcho >= 18.0 &&
        (freshRise >= 8.0 || _promptSpeechCandidateSince != null);

    if (promptRms > interruptThreshold && hasVoiceLikeRise) {
      _lastVoiceLevelTime = now;
      _lastSpeechTime = now;
      _promptSpeechCandidateSince ??= now;
      if (!_isSpeaking) {
        setState(() => _isSpeaking = true);
      }
      if (now.difference(_promptSpeechCandidateSince!).inMilliseconds >=
          _promptSpeechConfirmationMs) {
        _interruptPromptPlaybackDueToSpeech();
      }
      _promptPlaybackLastRms = promptRms;
      return;
    }

    updatePromptEchoLevel();
    _promptPlaybackLastRms = promptRms;
    _promptSpeechCandidateSince = null;
    if (_isSpeaking) {
      setState(() => _isSpeaking = false);
    }
  }

  void _handleAmplitude(double rms) {
    _currentRMS = rms.isFinite ? rms : -100.0;
    final now = DateTime.now();

    if (_promptState == PromptState.PLAYING_AUDIO ||
        _promptState == PromptState.PROMPT_REPEAT) {
      _handlePromptPlaybackAmplitude(now);
      return;
    }

    if (now.isBefore(_ignorePromptAudioUntil)) {
      if (_isSpeaking) setState(() => _isSpeaking = false);
      return;
    }

    // Calibrate the room before making prompt decisions. Recognition still
    // receives these samples, so starting an ayah immediately is not clipped.
    if (now.difference(_listeningStartedAt).inMilliseconds < 1200) {
      final calibrated = _currentRMS.clamp(-80.0, -20.0);
      _noiseFloor = (_noiseFloor * 0.65) + (calibrated * 0.35);
      return;
    }

    final isLikelySilence =
        !_isSpeaking && _currentRMS < (_noiseFloor + 6).clamp(-55.0, -30.0);
    if (isLikelySilence) {
      _noiseFloor = (_noiseFloor * 0.92) + (_currentRMS * 0.08);
    }

    final adaptiveThreshold = (_noiseFloor + 8.0).clamp(-48.0, -18.0);
    final speechThreshold = adaptiveThreshold < _vadThreshold
        ? _vadThreshold
        : adaptiveThreshold;
    final aboveThreshold = _currentRMS > speechThreshold;

    if (aboveThreshold) {
      _lastVoiceLevelTime = now;
      _lastSpeechTime = now;
      if (_taraweehModeEnabled && !_hasRecitedSinceTaraweehStart) {
        _hasRecitedSinceTaraweehStart = true;
      }
      if (!_isSpeaking) {
        _addLog(
          '[SPEECH] detected '
          '(RMS: ${_currentRMS.toStringAsFixed(1)}, '
          'threshold: ${speechThreshold.toStringAsFixed(1)})',
        );
        _lastLoggedPauseSecond = -1;
        setState(() => _isSpeaking = true);
      }
    } else if (_isSpeaking &&
        now.difference(_lastVoiceLevelTime).inMilliseconds > 900) {
      _addLog('[SPEECH] ended');
      _addLog('[PAUSE] timer started');
      setState(() => _isSpeaking = false);
    }
  }

  Future<void> _stopListening({String? status}) async {
    if (_isStarting && !_isListening) {
      // Start failures still need full cleanup.
      _isStarting = false;
    }
    _addLog('Stopping microphone...');
    _cancelPrompt();
    _promptLoopTimer?.cancel();
    _promptLoopTimer = null;
    await _audioSub?.cancel();
    _audioSub = null;
    await _ampSub?.cancel();
    _ampSub = null;
    await _captureStateSub?.cancel();
    _captureStateSub = null;
    await widget.audioService.stop();

    if (mounted) {
      setState(() {
        _isListening = false;
        _isStarting = false;
        _isSpeaking = false;
        _currentRMS = -100;
        _progressCoverage = 0.0;
        _recognitionStatus = status ?? 'Ready. Tap the mic to start.';
      });
    }
    _addLog('Microphone stopped.');
  }

  Future<void> _promptTaraweehConfig() async {
    int surah = 1;
    int ayah = 1;
    final result = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
          ),
          title: const Text(
            'Start Taraweeh Mode',
            style: TextStyle(fontWeight: FontWeight.bold),
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                decoration: InputDecoration(
                  labelText: 'Starting Surah Number (1-114)',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                keyboardType: TextInputType.number,
                onChanged: (val) => surah = int.tryParse(val) ?? 1,
              ),
              const SizedBox(height: 12),
              TextField(
                decoration: InputDecoration(
                  labelText: 'Starting Ayah Number',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                keyboardType: TextInputType.number,
                onChanged: (val) => ayah = int.tryParse(val) ?? 1,
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              onPressed: () {
                _lastSurah = surah;
                _lastAyah = ayah;
                Navigator.pop(context, true);
              },
              child: const Text('Start'),
            ),
          ],
        );
      },
    );

    if (result == true) {
      if (!mounted) return;
      surah = surah.clamp(1, 114).toInt();
      ayah = ayah < 1 ? 1 : ayah;
      widget.engine.startTaraweeh(surah, ayah);
      setState(() {
        _taraweehModeEnabled = true;
        _currentSurah = surah;
        _currentAyah = ayah;
        _wordPosition = 0;
        _totalWords = 0;
        _wordCoverage = 0.0;
        _progressCoverage = 0.0;
        _sectionPosition = 0;
        _totalSections = 0;
        _sectionCoverage = 0.0;
        _needsCurrentAyahCorrection = false;
        _taraweehJustStarted = true;
        _hasRecitedSinceTaraweehStart = false;
        _taraweehSurahComplete = false;
      });
      _addLog('Taraweeh Mode configured for $surah:$ayah');
      if (!_isListening) {
        await _startListening();
      }
    }
  }

  void _stopTaraweeh() {
    widget.engine.stopTaraweeh();
    _cancelPrompt();
    setState(() {
      _taraweehModeEnabled = false;
      _taraweehJustStarted = false;
      _hasRecitedSinceTaraweehStart = false;
      _taraweehSurahComplete = false;
      _progressCoverage = 0.0;
      _needsCurrentAyahCorrection = false;
    });
    _addLog('Taraweeh Mode stopped');
  }

  void _openSettings() {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => SettingsScreen(
          engine: widget.engine,
          audioService: widget.audioService,
          onSaved: () async {
            _addLog('Settings updated.');
            if (widget.engine.supportsRemoteBackend) {
              _addLog('Reconnecting to backend...');
              await widget.engine.disconnect();
              await _initEngine();
            }
            await _loadSettings();
          },
        ),
      ),
    );
  }

  // ── Debug Screen ──────────────────────────────────────────────────────────

  Widget _buildDebugScreen() {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text(
          'Diagnostics & Debug',
          style: TextStyle(color: Colors.yellow),
        ),
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
            const Text(
              '--- STATE ---',
              style: TextStyle(
                color: Colors.yellow,
                fontWeight: FontWeight.bold,
                fontSize: 18,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Recording: ${_isListening ? "YES" : "NO"}',
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
            Text(
              'Recognition Ready: ${widget.engine.isReady ? "YES" : "NO"}',
              style: TextStyle(
                color: widget.engine.isReady
                    ? Colors.greenAccent
                    : Colors.orangeAccent,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              'Recognition Status: $_recognitionStatus',
              style: const TextStyle(color: Colors.white70, fontSize: 14),
            ),
            Text(
              'Speech Detected: ${_isSpeaking ? "TRUE" : "FALSE"}',
              style: const TextStyle(
                color: Colors.cyanAccent,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              'Audio RMS: ${_currentRMS.toStringAsFixed(2)} dB | Noise Floor: ${_noiseFloor.toStringAsFixed(1)} dB',
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
            Text(
              'Tracking State: $_trackingMode',
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
            Text(
              'Confidence: ${(_confidence * 100).toStringAsFixed(0)}%',
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
            const SizedBox(height: 16),
            const Text(
              '--- PROMPT MACHINE ---',
              style: TextStyle(
                color: Colors.yellow,
                fontWeight: FontWeight.bold,
                fontSize: 18,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Prompt State: ${_promptState.name}',
              style: const TextStyle(
                color: Colors.orangeAccent,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              'Prompt Message: $_promptStateMessage',
              style: const TextStyle(color: Colors.white70, fontSize: 14),
            ),
            Text(
              'Pause Timer: ${(_isListening ? DateTime.now().difference(_lastSpeechTime).inMilliseconds / 1000 : 0.0).toStringAsFixed(1)} s',
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
            Text(
              'Anchor Surah: ${_currentSurah > 0 ? _currentSurah : _lastSurah}',
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
            Text(
              'Anchor Ayah: ${_currentAyah > 0 ? _currentAyah : _lastAyah}',
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
            Text(
              'Repeat Count: $_promptRepeatCount / $_promptMaxRepeats',
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
            const SizedBox(height: 24),
            const Text(
              '--- DEVELOPER CONTROLS ---',
              style: TextStyle(
                color: Colors.yellow,
                fontWeight: FontWeight.bold,
                fontSize: 18,
              ),
            ),
            const SizedBox(height: 8),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.redAccent,
              ),
              onPressed: () {
                _setPromptState(
                  PromptState.PROMPT_DISPLAYED,
                  'DEV FORCE TRIGGER',
                );
                int anchorSurah = _currentSurah > 0
                    ? _currentSurah
                    : _lastSurah;
                int anchorAyah = _currentAyah > 0 ? _currentAyah : _lastAyah;
                if (anchorSurah > 0 && anchorAyah > 0) {
                  final promptCurrent = _shouldCorrectCurrentAyah(anchorAyah);
                  setState(() {
                    _assistedSurah = anchorSurah;
                    _assistedAyah = promptCurrent ? anchorAyah : anchorAyah + 1;
                    _assistedWordIndex = promptCurrent ? _wordPosition : 0;
                    _promptRepeatCount = 1;
                  });
                  _setPromptState(PromptState.PLAYING_AUDIO, 'PLAYING AUDIO');
                  widget.engine.sendAssistedPrompt(
                    _assistedSurah,
                    _assistedAyah,
                    count: _promptAyahCount,
                    wordIndex: _assistedWordIndex,
                  );
                }
              },
              child: const Text(
                'Trigger Prompt Now',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(height: 8),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blueAccent,
              ),
              onPressed: () {
                setState(() {
                  _lastSurah = _currentSurah > 0 ? _currentSurah : 1;
                  _lastAyah = _currentAyah > 0 ? _currentAyah : 1;
                  _trackingMode = 'LOCKED (DEV)';
                  _confidence = 0.99;
                });
              },
              child: const Text(
                'Force Lock Current Position',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
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
              child: const Text(
                'Reset Position',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              '--- LOGS ---',
              style: TextStyle(
                color: Colors.yellow,
                fontWeight: FontWeight.bold,
                fontSize: 18,
              ),
            ),
            const SizedBox(height: 8),
            ..._logs
                .take(50)
                .map(
                  (log) => Padding(
                    padding: const EdgeInsets.only(bottom: 4),
                    child: Text(
                      log,
                      style: TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 12,
                        color: log.contains('[ERROR]')
                            ? Colors.redAccent
                            : (log.contains('[PROMPT]')
                                  ? Colors.orangeAccent
                                  : (log.contains('[SPEECH]')
                                        ? Colors.cyanAccent
                                        : Colors.white70)),
                      ),
                    ),
                  ),
                ),
          ],
        ),
      ),
    );
  }

  // ── Main Build ────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    if (_showDebugScreen) return _buildDebugScreen();
    return _buildMainScreen();
  }

  Color get _connectionColor {
    if (_connectionStatus == 'Connected') return Colors.green;
    if (_connectionStatus.contains('onnect')) return Colors.orange;
    return Colors.red;
  }

  Widget _buildMainScreen() {
    final bool isActive = _currentSurah > 0;
    final bool isPromptVisible =
        _promptModeEnabled &&
        _promptState != PromptState.DISABLED &&
        _promptState != PromptState.WAITING_FOR_LOCK &&
        _promptState != PromptState.PROMPT_READY &&
        _promptState != PromptState.IDLE;

    return Scaffold(
      backgroundColor: const Color(0xFFF8F5FF),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1B1464),
        elevation: 0,
        title: Row(
          children: [
            Container(
              width: 10,
              height: 10,
              decoration: BoxDecoration(
                color: _connectionColor,
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: _connectionColor.withOpacity(0.5),
                    blurRadius: 6,
                    spreadRadius: 2,
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            Text(
              _debugModeEnabled ? 'Live Recitation (Debug)' : 'Live Recitation',
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 18,
              ),
            ),
          ],
        ),
        actions: [
          if (_debugModeEnabled)
            IconButton(
              icon: const Icon(Icons.bug_report, color: Colors.orange),
              onPressed: () => setState(() => _showDebugScreen = true),
              tooltip: 'Open Debug Screen',
            ),
          IconButton(
            icon: const Icon(Icons.settings, color: Colors.white70),
            onPressed: _openSettings,
            tooltip: 'Settings',
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            // ── Status / Control Strip ────────────────────────────────────
            Container(
              color: const Color(0xFF1B1464),
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: Row(
                children: [
                  // VAD indicator
                  _buildVadIndicator(),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      _recognitionStatus,
                      style: TextStyle(
                        color: _isListening
                            ? Colors.greenAccent
                            : Colors.white54,
                        fontSize: 13,
                        fontWeight: FontWeight.w500,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  // Taraweeh chip
                  GestureDetector(
                    onTap: () {
                      if (_taraweehModeEnabled) {
                        _stopTaraweeh();
                      } else {
                        _promptTaraweehConfig();
                      }
                    },
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 300),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 5,
                      ),
                      decoration: BoxDecoration(
                        color: _taraweehModeEnabled
                            ? Colors.greenAccent.withOpacity(0.2)
                            : Colors.white12,
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                          color: _taraweehModeEnabled
                              ? Colors.greenAccent
                              : Colors.white24,
                        ),
                      ),
                      child: Text(
                        _taraweehModeEnabled ? 'Taraweeh ON' : 'Taraweeh',
                        style: TextStyle(
                          color: _taraweehModeEnabled
                              ? Colors.greenAccent
                              : Colors.white54,
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  // Prompt chip
                  GestureDetector(
                    onTap: () {
                      setState(() {
                        _promptModeEnabled = !_promptModeEnabled;
                        if (_promptModeEnabled) _loadSettings();
                      });
                    },
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 300),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 5,
                      ),
                      decoration: BoxDecoration(
                        color: _promptModeEnabled
                            ? Colors.orange.withOpacity(0.2)
                            : Colors.white12,
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                          color: _promptModeEnabled
                              ? Colors.orange
                              : Colors.white24,
                        ),
                      ),
                      child: Text(
                        _promptModeEnabled ? 'Audio Prompt ON' : 'Audio Prompt',
                        style: TextStyle(
                          color: _promptModeEnabled
                              ? Colors.orange
                              : Colors.white54,
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ),
                  if (_isMutashabihat) ...[
                    const SizedBox(width: 6),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 5,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.orange.withOpacity(0.3),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: Colors.orange),
                      ),
                      child: const Text(
                        '⚠️ Similar',
                        style: TextStyle(
                          color: Colors.orange,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ),

            // ── Main Content ──────────────────────────────────────────────
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    // Ayah display card
                    _buildAyahCard(isActive),

                    const SizedBox(height: 16),

                    // Prompt card
                    if (isPromptVisible) _buildPromptCard(),

                    if (_debugModeEnabled && _trackingMode.isNotEmpty) ...[
                      const SizedBox(height: 12),
                      _buildDebugStrip(),
                    ],
                  ],
                ),
              ),
            ),

            // ── Bottom Mic Button ────────────────────────────────────────
            _buildMicButton(),
          ],
        ),
      ),
    );
  }

  Widget _buildVadIndicator() {
    if (!_isListening) {
      return const Icon(Icons.mic_off, color: Colors.white38, size: 20);
    }
    return AnimatedBuilder(
      animation: _pulseAnimation,
      builder: (context, child) {
        double level = _isSpeaking ? _pulseAnimation.value : 0.4;
        return Container(
          width: 20,
          height: 20,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: _isSpeaking
                ? Colors.greenAccent.withOpacity(level)
                : Colors.white24,
            boxShadow: _isSpeaking
                ? [
                    BoxShadow(
                      color: Colors.greenAccent.withOpacity(0.6),
                      blurRadius: 8,
                      spreadRadius: 2,
                    ),
                  ]
                : [],
          ),
          child: Icon(
            Icons.mic,
            size: 12,
            color: _isSpeaking ? Colors.white : Colors.white38,
          ),
        );
      },
    );
  }

  Widget _buildAyahCard(bool isActive) {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF1B1464), Color(0xFF2D3A9B)],
        ),
        borderRadius: BorderRadius.circular(24),
        boxShadow: const [
          BoxShadow(
            color: Color(0x331B1464),
            blurRadius: 20,
            offset: Offset(0, 8),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          children: [
            if (!isActive) ...[
              const Icon(Icons.menu_book, color: Colors.white38, size: 64),
              const SizedBox(height: 16),
              const Text(
                'No verse detected yet',
                style: TextStyle(color: Colors.white54, fontSize: 18),
              ),
              const SizedBox(height: 8),
              const Text(
                'Press the mic button and start reciting',
                style: TextStyle(color: Colors.white38, fontSize: 14),
                textAlign: TextAlign.center,
              ),
            ] else ...[
              // Surah name
              Text(
                _surahNameEn,
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                  letterSpacing: 2,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                '$_surahNameEn:$_currentAyah',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),
              if (_surahNameAr.isNotEmpty)
                Text(
                  _surahNameAr,
                  textDirection: TextDirection.rtl,
                  style: const TextStyle(color: Colors.white70, fontSize: 16),
                ),
              if (_totalSections > 0) ...[
                const SizedBox(height: 4),
                Text(
                  'Section $_sectionPosition of $_totalSections',
                  style: const TextStyle(color: Colors.white54, fontSize: 13),
                ),
              ],
              const SizedBox(height: 14),

              if (_prevAyahText.isNotEmpty) ...[
                _buildContextAyah(
                  label: 'Previous ayah',
                  text: _prevAyahText,
                  color: Colors.white54,
                ),
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 10),
                  child: Divider(color: Colors.white24, height: 1),
                ),
              ],

              if (_currentAyahText.isNotEmpty) ...[
                const Text(
                  'Current ayah',
                  style: TextStyle(
                    color: Colors.yellowAccent,
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 4),
                _buildCurrentAyahText(),
              ],

              if (_nextAyahText.isNotEmpty) ...[
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 10),
                  child: Divider(color: Colors.white24, height: 1),
                ),
                _buildContextAyah(
                  label: 'Next ayah',
                  text: _nextAyahText,
                  color: Colors.white70,
                ),
              ],

              if (_isPromptConfirmed) ...[
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.symmetric(
                    vertical: 8,
                    horizontal: 20,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.greenAccent.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.greenAccent),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.check_circle,
                        color: Colors.greenAccent,
                        size: 20,
                      ),
                      SizedBox(width: 8),
                      Text(
                        'Recovery Successful',
                        style: TextStyle(
                          color: Colors.greenAccent,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildPromptCard() {
    Color cardColor;
    Color borderColor;
    Widget statusContent;

    if (_promptState == PromptState.PLAYING_AUDIO ||
        _promptState == PromptState.PROMPT_REPEAT) {
      cardColor = const Color(0xFFFFF3E0);
      borderColor = Colors.orange;
      statusContent = Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.volume_up, color: Colors.orange, size: 24),
          const SizedBox(width: 8),
          Text(
            _promptStateMessage.contains('MISTAKE DETECTED')
                ? 'Mistake detected - correcting'
                : _promptState == PromptState.PROMPT_REPEAT
                ? 'Repeating luqmah'
                : 'Playing luqmah',
            style: const TextStyle(
              color: Colors.orange,
              fontWeight: FontWeight.bold,
              fontSize: 15,
            ),
          ),
        ],
      );
    } else if (_promptState == PromptState.PAUSE_TIMER_RUNNING) {
      cardColor = const Color(0xFFF3F4FF);
      borderColor = const Color(0xFF2D3A9B);
      statusContent = Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.timer, color: Color(0xFF2D3A9B), size: 20),
          const SizedBox(width: 8),
          Text(
            _promptStateMessage,
            style: const TextStyle(color: Color(0xFF2D3A9B), fontSize: 14),
          ),
        ],
      );
    } else if (_promptState == PromptState.PROMPT_EXPIRED) {
      cardColor = const Color(0xFFFCE4EC);
      borderColor = Colors.redAccent;
      statusContent = const Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.hourglass_empty, color: Colors.redAccent, size: 20),
          SizedBox(width: 8),
          Text(
            'Resume reciting when ready',
            style: TextStyle(color: Colors.redAccent, fontSize: 14),
          ),
        ],
      );
    } else {
      cardColor = const Color(0xFFFFF8E1);
      borderColor = Colors.amber;
      statusContent = Text(
        _promptStateMessage,
        textAlign: TextAlign.center,
        style: const TextStyle(color: Colors.amber, fontSize: 14),
      );
    }

    final showPromptAyah = _promptAyahText.isNotEmpty && _assistedAyah > 0;
    final content = Column(
      children: [
        statusContent,
        if (showPromptAyah) ...[
          const SizedBox(height: 10),
          Text(
            _promptAudioVerses.isEmpty
                ? 'Luqmah $_assistedSurah:$_assistedAyah'
                : 'Luqmah ${_promptAudioVerses.map((verse) {
                    final label = '${verse['surah']}:${verse['ayah']}';
                    final strategy = verse['prompt_strategy']?.toString();
                    final startWord = verse['start_word'];
                    if (strategy == 'phrase_boundary' && startWord is int && startWord > 1) {
                      return '$label from phrase';
                    }
                    return label;
                  }).join(' + ')}',
            style: TextStyle(
              color: borderColor,
              fontSize: 12,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            _promptAyahText,
            textAlign: TextAlign.center,
            textDirection: TextDirection.rtl,
            style: const TextStyle(
              fontSize: 20,
              height: 1.7,
              fontFamily: 'Amiri',
              color: Color(0xFF1B1464),
            ),
          ),
        ],
      ],
    );

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: borderColor, width: 2),
        boxShadow: [
          BoxShadow(
            color: borderColor.withOpacity(0.2),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: content,
    );
  }

  Widget _buildContextAyah({
    required String label,
    required String text,
    required Color color,
  }) {
    return Column(
      children: [
        Text(
          label,
          style: TextStyle(
            color: color,
            fontSize: 11,
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          text,
          textAlign: TextAlign.center,
          textDirection: TextDirection.rtl,
          style: TextStyle(
            color: color,
            fontSize: 18,
            height: 1.6,
            fontFamily: 'Amiri',
          ),
        ),
      ],
    );
  }

  Widget _buildDebugStrip() {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.grey[900],
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Tracking: $_trackingMode | Window: $_searchWindow',
            style: const TextStyle(color: Colors.yellow, fontSize: 11),
          ),
          Text(
            'Fallback: $_fallbackCount | Prompt: ${_promptState.name}',
            style: const TextStyle(color: Colors.orange, fontSize: 11),
          ),
          if (_transcript.isNotEmpty)
            Text(
              'Heard: $_transcript',
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(color: Colors.white70, fontSize: 11),
            ),
          if (_assistedAyah > 0)
            Text(
              'Prompted Ayah: $_assistedSurah:$_assistedAyah | Repeat: $_promptRepeatCount/$_promptMaxRepeats',
              style: const TextStyle(color: Colors.greenAccent, fontSize: 11),
            ),
        ],
      ),
    );
  }

  Widget _buildMicButton() {
    final bool canPress = !_isStarting;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: GestureDetector(
        onTap: canPress ? _toggleListening : null,
        child: AnimatedBuilder(
          animation: _pulseAnimation,
          builder: (context, child) {
            double scale = (_isListening && _isSpeaking)
                ? _pulseAnimation.value
                : 1.0;
            return Transform.scale(
              scale: scale,
              child: Container(
                width: 72,
                height: 72,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(
                    colors: _isListening
                        ? [Colors.red.shade400, Colors.red.shade700]
                        : canPress
                        ? [const Color(0xFF2D3A9B), const Color(0xFF1B1464)]
                        : [Colors.grey.shade400, Colors.grey.shade600],
                  ),
                  boxShadow: [
                    BoxShadow(
                      color:
                          (_isListening ? Colors.red : const Color(0xFF2D3A9B))
                              .withOpacity(0.4),
                      blurRadius: 20,
                      offset: const Offset(0, 8),
                    ),
                  ],
                ),
                child: _isStarting
                    ? const Center(
                        child: SizedBox(
                          width: 24,
                          height: 24,
                          child: CircularProgressIndicator(
                            color: Colors.white,
                            strokeWidth: 2,
                          ),
                        ),
                      )
                    : Icon(
                        _isListening ? Icons.stop_rounded : Icons.mic_rounded,
                        color: Colors.white,
                        size: 34,
                      ),
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _buildCurrentAyahText() {
    if (_currentAyahText.isEmpty) return const SizedBox();

    List<String> words = _currentAyahText.split(' ');
    return Directionality(
      textDirection: TextDirection.rtl,
      child: Wrap(
        alignment: WrapAlignment.center,
        spacing: 8.0,
        runSpacing: 8.0,
        children: List.generate(words.length, (index) {
          int targetIndex = _wordPosition - 1;
          if (targetIndex >= words.length) targetIndex = words.length - 1;
          if (targetIndex < 0) targetIndex = 0;

          Color color = Colors.white60;
          FontWeight weight = FontWeight.normal;
          double fontSize = 22;

          if (_wordPosition == 0) {
            color = Colors.white;
          } else if (index < targetIndex) {
            color = Colors.white38; // Already spoken (dimmed)
          } else if (index == targetIndex) {
            color = Colors.yellowAccent; // Currently active word
            weight = FontWeight.bold;
            fontSize = 25;
          }

          return Text(
            words[index],
            style: TextStyle(
              fontSize: fontSize,
              fontFamily: 'Amiri',
              color: color,
              fontWeight: weight,
            ),
          );
        }),
      ),
    );
  }
}
