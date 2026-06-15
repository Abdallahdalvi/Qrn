import 'dart:async';
import 'dart:typed_data';
import 'package:record/record.dart';
import 'package:permission_handler/permission_handler.dart';
import '../core/global_logger.dart';
import 'recitation_audio_processor.dart';

enum AudioCaptureState { idle, starting, recording, stopping, failed }

class AudioCaptureService {
  final AudioRecorder _audioRecorder = AudioRecorder();
  final RecitationAudioProcessor _processor = RecitationAudioProcessor();
  StreamSubscription<RecordState>? _recordSub;
  StreamSubscription<Uint8List>? _streamSub;

  AudioCaptureState _state = AudioCaptureState.idle;
  AudioCaptureState get state => _state;
  bool get isRecording => _state == AudioCaptureState.recording;

  bool enhancementEnabled = true;
  String? lastError;

  final _audioStreamController = StreamController<Uint8List>.broadcast();
  Stream<Uint8List> get onAudioChunk => _audioStreamController.stream;

  final _amplitudeStreamController = StreamController<Amplitude>.broadcast();
  Stream<Amplitude> get onAmplitude => _amplitudeStreamController.stream;

  final _stateController = StreamController<AudioCaptureState>.broadcast();
  Stream<AudioCaptureState> get onStateChanged => _stateController.stream;

  void _setState(AudioCaptureState state) {
    _state = state;
    if (!_stateController.isClosed) {
      _stateController.add(state);
    }
  }

  Future<bool> requestPermission() async {
    globalLogger.log('MIC PERMISSION CHECK');
    try {
      final status = await Permission.microphone.request();
      globalLogger.log('MIC PERMISSION RESULT: $status');

      if (status.isGranted) {
        final recordPerm = await _audioRecorder.hasPermission(request: false);
        globalLogger.log('MIC PERMISSION RESULT (Recorder): $recordPerm');
        return recordPerm;
      }
      return false;
    } catch (e, stack) {
      globalLogger.logError(
        'AudioCaptureService: Exception in requestPermission: [${e.runtimeType}] $e',
        stack,
      );
      return false;
    }
  }

  Future<bool> start() async {
    if (_state == AudioCaptureState.recording) return true;
    if (_state == AudioCaptureState.starting ||
        _state == AudioCaptureState.stopping) {
      globalLogger.log(
        'AudioCaptureService: Recorder is busy ($_state).',
        isError: true,
      );
      return false;
    }

    globalLogger.log('AUDIO RECORDER INITIALIZATION');
    _setState(AudioCaptureState.starting);
    lastError = null;

    try {
      final hasPerm = await _audioRecorder.hasPermission(request: false);
      if (!hasPerm) {
        throw StateError('Microphone permission was not granted.');
      }

      await _streamSub?.cancel();
      _streamSub = null;
      await _recordSub?.cancel();
      _recordSub = null;
      if (await _audioRecorder.isRecording()) {
        await _audioRecorder.stop();
      }

      _processor.reset();
      _recordSub = _audioRecorder.onStateChanged().listen(
        (recordState) {
          if (recordState == RecordState.record) {
            _setState(AudioCaptureState.recording);
          } else if (recordState == RecordState.stop &&
              _state != AudioCaptureState.stopping) {
            _setState(AudioCaptureState.idle);
          }
        },
        onError: (Object error, StackTrace stack) {
          lastError = error.toString();
          _setState(AudioCaptureState.failed);
          globalLogger.logError('Native recorder state error: $error', stack);
        },
      );

      globalLogger.log('AUDIO RECORDER START');
      final stream = await _audioRecorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: 16000,
          numChannels: 1,
          autoGain: false,
          echoCancel: false,
          noiseSuppress: false,
          audioInterruption: AudioInterruptionMode.none,
          androidConfig: AndroidRecordConfig(
            manageBluetooth: false,
            audioSource: AndroidAudioSource.mic,
          ),
        ),
      );

      var lastAmplitudeAt = DateTime.fromMillisecondsSinceEpoch(0);
      _streamSub = stream.listen(
        (data) {
          final processed = _processor.process(
            data,
            enhance: enhancementEnabled,
          );
          if (processed.data.isNotEmpty && !_audioStreamController.isClosed) {
            _audioStreamController.add(processed.data);
          }

          final now = DateTime.now();
          if (now.difference(lastAmplitudeAt).inMilliseconds >= 100 &&
              !_amplitudeStreamController.isClosed) {
            lastAmplitudeAt = now;
            _amplitudeStreamController.add(
              Amplitude(current: processed.rmsDb, max: processed.rmsDb),
            );
          }
        },
        onError: (Object error, StackTrace stack) {
          lastError = error.toString();
          _setState(AudioCaptureState.failed);
          globalLogger.logError(
            'AudioCaptureService stream error: $error',
            stack,
          );
        },
        onDone: () {
          if (_state != AudioCaptureState.stopping) {
            _setState(AudioCaptureState.idle);
          }
          globalLogger.log('AudioCaptureService: Stream closed.');
        },
      );

      await Future<void>.delayed(const Duration(milliseconds: 120));
      if (!await _audioRecorder.isRecording()) {
        throw StateError(lastError ?? 'Android did not start the microphone.');
      }

      _setState(AudioCaptureState.recording);
      globalLogger.log('AudioCaptureService: Recording confirmed.');
      return true;
    } catch (e, stack) {
      lastError = e.toString();
      _setState(AudioCaptureState.failed);
      globalLogger.logError(
        'AUDIO RECORDER INITIALIZATION EXCEPTION: '
        '[${e.runtimeType}] $e',
        stack,
      );
      try {
        if (await _audioRecorder.isRecording()) {
          await _audioRecorder.stop().timeout(const Duration(seconds: 2));
        }
      } catch (_) {
        // Preserve the original recorder start error.
      }
      await _cleanupRecorder();
      return false;
    }
  }

  Future<void> stop() async {
    if (_state == AudioCaptureState.idle &&
        _streamSub == null &&
        _recordSub == null) {
      return;
    }
    if (_state == AudioCaptureState.stopping) return;

    globalLogger.log('AudioCaptureService: stop() called');
    _setState(AudioCaptureState.stopping);
    try {
      await _audioRecorder.stop().timeout(const Duration(seconds: 3));
      globalLogger.log('AudioCaptureService: Stopped successfully.');
    } catch (e, stack) {
      globalLogger.logError(
        'AudioCaptureService: Exception in stop(): $e',
        stack,
      );
    } finally {
      await _cleanupRecorder();
      _setState(AudioCaptureState.idle);
    }
  }

  Future<void> _cleanupRecorder() async {
    await _streamSub?.cancel();
    _streamSub = null;
    await _recordSub?.cancel();
    _recordSub = null;
    _processor.reset();
  }

  Future<void> dispose() async {
    try {
      await stop();
      await _audioRecorder.dispose();
      await _audioStreamController.close();
      await _amplitudeStreamController.close();
      await _stateController.close();
      globalLogger.log('AudioCaptureService: Disposed successfully.');
    } catch (e, stack) {
      globalLogger.logError(
        'AudioCaptureService: Exception in dispose(): $e',
        stack,
      );
    }
  }
}
