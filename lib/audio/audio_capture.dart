import 'dart:async';
import 'dart:typed_data';
import 'package:record/record.dart';
import 'package:permission_handler/permission_handler.dart';
import '../core/global_logger.dart';

class AudioCaptureService {
  final AudioRecorder _audioRecorder = AudioRecorder();
  StreamSubscription<RecordState>? _recordSub;
  StreamSubscription<Amplitude>? _ampSub;
  StreamSubscription<Uint8List>? _streamSub;
  
  final _audioStreamController = StreamController<Uint8List>.broadcast();
  Stream<Uint8List> get onAudioChunk => _audioStreamController.stream;

  final _amplitudeStreamController = StreamController<Amplitude>.broadcast();
  Stream<Amplitude> get onAmplitude => _amplitudeStreamController.stream;

  Future<bool> requestPermission() async {
    globalLogger.log('MIC PERMISSION CHECK');
    try {
      final status = await Permission.microphone.request();
      globalLogger.log('MIC PERMISSION RESULT: $status');
      
      if (status.isGranted) {
        final recordPerm = await _audioRecorder.hasPermission();
        globalLogger.log('MIC PERMISSION RESULT (Recorder): $recordPerm');
        return recordPerm;
      }
      return false;
    } catch (e, stack) {
      globalLogger.logError('AudioCaptureService: Exception in requestPermission: [${e.runtimeType}] $e', stack);
      return false;
    }
  }

  Future<void> start() async {
    globalLogger.log('AUDIO RECORDER INITIALIZATION');
    try {
      final hasPerm = await _audioRecorder.hasPermission();
      if (!hasPerm) {
        globalLogger.log('AUDIO RECORDER INITIALIZATION FAILED (No permission)', isError: true);
        return;
      }
      
      globalLogger.log('AUDIO RECORDER START');
      final stream = await _audioRecorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: 16000,
          numChannels: 1,
          echoCancel: true,
          autoGain: true,
          noiseSuppress: true,
        ),
      );
      
      _streamSub = stream.listen((data) {
        globalLogger.log('AUDIO CHUNK GENERATED');
        globalLogger.log('AUDIO CHUNK SIZE: ${data.length}');
        _audioStreamController.add(data);
      }, onError: (err, stack) {
        globalLogger.logError('AudioCaptureService Stream Error: [${err.runtimeType}] $err', stack);
      }, onDone: () {
        globalLogger.log('AudioCaptureService: Stream closed.');
      });

      _ampSub = _audioRecorder.onAmplitudeChanged(const Duration(milliseconds: 100)).listen((amp) {
        _amplitudeStreamController.add(amp);
      });
    } catch (e, stack) {
      globalLogger.logError('AUDIO RECORDER INITIALIZATION EXCEPTION: [${e.runtimeType}] $e', stack);
    }
  }

  Future<void> stop() async {
    globalLogger.log('AudioCaptureService: stop() called');
    try {
      await _streamSub?.cancel();
      await _audioRecorder.stop();
      globalLogger.log('AudioCaptureService: Stopped successfully.');
    } catch (e, stack) {
      globalLogger.logError('AudioCaptureService: Exception in stop(): $e', stack);
    }
  }

  Future<void> dispose() async {
    try {
      await stop();
      await _recordSub?.cancel();
      await _ampSub?.cancel();
      await _audioRecorder.dispose();
      await _audioStreamController.close();
      globalLogger.log('AudioCaptureService: Disposed successfully.');
    } catch (e, stack) {
      globalLogger.logError('AudioCaptureService: Exception in dispose(): $e', stack);
    }
  }
}
