import 'dart:math' as math;
import 'dart:typed_data';

class ProcessedAudioChunk {
  final Uint8List data;
  final double rmsDb;

  const ProcessedAudioChunk({required this.data, required this.rmsDb});
}

/// Gentle speech-focused PCM processing that avoids device-specific Android
/// audio effects. It removes DC rumble and raises quiet recitation without
/// hard-gating soft consonants or breathy ayah openings.
class RecitationAudioProcessor {
  double _previousInput = 0;
  double _previousOutput = 0;
  double _gain = 1;

  void reset() {
    _previousInput = 0;
    _previousOutput = 0;
    _gain = 1;
  }

  ProcessedAudioChunk process(Uint8List input, {bool enhance = true}) {
    final usableLength = input.lengthInBytes - (input.lengthInBytes % 2);
    if (usableLength == 0) {
      return ProcessedAudioChunk(data: Uint8List(0), rmsDb: -100);
    }

    final inputData = ByteData.sublistView(input, 0, usableLength);
    final samples = Float64List(usableLength ~/ 2);
    double sumSquares = 0;

    for (var i = 0; i < samples.length; i++) {
      final sample = inputData.getInt16(i * 2, Endian.little) / 32768.0;
      final filtered = sample - _previousInput + (0.995 * _previousOutput);
      _previousInput = sample;
      _previousOutput = filtered;
      samples[i] = filtered;
      sumSquares += filtered * filtered;
    }

    final rms = math.sqrt(sumSquares / samples.length);
    final rmsDb = rms > 0 ? 20 * math.log(rms) / math.ln10 : -100.0;

    if (enhance && rmsDb > -58) {
      const targetRmsDb = -22.0;
      final desiredGain = math
          .pow(10, (targetRmsDb - rmsDb) / 20)
          .toDouble()
          .clamp(1.0, 3.2);
      final smoothing = desiredGain > _gain ? 0.12 : 0.04;
      _gain += (desiredGain - _gain) * smoothing;
    } else {
      _gain += (1.0 - _gain) * 0.08;
    }

    final output = Uint8List(usableLength);
    final outputData = ByteData.sublistView(output);
    for (var i = 0; i < samples.length; i++) {
      var sample = samples[i] * (enhance ? _gain : 1.0);
      sample = sample.clamp(-0.98, 0.98);
      outputData.setInt16(i * 2, (sample * 32767).round(), Endian.little);
    }

    return ProcessedAudioChunk(data: output, rmsDb: rmsDb);
  }
}
