import 'dart:typed_data';

import 'package:alfatih_mobile/audio/recitation_audio_processor.dart';
import 'package:flutter_test/flutter_test.dart';

Uint8List pcm16(List<int> samples) {
  final bytes = Uint8List(samples.length * 2);
  final data = ByteData.sublistView(bytes);
  for (var i = 0; i < samples.length; i++) {
    data.setInt16(i * 2, samples[i], Endian.little);
  }
  return bytes;
}

int peak(Uint8List bytes) {
  final data = ByteData.sublistView(bytes);
  var result = 0;
  for (var offset = 0; offset < bytes.length; offset += 2) {
    final value = data.getInt16(offset, Endian.little).abs();
    if (value > result) result = value;
  }
  return result;
}

void main() {
  test('raises quiet recitation without clipping', () {
    final processor = RecitationAudioProcessor();
    final quiet = pcm16(List<int>.generate(320, (i) => i.isEven ? 500 : -500));

    final output = processor.process(quiet);

    expect(output.rmsDb, greaterThan(-45));
    expect(peak(output.data), greaterThan(peak(quiet)));
    expect(peak(output.data), lessThan(32767));
  });

  test('keeps silence quiet and supports odd byte chunks', () {
    final processor = RecitationAudioProcessor();

    final output = processor.process(Uint8List(321));

    expect(output.data.length, 320);
    expect(output.rmsDb, lessThanOrEqualTo(-100));
    expect(peak(output.data), 0);
  });

  test('reset prevents filter state leaking into a new recording', () {
    final processor = RecitationAudioProcessor();
    processor.process(pcm16(List<int>.filled(100, 1200)));

    processor.reset();
    final output = processor.process(pcm16(List<int>.filled(100, 0)));

    expect(peak(output.data), 0);
  });
}
