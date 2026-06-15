import 'package:flutter_test/flutter_test.dart';
import 'package:alfatih_mobile/audio/luqmah_reciters.dart';

void main() {
  test('luqmah qari catalog has a valid unique default and choices', () {
    final folders = LuqmahReciters.all
        .map((reciter) => reciter.folder)
        .toList();

    expect(LuqmahReciters.all.length, greaterThanOrEqualTo(5));
    expect(folders.toSet().length, folders.length);
    expect(folders, contains(LuqmahReciters.defaultFolder));
    expect(
      LuqmahReciters.fromFolder('not-a-real-folder').folder,
      LuqmahReciters.defaultFolder,
    );
  });
}
