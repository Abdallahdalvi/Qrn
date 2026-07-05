import 'dart:async';

abstract class RecitationEngine {
  Stream<Map<String, dynamic>> get onEvent;
  List<Map<String, dynamic>> get eventHistory;
  bool get isReady;
  bool get supportsRemoteBackend;

  String get serverIp;
  String get serverPort;

  Future<void> initialize();
  Future<void> disconnect();
  Future<void> saveSettings(String ip, String port);
  Future<bool> testConnection(String ip, String port);

  void processAudioChunk(List<int> pcm16Bytes);
  void startTaraweeh(int surah, int ayah);
  void stopTaraweeh();
  void sendAssistedPrompt(
    int surah,
    int ayah, {
    int count = 1,
    int wordIndex = 0,
  });
  void discardPendingAudio();
  void clearAssistedPrompt();

  Future<void> dispose();
}
