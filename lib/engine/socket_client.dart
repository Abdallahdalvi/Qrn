import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:shared_preferences/shared_preferences.dart';
import '../core/global_logger.dart';

class TarteelSocketClient {
  String _serverIp = '10.0.2.2';
  String _serverPort = '8000';

  String get serverIp => _serverIp;
  String get serverPort => _serverPort;
  String get serverHost => '$_serverIp:$_serverPort';
  String get connectionUrl => 'ws://$serverHost/ws/recitation';

  WebSocket? _socket;
  
  final _eventsController = StreamController<Map<String, dynamic>>.broadcast();
  Stream<Map<String, dynamic>> get onEvent => _eventsController.stream;

  final List<Map<String, dynamic>> _eventHistory = [];
  List<Map<String, dynamic>> get eventHistory => List.unmodifiable(_eventHistory);

  bool _isReady = false;
  bool get isReady => _isReady;

  Timer? _reconnectTimer;
  Timer? _heartbeatTimer;
  int _reconnectAttempts = 0;
  bool _intentionalDisconnect = false;

  final List<int> _audioBuffer = [];
  static const int MAX_BUFFER_SIZE = 16000 * 5; // Max 5 seconds of 16kHz PCM16 buffer

  bool _isTaraweehActive = false;
  int _taraweehSurah = 1;
  int _taraweehAyah = 1;

  TarteelSocketClient() {
    loadSettings();
  }

  void _emitEvent(Map<String, dynamic> event) {
    _eventHistory.add(event);
    _eventsController.add(event);
  }

  Future<void> loadSettings() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      _serverIp = prefs.getString('server_ip') ?? '10.0.2.2';
      _serverPort = prefs.getString('server_port') ?? '8000';
    } catch (e) {
      // Ignore prefs error
    }
  }

  Future<void> saveSettings(String ip, String port) async {
    _serverIp = ip;
    _serverPort = port;
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('server_ip', ip);
      await prefs.setString('server_port', port);
    } catch (e) {
      // Ignore prefs error
    }
  }

  Future<bool> testConnection(String ip, String port) async {
    try {
      final url = 'ws://$ip:$port/ws/recitation';
      final testSocket = await WebSocket.connect(url).timeout(const Duration(seconds: 3));
      await testSocket.close();
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<void> initialize() async {
    _intentionalDisconnect = false;
    _reconnectAttempts = 0;
    await _connect();
  }

  Future<void> _connect() async {
    if (_socket != null && _socket!.readyState == WebSocket.open) {
      return;
    }

    await loadSettings();
    _isReady = false;
    _stopHeartbeat();
    _reconnectTimer?.cancel();

    try {
      final url = connectionUrl;
      final statusMsg = _reconnectAttempts > 0 ? 'Reconnecting...' : 'Connecting...';
      _emitEvent({'type': 'status', 'message': statusMsg});
      
      _socket = await WebSocket.connect(url).timeout(const Duration(seconds: 5));
      _isReady = true;
      _reconnectAttempts = 0;
      
      _emitEvent({'type': 'status', 'message': 'Connected to backend.'});
      _emitEvent({'type': 'ready', 'status': 'success'});
      
      if (_isTaraweehActive) {
        _socket!.add(jsonEncode({
          "type": "start_taraweeh",
          "surah": _taraweehSurah,
          "ayah": _taraweehAyah
        }));
        _emitEvent({'type': 'status', 'message': 'Restored Taraweeh state.'});
      }

      _flushAudioBuffer();
      _startHeartbeat();

      _socket!.listen(
        (data) {
          if (data is String) {
            try {
              final Map<String, dynamic> event = jsonDecode(data);
              // Ignore simple pong messages to keep logs clean
              if (event['type'] != 'pong') {
                _emitEvent(event);
              }
            } catch (e) {
              _emitEvent({'type': 'status', 'message': 'Received message: $data'});
            }
          }
        },
        onError: (err) {
          _handleDisconnect('WebSocket Error: $err');
        },
        onDone: () {
          _handleDisconnect('WebSocket connection closed by server.');
        },
      );
    } catch (e) {
      _handleDisconnect('Connection failed: $e');
    }
  }

  void _handleDisconnect(String reason) {
    _isReady = false;
    _socket?.close();
    _socket = null;
    _stopHeartbeat();
    
    _emitEvent({'type': 'error', 'message': reason});
    _emitEvent({'type': 'status', 'message': 'Disconnected'});
    
    if (!_intentionalDisconnect) {
      _scheduleReconnect();
    }
  }

  void _scheduleReconnect() {
    _reconnectTimer?.cancel();
    // Exponential backoff: 2s, 4s, 8s, 16s... max 30s
    final delaySeconds = (2 * (1 << _reconnectAttempts)).clamp(2, 30);
    _reconnectAttempts++;
    
    _emitEvent({'type': 'status', 'message': 'Waiting $delaySeconds seconds before reconnecting...'});
    
    _reconnectTimer = Timer(Duration(seconds: delaySeconds), () {
      _connect();
    });
  }

  void _startHeartbeat() {
    _heartbeatTimer?.cancel();
    _heartbeatTimer = Timer.periodic(const Duration(seconds: 10), (timer) {
      if (_socket != null && _isReady && _socket!.readyState == WebSocket.open) {
        _socket!.add(jsonEncode({'type': 'ping'}));
      }
    });
  }

  void _stopHeartbeat() {
    _heartbeatTimer?.cancel();
  }

  void _flushAudioBuffer() {
    if (_audioBuffer.isNotEmpty && _socket != null && _isReady) {
      globalLogger.log('Flushing ${_audioBuffer.length} buffered audio bytes...');
      _socket!.add(_audioBuffer);
      _audioBuffer.clear();
    }
  }

  void processAudioChunk(List<int> pcm16Bytes) {
    try {
      if (_socket != null && _isReady && _socket!.readyState == WebSocket.open) {
        globalLogger.log('WEBSOCKET SEND');
        _socket!.add(pcm16Bytes);
      } else {
        // Buffer audio if disconnected, drop if it exceeds MAX_BUFFER_SIZE
        if (_audioBuffer.length + pcm16Bytes.length <= MAX_BUFFER_SIZE) {
          _audioBuffer.addAll(pcm16Bytes);
        } else {
          // Slide buffer to make room
          final overflow = (_audioBuffer.length + pcm16Bytes.length) - MAX_BUFFER_SIZE;
          _audioBuffer.removeRange(0, overflow);
          _audioBuffer.addAll(pcm16Bytes);
        }
      }
    } catch (e, stack) {
      globalLogger.logError('WEBSOCKET EXCEPTION: [${e.runtimeType}] $e', stack);
    }
  }

  Future<void> disconnect() async {
    _intentionalDisconnect = true;
    _reconnectTimer?.cancel();
    _stopHeartbeat();
    _isReady = false;
    await _socket?.close();
    _socket = null;
    _emitEvent({'type': 'status', 'message': 'WebSocket connection intentionally closed.'});
  }

  void startTaraweeh(int surah, int ayah) {
    _isTaraweehActive = true;
    _taraweehSurah = surah;
    _taraweehAyah = ayah;
    if (_socket != null && _isReady && _socket!.readyState == WebSocket.open) {
      _socket!.add(jsonEncode({
        "type": "start_taraweeh",
        "surah": surah,
        "ayah": ayah
      }));
    }
  }

  void sendAssistedPrompt(int ayah) {
    if (_socket != null && _isReady && _socket!.readyState == WebSocket.open) {
      _socket!.add(jsonEncode({
        "type": "assisted_prompt",
        "ayah": ayah
      }));
    }
  }

  void clearAssistedPrompt() {
    if (_socket != null && _isReady && _socket!.readyState == WebSocket.open) {
      _socket!.add(jsonEncode({
        "type": "clear_assisted_prompt"
      }));
    }
  }

  Future<void> dispose() async {
    await disconnect();
    await _eventsController.close();
  }
}
