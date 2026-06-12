import 'dart:async';

class GlobalLogger {
  static final GlobalLogger _instance = GlobalLogger._internal();
  factory GlobalLogger() => _instance;
  GlobalLogger._internal();

  final _logController = StreamController<String>.broadcast();
  Stream<String> get onLog => _logController.stream;

  final List<String> _history = [];
  List<String> get history => List.unmodifiable(_history);

  void log(String message, {bool isError = false}) {
    final prefix = isError ? '[ERROR] ' : '[INFO] ';
    final time = DateTime.now().toIso8601String().split('T').last;
    final formattedMessage = "$time: $prefix$message";
    
    print(formattedMessage); // Print to console
    
    _history.add(formattedMessage);
    if (_history.length > 200) {
      _history.removeAt(0); // Keep last 200 logs
    }
    
    _logController.add(formattedMessage);
  }

  void logError(Object error, [StackTrace? stackTrace]) {
    log('ERROR:\n$error\n\nSTACK TRACE:\n$stackTrace', isError: true);
  }
}

final globalLogger = GlobalLogger();
