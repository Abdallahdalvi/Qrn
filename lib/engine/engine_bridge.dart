import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';

class TarteelEngineBridge {
  HeadlessInAppWebView? _headlessWebView;
  final InAppLocalhostServer _localhostServer = InAppLocalhostServer(
    port: 8080,
    documentRoot: 'assets',
  );
  
  final _eventsController = StreamController<Map<String, dynamic>>.broadcast();
  Stream<Map<String, dynamic>> get onEvent => _eventsController.stream;

  final List<Map<String, dynamic>> _eventHistory = [];
  List<Map<String, dynamic>> get eventHistory => List.unmodifiable(_eventHistory);

  bool _isReady = false;
  bool get isReady => _isReady;

  void _emitEvent(Map<String, dynamic> event) {
    _eventHistory.add(event);
    _eventsController.add(event);
  }

  Future<void> initialize() async {
    try {
      _emitEvent({'type': 'status', 'message': 'Starting local server...'});
      await _localhostServer.start();
      _emitEvent({'type': 'status', 'message': 'Local server started successfully.'});
    } catch (e, stack) {
      _emitEvent({
        'type': 'error',
        'message': 'Failed to start local server: $e\nStacktrace: $stack'
      });
      return;
    }

    try {
      _emitEvent({'type': 'status', 'message': 'Starting Headless WebView...'});
      _headlessWebView = HeadlessInAppWebView(
        initialUrlRequest: URLRequest(url: WebUri("http://localhost:8080/web/index.html")),
        initialSettings: InAppWebViewSettings(
          javaScriptEnabled: true,
          allowFileAccessFromFileURLs: true,
          allowUniversalAccessFromFileURLs: true,
        ),
        onWebViewCreated: (controller) {
          _emitEvent({'type': 'status', 'message': 'WebView created, adding JS handlers...'});
          controller.addJavaScriptHandler(
            handlerName: 'onEngineEvent',
            callback: (args) {
              if (args.isNotEmpty) {
                final event = args[0] as Map<String, dynamic>;
                if (event['type'] == 'ready') {
                  _isReady = true;
                }
                _emitEvent(event);
              }
            },
          );
        },
        onLoadStart: (controller, url) {
          _emitEvent({'type': 'status', 'message': 'WebView load started: $url'});
        },
        onLoadStop: (controller, url) async {
          _emitEvent({'type': 'status', 'message': 'WebView load stopped: $url'});
        },
        onConsoleMessage: (controller, consoleMessage) {
          if (kDebugMode) {
            print('[WebEngine] ${consoleMessage.messageLevel}: ${consoleMessage.message}');
          }
          final msg = consoleMessage.message;
          if (consoleMessage.messageLevel == ConsoleMessageLevel.ERROR) {
            _emitEvent({'type': 'error', 'message': 'JS Error: $msg'});
          } else {
            _emitEvent({'type': 'status', 'message': 'JS Log: $msg'});
          }
        },
        onReceivedError: (controller, request, error) {
          _emitEvent({'type': 'error', 'message': 'WebView Error: ${error.description}'});
        },
        onReceivedHttpError: (controller, request, errorResponse) {
          _emitEvent({'type': 'error', 'message': 'HTTP Error: ${request.url} - ${errorResponse.statusCode}'});
        },
      );

      await _headlessWebView?.run();
      _emitEvent({'type': 'status', 'message': 'Headless WebView running.'});
    } catch (e, stack) {
      _emitEvent({
        'type': 'error',
        'message': 'Failed to launch Headless WebView: $e\nStacktrace: $stack'
      });
    }
  }

  Future<void> processAudioChunk(Uint8List pcmFloat32Bytes) async {
    if (!_isReady || _headlessWebView == null) return;
    
    // Encode as Base64 to send across the JS bridge
    String base64Data = base64Encode(pcmFloat32Bytes);
    
    await _headlessWebView?.webViewController?.evaluateJavascript(
      source: 'window.processAudioChunk("$base64Data");'
    );
  }

  Future<void> dispose() async {
    await _headlessWebView?.dispose();
    await _localhostServer.close();
    await _eventsController.close();
  }
}
