import 'dart:async';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'core/global_logger.dart';
import 'engine/recitation_engine.dart';
import 'engine/socket_client.dart';
import 'audio/audio_capture.dart';
import 'ui/live_recitation_screen.dart';
import 'ui/surah_index_screen.dart';

void main() async {
  runZonedGuarded(
    () async {
      WidgetsFlutterBinding.ensureInitialized();

      // Catch Flutter framework errors
      FlutterError.onError = (FlutterErrorDetails details) {
        FlutterError.presentError(details);
        globalLogger.logError(details.exception, details.stack);
      };

      // Catch unhandled asynchronous errors
      PlatformDispatcher.instance.onError = (error, stack) {
        globalLogger.logError(error, stack);
        return true; // Prevent default crash behavior
      };

      final engine = TarteelSocketClient();
      final audioService = AudioCaptureService();

      runApp(AlfatihApp(engine: engine, audioService: audioService));
    },
    (error, stack) {
      globalLogger.logError(error, stack);
    },
  );
}

class AlfatihApp extends StatelessWidget {
  final RecitationEngine engine;
  final AudioCaptureService audioService;

  const AlfatihApp({Key? key, required this.engine, required this.audioService})
    : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Al-Fatih Alal-Imaam',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: MainAppScreen(engine: engine, audioService: audioService),
    );
  }
}

class MainAppScreen extends StatefulWidget {
  final RecitationEngine engine;
  final AudioCaptureService audioService;

  const MainAppScreen({
    Key? key,
    required this.engine,
    required this.audioService,
  }) : super(key: key);

  @override
  _MainAppScreenState createState() => _MainAppScreenState();
}

class _MainAppScreenState extends State<MainAppScreen> {
  int _currentIndex = 0;
  late final List<Widget> _screens;

  @override
  void initState() {
    super.initState();
    _screens = [
      LiveRecitationScreen(
        engine: widget.engine,
        audioService: widget.audioService,
      ),
      const SurahIndexScreen(),
    ];
  }

  @override
  void dispose() {
    unawaited(widget.audioService.dispose());
    unawaited(widget.engine.dispose());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _currentIndex, children: _screens),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.mic),
            label: 'Live Recitation',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.menu_book),
            label: 'Quran Reader',
          ),
        ],
      ),
    );
  }
}
