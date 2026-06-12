import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../engine/socket_client.dart';
import '../audio/audio_capture.dart';
import 'package:record/record.dart';
import 'dart:async';

class SettingsScreen extends StatefulWidget {
  final TarteelSocketClient engine;
  final AudioCaptureService audioService;
  final VoidCallback onSaved;

  const SettingsScreen({
    Key? key,
    required this.engine,
    required this.audioService,
    required this.onSaved,
  }) : super(key: key);

  @override
  _SettingsScreenState createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late TextEditingController _ipController;
  late TextEditingController _portController;
  String _testStatus = '';
  bool _isTesting = false;
  StreamSubscription<Amplitude>? _ampSub;
  double _currentRMS = -100.0;

  @override
  void initState() {
    super.initState();
    _ipController = TextEditingController(text: widget.engine.serverIp);
    _portController = TextEditingController(text: widget.engine.serverPort);
    _loadExtraSettings();
    
    _ampSub = widget.audioService.onAmplitude.listen((amp) {
      if (mounted) {
        setState(() => _currentRMS = amp.current);
      }
    });
  }

  int _promptTimeout = 15;
  String _promptAggressiveness = 'Normal';
  int _promptRepeatInterval = 10;
  int _promptMaxRepeats = 3;
  bool _debugModeEnabled = false;
  double _vadThreshold = -20.0;

  Future<void> _loadExtraSettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _promptTimeout = prefs.getInt('prompt_timeout') ?? 15;
      _promptAggressiveness = prefs.getString('prompt_aggressiveness') ?? 'Normal';
      _promptRepeatInterval = prefs.getInt('prompt_repeat_interval') ?? 10;
      _promptMaxRepeats = prefs.getInt('prompt_max_repeats') ?? 3;
      _debugModeEnabled = prefs.getBool('debug_mode') ?? false;
      _vadThreshold = prefs.getDouble('vad_threshold') ?? -20.0;
    });
  }

  @override
  void dispose() {
    _ampSub?.cancel();
    _ipController.dispose();
    _portController.dispose();
    super.dispose();
  }

  Future<void> _testConnection() async {
    final ip = _ipController.text.trim();
    final port = _portController.text.trim();
    if (ip.isEmpty || port.isEmpty) {
      setState(() {
        _testStatus = 'IP and Port cannot be empty';
      });
      return;
    }

    setState(() {
      _isTesting = true;
      _testStatus = 'Testing connection...';
    });

    final success = await widget.engine.testConnection(ip, port);

    setState(() {
      _isTesting = false;
      _testStatus = success ? 'Connected successfully!' : 'Connection failed';
    });
  }

  Future<void> _saveSettings() async {
    final ip = _ipController.text.trim();
    final port = _portController.text.trim();
    if (ip.isEmpty || port.isEmpty) {
      return;
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt('prompt_timeout', _promptTimeout);
    await prefs.setString('prompt_aggressiveness', _promptAggressiveness);
    await prefs.setInt('prompt_repeat_interval', _promptRepeatInterval);
    await prefs.setInt('prompt_max_repeats', _promptMaxRepeats);
    await prefs.setBool('debug_mode', _debugModeEnabled);
    await prefs.setDouble('vad_threshold', _vadThreshold);

    await widget.engine.saveSettings(ip, port);
    widget.onSaved();
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final currentUrl = 'ws://${_ipController.text.trim()}:${_portController.text.trim()}/ws/recitation';

    return Scaffold(
      appBar: AppBar(
        title: const Text('Connection Settings'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Configure Backend Server',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text(
              'Enter the LAN IP and Port of your Windows host machine to connect from your physical Android phone.',
              style: TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 24),
            TextField(
              controller: _ipController,
              decoration: const InputDecoration(
                labelText: 'Server IP Address',
                hintText: 'e.g. 192.168.1.100',
                border: OutlineInputBorder(),
              ),
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              onChanged: (_) => setState(() {}),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _portController,
              decoration: const InputDecoration(
                labelText: 'Server Port',
                hintText: 'e.g. 8000',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.number,
              onChanged: (_) => setState(() {}),
            ),
            const SizedBox(height: 24),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Expanded(
                  child: Text(
                    'Voice Activity Detection (VAD)',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                ),
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: _currentRMS > _vadThreshold ? Colors.green.withOpacity(0.2) : Colors.red.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: _currentRMS > _vadThreshold ? Colors.green : Colors.red),
                  ),
                  child: Text(
                    'Live RMS: ${_currentRMS.toStringAsFixed(1)} dB',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: _currentRMS > _vadThreshold ? Colors.green : Colors.red,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            const Text(
              'Set the minimum volume threshold for speech detection. Make sure your "Live RMS" stays below this number when you are completely silent.',
              style: TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Text('${_vadThreshold.toStringAsFixed(1)} dB', style: const TextStyle(fontWeight: FontWeight.bold)),
                Expanded(
                  child: Slider(
                    value: _vadThreshold,
                    min: -100.0,
                    max: 0.0,
                    divisions: 100,
                    label: _vadThreshold.toStringAsFixed(1),
                    onChanged: (val) {
                      setState(() => _vadThreshold = val);
                    },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),
            const Text(
              'Prompt Mode Settings',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            DropdownButtonFormField<int>(
              value: _promptTimeout,
              decoration: const InputDecoration(labelText: 'Pause Timeout', border: OutlineInputBorder()),
              items: const [
                DropdownMenuItem(value: 5, child: Text('5 Seconds')),
                DropdownMenuItem(value: 10, child: Text('10 Seconds')),
                DropdownMenuItem(value: 15, child: Text('15 Seconds (Default)')),
                DropdownMenuItem(value: 20, child: Text('20 Seconds')),
                DropdownMenuItem(value: 30, child: Text('30 Seconds')),
              ],
              onChanged: (val) {
                if (val != null) setState(() => _promptTimeout = val);
              },
            ),
            const SizedBox(height: 16),
            const SizedBox(height: 16),
            DropdownButtonFormField<String>(
              value: _promptAggressiveness,
              decoration: const InputDecoration(labelText: 'Prompt Aggressiveness', border: OutlineInputBorder()),
              items: const [
                DropdownMenuItem(value: 'Off', child: Text('Off')),
                DropdownMenuItem(value: 'Conservative', child: Text('Conservative')),
                DropdownMenuItem(value: 'Normal', child: Text('Normal')),
                DropdownMenuItem(value: 'Aggressive', child: Text('Aggressive')),
              ],
              onChanged: (val) {
                if (val != null) setState(() => _promptAggressiveness = val);
              },
            ),
            const SizedBox(height: 16),
            DropdownButtonFormField<int>(
              value: _promptRepeatInterval,
              decoration: const InputDecoration(labelText: 'Prompt Repeat Interval', border: OutlineInputBorder()),
              items: const [
                DropdownMenuItem(value: 5, child: Text('5 Seconds')),
                DropdownMenuItem(value: 10, child: Text('10 Seconds (Default)')),
                DropdownMenuItem(value: 15, child: Text('15 Seconds')),
              ],
              onChanged: (val) {
                if (val != null) setState(() => _promptRepeatInterval = val);
              },
            ),
            const SizedBox(height: 16),
            DropdownButtonFormField<int>(
              value: _promptMaxRepeats,
              decoration: const InputDecoration(labelText: 'Maximum Repeats', border: OutlineInputBorder()),
              items: const [
                DropdownMenuItem(value: 1, child: Text('1 Repeat')),
                DropdownMenuItem(value: 2, child: Text('2 Repeats')),
                DropdownMenuItem(value: 3, child: Text('3 Repeats (Default)')),
                DropdownMenuItem(value: 5, child: Text('5 Repeats')),
              ],
              onChanged: (val) {
                if (val != null) setState(() => _promptMaxRepeats = val);
              },
            ),
            const SizedBox(height: 24),
            const Text(
              'Developer Settings',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            SwitchListTile(
              title: const Text('Enable Debug Mode'),
              subtitle: const Text('Show tracking states, search windows, and raw logs on the main screen.'),
              value: _debugModeEnabled,
              onChanged: (val) {
                setState(() => _debugModeEnabled = val);
              },
            ),
            const SizedBox(height: 24),
            const Text(
              'Target URL:',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 4),
            Text(
              currentUrl,
              style: const TextStyle(
                fontFamily: 'monospace',
                fontSize: 13,
                color: Colors.blueGrey,
              ),
            ),
            const SizedBox(height: 24),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: _isTesting ? null : _testConnection,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blueGrey.shade100,
                      foregroundColor: Colors.black87,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                    ),
                    child: _isTesting
                        ? const SizedBox(
                            height: 18,
                            width: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Text('Test Connection'),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton(
                    onPressed: _saveSettings,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                    ),
                    child: const Text('Save & Connect'),
                  ),
                ),
              ],
            ),
            if (_testStatus.isNotEmpty) ...[
              const SizedBox(height: 24),
              Card(
                color: _testStatus.contains('successfully')
                    ? Colors.green.shade50
                    : (_testStatus.contains('failed') ? Colors.red.shade50 : Colors.blue.shade50),
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Text(
                    _testStatus,
                    style: TextStyle(
                      color: _testStatus.contains('successfully')
                          ? Colors.green
                          : (_testStatus.contains('failed') ? Colors.red : Colors.blue),
                      fontWeight: FontWeight.bold,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
