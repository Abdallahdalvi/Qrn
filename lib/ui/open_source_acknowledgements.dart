import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;

class OpenSourceAcknowledgementsScreen extends StatelessWidget {
  const OpenSourceAcknowledgementsScreen({Key? key}) : super(key: key);

  Future<String> _loadNotices() async {
    try {
      return await rootBundle.loadString('THIRD_PARTY_NOTICES.md');
    } catch (e) {
      return "Could not load open source notices: $e";
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Open Source Acknowledgements'),
      ),
      body: FutureBuilder<String>(
        future: _loadNotices(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          final text = snapshot.data ?? 'No data found';
          return SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: Text(text),
          );
        },
      ),
    );
  }
}
