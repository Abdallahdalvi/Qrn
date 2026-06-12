import 'package:flutter/material.dart';
import '../api/quran_api.dart';
import '../providers/quran_preferences.dart';

class QariSelectionScreen extends StatefulWidget {
  const QariSelectionScreen({Key? key}) : super(key: key);

  @override
  _QariSelectionScreenState createState() => _QariSelectionScreenState();
}

class _QariSelectionScreenState extends State<QariSelectionScreen> {
  List<dynamic> _reciters = [];
  bool _isLoading = true;
  int _selectedId = 7;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    try {
      final reciters = await QuranApi.fetchReciters();
      final currentId = await QuranPreferences.getReciterId();
      setState(() {
        _reciters = reciters;
        _selectedId = currentId;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to load Qaris: $e')),
      );
    }
  }

  Future<void> _selectReciter(dynamic reciter) async {
    final id = reciter['id'] as int;
    final name = reciter['reciter_name'] ?? reciter['translated_name']?['name'] ?? 'Unknown Qari';
    
    await QuranPreferences.saveReciter(id, name);
    setState(() {
      _selectedId = id;
    });
    
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Saved $name as default Qari!')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Select Qari'),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : ListView.separated(
              itemCount: _reciters.length,
              separatorBuilder: (_, __) => const Divider(height: 1),
              itemBuilder: (context, index) {
                final reciter = _reciters[index];
                final id = reciter['id'] as int;
                final name = reciter['reciter_name'] ?? reciter['translated_name']?['name'] ?? 'Unknown Qari';
                final style = reciter['style'] ?? '';
                final isSelected = id == _selectedId;

                return ListTile(
                  leading: CircleAvatar(
                    backgroundColor: isSelected ? Colors.blue : Colors.grey.shade300,
                    child: Icon(Icons.person, color: isSelected ? Colors.white : Colors.grey.shade600),
                  ),
                  title: Text(
                    name,
                    style: TextStyle(
                      fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                    ),
                  ),
                  subtitle: style.isNotEmpty ? Text(style) : null,
                  trailing: isSelected ? const Icon(Icons.check_circle, color: Colors.blue) : null,
                  onTap: () => _selectReciter(reciter),
                );
              },
            ),
    );
  }
}
