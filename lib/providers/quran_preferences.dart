import 'package:shared_preferences/shared_preferences.dart';

class QuranPreferences {
  static const String _reciterIdKey = 'selected_reciter_id';
  static const String _reciterNameKey = 'selected_reciter_name';

  // Default to Mishary Al-Afasy (ID: 7)
  static Future<int> getReciterId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getInt(_reciterIdKey) ?? 7;
  }

  static Future<String> getReciterName() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_reciterNameKey) ?? 'Mishari Rashid al-`Afasy';
  }

  static Future<void> saveReciter(int id, String name) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_reciterIdKey, id);
    await prefs.setString(_reciterNameKey, name);
  }
}
