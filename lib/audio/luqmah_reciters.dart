class LuqmahReciter {
  final String name;
  final String folder;

  const LuqmahReciter({required this.name, required this.folder});
}

abstract final class LuqmahReciters {
  static const defaultFolder = 'Alafasy_128kbps';

  static const all = <LuqmahReciter>[
    LuqmahReciter(name: 'Mishary Rashid Alafasy', folder: defaultFolder),
    LuqmahReciter(
      name: 'Abdul Basit Abdul Samad',
      folder: 'Abdul_Basit_Murattal_192kbps',
    ),
    LuqmahReciter(name: 'Mahmoud Khalil Al-Husary', folder: 'Husary_128kbps'),
    LuqmahReciter(
      name: 'Muhammad Siddiq Al-Minshawi',
      folder: 'Minshawy_Murattal_128kbps',
    ),
    LuqmahReciter(
      name: 'Abdul Rahman Al-Sudais',
      folder: 'Abdurrahmaan_As-Sudais_192kbps',
    ),
    LuqmahReciter(name: 'Maher Al-Muaiqly', folder: 'Maher_AlMuaiqly_64kbps'),
    LuqmahReciter(name: 'Saud Al-Shuraim', folder: 'Saood_ash-Shuraym_128kbps'),
  ];

  static LuqmahReciter fromFolder(String? folder) {
    return all.firstWhere(
      (reciter) => reciter.folder == folder,
      orElse: () => all.first,
    );
  }
}
