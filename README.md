# Al-Fatih Alal-Imaam (Qrn)

A highly advanced AI-powered Quran recitation assistant that automatically tracks your recitation, seamlessly follows you across Ayahs, and intelligently prompts you when you get stuck or make a mistake.

## 🚀 How to Use (Simple Guide)

1. **Install the App:** Download the latest `app-release.apk` (found in `build/app/outputs/flutter-apk/`) to your Android device and install it.
2. **Start the Backend Engine (For PC Hosting):**
   - Make sure you have Python installed.
   - Simply double-click the `start_backend.bat` file in the main folder to automatically launch the AI engine.
3. **Connect Your Phone:**
   - Make sure your phone and your PC are on the **same Wi-Fi network**.
   - Open the App on your phone.
   - Go to **Settings** (Gear icon) -> **Server Address**.
   - Enter the Local IP Address of your PC (e.g., `192.168.1.100`) and Port `8000`.
4. **Start Reciting!**
   - Go back to the main screen.
   - Tap the **Microphone** button to start tracking.
   - The AI will automatically figure out what Surah and Ayah you are reading.
   - **Taraweeh Mode:** If you enable this, it locks onto your current position and safely tracks you forward.
   - **Prompt Mode:** If you get stuck, simply stay silent. The AI will detect your pause, play the audio of the next Ayah to help you, and securely lock its tracking to your recovery!

## ✨ Features
- **Acoustic Tracking:** Tracks your precise Surah and Ayah using advanced phonetic AI models.
- **Assisted Recovery:** Repeats the next Ayah's audio up to 3 times if you're stuck, and immediately locks its tracking to ensure it follows you.
- **Mutashabihat Protection:** Safely locks tracking so the engine won't randomly jump to a similar-sounding Ayah in a different Surah.

Enjoy your recitation with your new AI teacher!
