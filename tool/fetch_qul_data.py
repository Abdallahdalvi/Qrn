import sys
import os
import json
import urllib.request

def main():
    if len(sys.argv) < 2:
        print("Usage: python tool/fetch_qul_data.py <chapter_number> [reciter_id]")
        sys.exit(1)
        
    chapter_number = sys.argv[1]
    reciter_id = sys.argv[2] if len(sys.argv) > 2 else "7"
    
    output_dir = "assets/mock_recitations"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Fetching data for Chapter {chapter_number}, Reciter {reciter_id}...")
    
    url = f"https://api.quran.com/api/v4/chapter_recitations/{reciter_id}/{chapter_number}?segments=true"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to fetch API data: {e}")
        sys.exit(1)
        
    audio_file = data.get("audio_file", {})
    audio_url = audio_file.get("audio_url")
    
    if not audio_url:
        print("No audio_url found in response.")
        sys.exit(1)
        
    if not audio_url.startswith("http"):
        # Could be missing domain
        if audio_url.startswith("//"):
            full_audio_url = "https:" + audio_url
        else:
            full_audio_url = "https://audio.qurancdn.com/" + audio_url
    else:
        full_audio_url = audio_url
        
    print(f"Downloading audio from {full_audio_url}...")
    
    mp3_path = os.path.join(output_dir, f"{chapter_number}.mp3")
    json_path = os.path.join(output_dir, f"{chapter_number}.json")
    
    try:
        req_audio = urllib.request.Request(full_audio_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_audio) as response:
            with open(mp3_path, "wb") as out_file:
                out_file.write(response.read())
    except Exception as e:
        print(f"Failed to download audio: {e}")
        sys.exit(1)
        
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audio_file, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved MP3 and JSON data to {output_dir}")

if __name__ == "__main__":
    main()
