$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

Write-Host "Installing Android SDK platforms and build-tools..."
android sdk install "platforms;android-34" "build-tools;34.0.0"

# Install Flutter if missing
if (-not (Test-Path "C:\src\flutter\bin\flutter.bat")) {
    Write-Host "Downloading and installing Flutter..."
    New-Item -ItemType Directory -Force -Path "C:\src" | Out-Null
    $flutterZip = "$env:TEMP\flutter.zip"
    curl.exe -L -o $flutterZip "https://storage.googleapis.com/flutter_infra_release/releases/stable/windows/flutter_windows_3.24.1-stable.zip"
    Write-Host "Extracting Flutter..."
    Expand-Archive -Path $flutterZip -DestinationPath "C:\src" -Force
}

# Setup PATH for this session
$env:PATH = "C:\src\flutter\bin;$env:PATH"

Write-Host "Building APK..."
cd C:\Users\devev\Qrn
flutter build apk --release
