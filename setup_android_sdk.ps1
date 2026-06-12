$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$env:JAVA_HOME = "C:\Program Files\ojdkbuild\java-17-openjdk-17.0.3.0.6-1"
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"

$androidSdkRoot = "C:\Users\CoreX\AppData\Local\Android\Sdk"
$cmdLineToolsUrl = "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip"
$zipPath = "$env:TEMP\cmdline-tools.zip"

Write-Host "Creating SDK directories..."
New-Item -ItemType Directory -Force -Path "$androidSdkRoot\cmdline-tools" | Out-Null

Write-Host "Downloading command-line tools..."
Invoke-WebRequest -Uri $cmdLineToolsUrl -OutFile $zipPath

Write-Host "Extracting command-line tools..."
Expand-Archive -Path $zipPath -DestinationPath "$androidSdkRoot\cmdline-tools" -Force

# The zip extracts to a folder named "cmdline-tools". We must rename it to "latest" inside the "cmdline-tools" folder
Rename-Item -Path "$androidSdkRoot\cmdline-tools\cmdline-tools" -NewName "latest" -Force

$sdkManager = "$androidSdkRoot\cmdline-tools\latest\bin\sdkmanager.bat"

Write-Host "Accepting licenses..."
# Yes to all licenses
cmd.exe /c "echo y| ""$sdkManager"" --licenses"

Write-Host "Installing platform tools and build tools..."
cmd.exe /c """$sdkManager"" ""platform-tools"" ""platforms;android-34"" ""build-tools;34.0.0"""

Write-Host "Android SDK Setup Complete."
