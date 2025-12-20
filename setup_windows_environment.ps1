# NNTrainer Windows Environment Setup Script
# Run this script as Administrator in PowerShell

param(
    [switch]$SkipVisualStudio,
    [switch]$SkipVcpkg,
    [string]$InstallPath = "C:\nntrainer-dev"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Green
Write-Host "NNTrainer Windows Development Environment Setup" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Error "This script must be run as Administrator. Please restart PowerShell as Administrator and try again."
    exit 1
}

# Create installation directory
Write-Host "Creating installation directory: $InstallPath" -ForegroundColor Yellow
New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null

# Enable developer features
Write-Host "Enabling Windows developer features..." -ForegroundColor Yellow
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

# Enable Windows features
Write-Host "Enabling Windows Subsystem for Linux (optional)..." -ForegroundColor Yellow
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart /quiet

# Install Chocolatey if not present
if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey package manager..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    
    # Refresh environment to use choco
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# Install essential development tools
Write-Host "Installing essential development tools..." -ForegroundColor Yellow
$packages = @(
    "git",
    "cmake --version=3.31.0",
    "python3 --version=3.11.0",
    "ninja",
    "7zip",
    "wget",
    "vcredist140"
)

foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    choco install $package -y --no-progress
}

# Install Python packages
Write-Host "Installing Python packages..." -ForegroundColor Yellow
pip install --upgrade pip
pip install meson==1.6.1 ninja

# Install Visual Studio Build Tools (if not skipped)
if (!$SkipVisualStudio) {
    Write-Host "Checking for Visual Studio Build Tools..." -ForegroundColor Yellow
    $vsPath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools"
    if (!(Test-Path $vsPath)) {
        Write-Host "Visual Studio Build Tools not found. Please install manually:" -ForegroundColor Red
        Write-Host "https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Red
        Write-Host "Required workloads:" -ForegroundColor Red
        Write-Host "- Desktop development with C++" -ForegroundColor Red
        Write-Host "- C++ CMake tools for Windows" -ForegroundColor Red
        Write-Host "- C++ Clang tools for Windows" -ForegroundColor Red
        Write-Host "- Windows 10/11 SDK (latest)" -ForegroundColor Red
        Read-Host "Press Enter after installing Visual Studio Build Tools..."
    } else {
        Write-Host "Visual Studio Build Tools found at: $vsPath" -ForegroundColor Green
    }
}

# Download and setup OpenBLAS
Write-Host "Setting up OpenBLAS..." -ForegroundColor Yellow
$openblasPath = Join-Path $InstallPath "openblas"
if (!(Test-Path $openblasPath)) {
    New-Item -ItemType Directory -Path $openblasPath -Force | Out-Null
    $openblasUrl = "https://github.com/xianyi/OpenBLAS/releases/download/v0.3.24/OpenBLAS-0.3.24-x64.zip"
    $openblasZip = Join-Path $InstallPath "openblas.zip"
    
    Write-Host "Downloading OpenBLAS..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $openblasUrl -OutFile $openblasZip
    
    Write-Host "Extracting OpenBLAS..." -ForegroundColor Cyan
    Expand-Archive -Path $openblasZip -DestinationPath $openblasPath -Force
    Remove-Item $openblasZip
}

# Setup vcpkg (if not skipped)
if (!$SkipVcpkg) {
    Write-Host "Setting up vcpkg..." -ForegroundColor Yellow
    $vcpkgPath = Join-Path $InstallPath "vcpkg"
    if (!(Test-Path $vcpkgPath)) {
        Write-Host "Cloning vcpkg..." -ForegroundColor Cyan
        git clone https://github.com/Microsoft/vcpkg.git $vcpkgPath
        
        Write-Host "Bootstrapping vcpkg..." -ForegroundColor Cyan
        & "$vcpkgPath\bootstrap-vcpkg.bat"
        
        Write-Host "Integrating vcpkg..." -ForegroundColor Cyan
        & "$vcpkgPath\vcpkg.exe" integrate install
        
        Write-Host "Installing vcpkg packages..." -ForegroundColor Cyan
        $vcpkgPackages = @("gtest:x64-windows", "benchmark:x64-windows", "jsoncpp:x64-windows")
        foreach ($pkg in $vcpkgPackages) {
            Write-Host "Installing $pkg..." -ForegroundColor Cyan
            & "$vcpkgPath\vcpkg.exe" install $pkg
        }
    }
}

# Download Dr. Memory for memory testing
Write-Host "Setting up Dr. Memory..." -ForegroundColor Yellow
$drmemoryUrl = "https://github.com/DynamoRIO/drmemory/releases/download/release_2.6.0/DrMemory-Windows-2.6.0.msi"
$drmemoryMsi = Join-Path $InstallPath "drmemory.msi"
if (!(Test-Path "${env:ProgramFiles(x86)}\Dr. Memory")) {
    Write-Host "Downloading Dr. Memory..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $drmemoryUrl -OutFile $drmemoryMsi
    
    Write-Host "Installing Dr. Memory..." -ForegroundColor Cyan
    Start-Process msiexec.exe -ArgumentList "/i", $drmemoryMsi, "/quiet" -Wait
    Remove-Item $drmemoryMsi
}

# Clone NNTrainer repository
Write-Host "Setting up NNTrainer repository..." -ForegroundColor Yellow
$nntrainerPath = Join-Path $InstallPath "nntrainer"
if (!(Test-Path $nntrainerPath)) {
    Write-Host "Cloning NNTrainer..." -ForegroundColor Cyan
    git clone https://github.com/nnstreamer/nntrainer.git $nntrainerPath
    
    Write-Host "Initializing submodules..." -ForegroundColor Cyan
    Push-Location $nntrainerPath
    git submodule update --init --recursive
    Pop-Location
}

# Create environment setup script
$envScript = @"
@echo off
echo Setting up NNTrainer Windows development environment...

REM Set up Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM Add tools to PATH
set PATH=%PATH%;C:\Program Files\CMake\bin
set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\Llvm\x64\bin
set PATH=%PATH%;$InstallPath\vcpkg

REM Set CMake prefix path for dependencies
set CMAKE_PREFIX_PATH=$openblasPath;%CMAKE_PREFIX_PATH%

REM Set vcpkg toolchain
set VCPKG_ROOT=$vcpkgPath
set CMAKE_TOOLCHAIN_FILE=$vcpkgPath\scripts\buildsystems\vcpkg.cmake

echo Environment setup complete!
echo.
echo To build NNTrainer:
echo   cd $nntrainerPath
echo   meson setup --native-file windows-native.ini builddir
echo   meson compile -C builddir
echo.
"@

$envScriptPath = Join-Path $InstallPath "setup_env.bat"
$envScript | Out-File -FilePath $envScriptPath -Encoding ASCII

# Create memory testing script
$memoryTestScript = @"
# NNTrainer Memory Testing Suite
param(
    [Parameter(Mandatory=`$true)]
    [string]`$BuildDir,
    [string]`$OutputDir = "memory_test_results"
)

`$ErrorActionPreference = "Stop"

Write-Host "Starting NNTrainer Memory Testing Suite..." -ForegroundColor Green

# Create output directory with timestamp
`$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
`$resultsDir = "`$OutputDir`_`$timestamp"
New-Item -ItemType Directory -Path `$resultsDir -Force | Out-Null

# Test applications list
`$testApps = @(
    "`$BuildDir\Applications\MNIST\mnist_main.exe",
    "`$BuildDir\Applications\SimpleFC\simplefc_main.exe", 
    "`$BuildDir\Applications\LogisticRegression\logistic_main.exe"
)

# Dr. Memory path
`$drmemoryExe = "`${env:ProgramFiles(x86)}\Dr. Memory\bin64\drmemory.exe"

if (!(Test-Path `$drmemoryExe)) {
    Write-Error "Dr. Memory not found at `$drmemoryExe"
    exit 1
}

# Test each application with Dr. Memory
foreach (`$app in `$testApps) {
    if (Test-Path `$app) {
        `$appName = [System.IO.Path]::GetFileNameWithoutExtension(`$app)
        Write-Host "Testing `$appName with Dr. Memory..." -ForegroundColor Yellow
        
        `$logDir = Join-Path `$resultsDir `$appName
        & `$drmemoryExe -logdir `$logDir -brief -- `$app
        
        Write-Host "Results saved to: `$logDir" -ForegroundColor Green
    } else {
        Write-Warning "Application not found: `$app"
    }
}

# Run unit tests with memory checking
Write-Host "Running unit tests..." -ForegroundColor Yellow
`$testOutput = Join-Path `$resultsDir "unittest_results.txt"
meson test -C `$BuildDir --verbose > `$testOutput 2>&1

Write-Host "Memory testing completed!" -ForegroundColor Green
Write-Host "Results saved in: `$resultsDir" -ForegroundColor Green
"@

$memoryTestScriptPath = Join-Path $InstallPath "run_memory_tests.ps1"
$memoryTestScript | Out-File -FilePath $memoryTestScriptPath -Encoding UTF8

# Final setup summary
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Installation path: $InstallPath" -ForegroundColor Cyan
Write-Host "NNTrainer source: $nntrainerPath" -ForegroundColor Cyan
Write-Host "Environment script: $envScriptPath" -ForegroundColor Cyan
Write-Host "Memory test script: $memoryTestScriptPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run '$envScriptPath' to set up environment variables" -ForegroundColor White
Write-Host "2. Navigate to '$nntrainerPath'" -ForegroundColor White
Write-Host "3. Build with: meson setup --native-file windows-native.ini builddir" -ForegroundColor White
Write-Host "4. Compile with: meson compile -C builddir" -ForegroundColor White
Write-Host "5. Test with: meson test -C builddir" -ForegroundColor White
Write-Host "6. Run memory tests with: '$memoryTestScriptPath -BuildDir builddir'" -ForegroundColor White
Write-Host ""

if (!$SkipVisualStudio) {
    Write-Host "NOTE: Make sure Visual Studio Build Tools 2022 is properly installed!" -ForegroundColor Red
}

Write-Host "Setup completed successfully!" -ForegroundColor Green