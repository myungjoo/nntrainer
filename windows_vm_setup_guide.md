# Windows/x64 Virtual Machine Setup Guide for NNTrainer

This guide provides step-by-step instructions for setting up a Windows/x64 virtual machine environment to build, test, and debug NNTrainer with memory analysis tools.

## Table of Contents
1. [Windows VM Setup](#windows-vm-setup)
2. [Development Environment Setup](#development-environment-setup)
3. [Dependencies Installation](#dependencies-installation)
4. [NNTrainer Build Process](#nntrainer-build-process)
5. [Testing Strategy](#testing-strategy)
6. [Memory Testing Tools](#memory-testing-tools)
7. [Bug Reporting](#bug-reporting)

## Windows VM Setup

### 1. Virtual Machine Requirements
- **OS**: Windows 10/11 x64 (minimum build 1903)
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: 4+ cores with virtualization support
- **Storage**: 100GB+ SSD space
- **Virtualization Platform**: VMware Workstation, VirtualBox, or Hyper-V

### 2. Windows VM Configuration
```powershell
# Enable Developer Mode (Run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Enable Windows Subsystem for Linux (optional for cross-validation)
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Enable Hyper-V platform
dism.exe /online /enable-feature /featurename:HypervisorPlatform /all /norestart
```

## Development Environment Setup

### 1. Visual Studio Build Tools 2022 Installation
```powershell
# Download and install Visual Studio Build Tools 2022 (17.13.x)
# https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Required workloads:
# - Desktop development with C++
# - C++ CMake tools for Windows
# - C++ Clang tools for Windows (for clang builds)
# - Windows 10/11 SDK (latest version)
```

### 2. Essential Tools Installation
```powershell
# Install Chocolatey package manager
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install essential development tools
choco install -y git
choco install -y cmake --version=3.31.0
choco install -y python --version=3.11.0
choco install -y ninja
choco install -y 7zip
choco install -y wget

# Install Meson build system
pip install meson==1.6.1
pip install ninja
```

### 3. Environment Setup
```powershell
# Add tools to PATH
$env:PATH += ";C:\Program Files\CMake\bin"
$env:PATH += ";C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\Llvm\x64\bin"

# Set up Visual Studio environment
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

## Dependencies Installation

### 1. OpenBLAS (Linear Algebra Library)
```powershell
# Download prebuilt OpenBLAS for Windows
wget https://github.com/xianyi/OpenBLAS/releases/download/v0.3.24/OpenBLAS-0.3.24-x64.zip -O openblas.zip
7z x openblas.zip -oC:\openblas
$env:CMAKE_PREFIX_PATH = "C:\openblas;$env:CMAKE_PREFIX_PATH"
```

### 2. Additional Dependencies
```powershell
# Install vcpkg for C++ package management
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Install required packages
.\vcpkg install gtest:x64-windows
.\vcpkg install benchmark:x64-windows
.\vcpkg install jsoncpp:x64-windows
```

## NNTrainer Build Process

### 1. Clone NNTrainer Repository
```powershell
# Clone main repository
git clone https://github.com/nnstreamer/nntrainer.git
cd nntrainer

# Initialize submodules
git submodule update --init --recursive
```

### 2. Build with MSVC (Recommended)
```powershell
# Setup build environment
meson setup --native-file windows-native.ini builddir --prefix=C:\nntrainer

# Compile the project
meson compile -C builddir

# Install binaries
meson install -C builddir
```

### 3. Build with Clang (Alternative)
```powershell
# Setup build environment with Clang
meson setup --native-file windows-native-clang.ini builddir-clang --prefix=C:\nntrainer-clang

# Compile the project
meson compile -C builddir-clang

# Install binaries
meson install -C builddir-clang
```

### 4. Verify Build
```powershell
# Check if binaries are created
ls builddir\nntrainer\*.dll
ls builddir\Applications\*\*.exe
```

## Testing Strategy

### 1. Unit Tests Execution
```powershell
# Run all unit tests
meson test -C builddir --verbose

# Run specific test categories
meson test -C builddir --suite unittest

# Generate test coverage report
python test\unittestcoverage.py --build-dir builddir
```

### 2. Sample Applications Testing
```powershell
# Test simple examples
cd builddir\Applications\SimpleFC
.\simplefc_main.exe

# Test MNIST example
cd ..\MNIST
.\mnist_main.exe

# Test LogisticRegression
cd ..\LogisticRegression
.\logistic_main.exe
```

### 3. Custom Application Development
Create a test application to validate functionality:

```cpp
// test_nntrainer.cpp
#include <nntrainer.h>
#include <model.h>
#include <dataset.h>
#include <iostream>

int main() {
    try {
        // Initialize NNTrainer
        ml::train::Model model;
        
        // Create a simple neural network
        model.addLayer(ml::train::createLayer("input", {"input_shape=1:1:784"}));
        model.addLayer(ml::train::createLayer("fully_connected", {"unit=128", "activation=relu"}));
        model.addLayer(ml::train::createLayer("fully_connected", {"unit=10", "activation=softmax"}));
        
        // Compile model
        model.setOptimizer(ml::train::createOptimizer("adam", {"learning_rate=0.001"}));
        model.setLoss("cross_softmax");
        model.compile();
        
        std::cout << "NNTrainer Windows build test: SUCCESS" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
```

## Memory Testing Tools

### 1. Application Verifier
```powershell
# Enable Application Verifier for memory debugging
# Download from Windows SDK or use built-in version
appverif.exe

# Enable heap verification for target applications
appverif -enable Heaps -for builddir\Applications\MNIST\mnist_main.exe
appverif -enable Handles -for builddir\Applications\MNIST\mnist_main.exe
```

### 2. CRT Debug Heap
Add to test applications:
```cpp
#ifdef _WIN32
#include <crtdbg.h>

int main() {
    // Enable memory leak detection
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    
    // Your application code here
    
    return 0;
}
#endif
```

### 3. Dr. Memory Installation
```powershell
# Download Dr. Memory
wget https://github.com/DynamoRIO/drmemory/releases/download/release_2.6.0/DrMemory-Windows-2.6.0.msi -O drmemory.msi
msiexec /i drmemory.msi /quiet

# Run memory analysis
"C:\Program Files (x86)\Dr. Memory\bin64\drmemory.exe" -logdir logs -- builddir\Applications\MNIST\mnist_main.exe
```

### 4. Intel Inspector (if available)
```powershell
# If Intel Inspector is installed, use it for comprehensive memory analysis
inspxe-cl -collect memory-access -result-dir inspector_results -- builddir\Applications\MNIST\mnist_main.exe
```

## Memory Testing Script

Create an automated testing script:

```powershell
# memory_test_suite.ps1
param(
    [Parameter(Mandatory=$true)]
    [string]$BuildDir
)

Write-Host "Starting Memory Testing Suite for NNTrainer Windows Build"

# Create results directory
$ResultsDir = "memory_test_results_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $ResultsDir

# Test applications with Dr. Memory
$TestApps = @(
    "$BuildDir\Applications\MNIST\mnist_main.exe",
    "$BuildDir\Applications\SimpleFC\simplefc_main.exe",
    "$BuildDir\Applications\LogisticRegression\logistic_main.exe"
)

foreach ($app in $TestApps) {
    if (Test-Path $app) {
        $appName = [System.IO.Path]::GetFileNameWithoutExtension($app)
        Write-Host "Testing $appName with Dr. Memory..."
        
        & "C:\Program Files (x86)\Dr. Memory\bin64\drmemory.exe" `
            -logdir "$ResultsDir\$appName" `
            -brief `
            -- $app
    }
}

# Run unit tests with memory checking
Write-Host "Running unit tests with memory verification..."
$env:_CRTDBG_FLAGS = "_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF"
meson test -C $BuildDir --verbose > "$ResultsDir\unittest_results.txt" 2>&1

Write-Host "Memory testing completed. Results saved in $ResultsDir"
```

## Bug Reporting

### 1. Memory Leak Detection
Monitor for:
- Heap corruption
- Buffer overflows/underflows
- Use-after-free errors
- Memory leaks
- Uninitialized memory access

### 2. Performance Profiling
```powershell
# Use Windows Performance Toolkit
wpa.exe  # Windows Performance Analyzer

# Profile specific applications
wpr.exe -start CPU  # Start CPU profiling
# Run your application
wpr.exe -stop profile.etl  # Stop and save profile
```

### 3. Crash Dump Analysis
```powershell
# Enable crash dumps
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /v DumpType /t REG_DWORD /d 2
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps" /v DumpFolder /t REG_SZ /d "C:\CrashDumps"

# Analyze dumps with WinDbg
windbg.exe -z C:\CrashDumps\app.dmp
```

### 4. Bug Report Template
When reporting bugs, include:

```markdown
## Bug Report: Memory Issue in NNTrainer Windows Build

### Environment
- Windows Version: [Windows 10/11 build number]
- Visual Studio Version: [17.13.x]
- NNTrainer Commit: [git commit hash]
- Build Configuration: [MSVC/Clang]

### Issue Description
[Detailed description of the memory issue]

### Reproduction Steps
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Memory Analysis Results
- Dr. Memory Log: [attach log file]
- Application Verifier: [any violations found]
- CRT Debug Output: [any leak reports]

### Stack Trace
```
[Include relevant stack traces]
```

### Additional Information
- Performance impact: [describe any performance issues]
- Frequency: [how often does this occur]
- Workaround: [any temporary fixes found]
```

## Expected Deliverables

1. **Fully configured Windows VM** with development environment
2. **Successfully built NNTrainer binaries** (both MSVC and Clang builds)
3. **Comprehensive test results** from unit tests and sample applications
4. **Memory analysis reports** from various testing tools
5. **Bug reports** for any memory issues discovered
6. **Performance benchmarks** comparing Windows vs Linux builds
7. **Documentation** of any Windows-specific issues or optimizations needed

## Success Criteria

- [ ] All unit tests pass on Windows build
- [ ] Sample applications run without crashes
- [ ] No critical memory leaks detected
- [ ] Performance within acceptable range of Linux builds
- [ ] Memory debugging tools properly configured and functional
- [ ] Automated testing scripts working correctly

This setup provides a comprehensive environment for Windows-native development and testing of NNTrainer with robust memory analysis capabilities.