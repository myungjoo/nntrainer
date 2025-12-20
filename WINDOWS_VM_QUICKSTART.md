# NNTrainer Windows VM Quick Start Guide

This guide provides a rapid deployment path for setting up a Windows/x64 virtual machine environment for NNTrainer development, testing, and memory analysis.

## 📋 Overview

This package includes:
- ✅ Comprehensive setup documentation (`windows_vm_setup_guide.md`)
- ✅ Automated environment setup script (`setup_windows_environment.ps1`)
- ✅ Build validation and testing script (`validate_build.ps1`)
- ✅ Memory testing capabilities with Dr. Memory integration
- ✅ Complete Windows-native build environment (not cross-compilation)

## 🚀 Quick Start (30 minutes)

### Step 1: Windows VM Setup
1. **Create Windows 10/11 x64 VM** (8GB+ RAM, 100GB+ storage)
2. **Enable Developer Mode** in Windows Settings
3. **Install Windows Updates** and reboot

### Step 2: Automated Environment Setup
```powershell
# Download this repository to your Windows VM
# Open PowerShell as Administrator

# Run the automated setup script
.\setup_windows_environment.ps1

# Wait for completion (~20-30 minutes)
```

### Step 3: Manual Visual Studio Installation
```powershell
# If not automated, install Visual Studio Build Tools 2022:
# https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Required workloads:
# ✅ Desktop development with C++
# ✅ C++ CMake tools for Windows  
# ✅ C++ Clang tools for Windows
# ✅ Windows 10/11 SDK (latest)
```

### Step 4: Build NNTrainer
```powershell
# Navigate to NNTrainer directory
cd C:\nntrainer-dev\nntrainer

# Set up environment
C:\nntrainer-dev\setup_env.bat

# Configure build (MSVC)
meson setup --native-file windows-native.ini builddir

# OR configure build (Clang)
# meson setup --native-file windows-native-clang.ini builddir-clang

# Compile
meson compile -C builddir

# Install
meson install -C builddir
```

### Step 5: Validate and Test
```powershell
# Run comprehensive validation
.\validate_build.ps1 -BuildDir builddir

# Run memory tests specifically
C:\nntrainer-dev\run_memory_tests.ps1 -BuildDir builddir

# Check results in generated report directories
```

## 📊 Expected Results

### Build Artifacts
- ✅ `nntrainer.dll` - Main library
- ✅ `nntrainer.lib` - Import library  
- ✅ Unit test executables
- ✅ Sample applications (MNIST, SimpleFC, LogisticRegression, etc.)

### Test Coverage
- ✅ **Unit Tests**: 150+ automated tests
- ✅ **Sample Apps**: 10+ working examples
- ✅ **Memory Tests**: Dr. Memory analysis for all apps
- ✅ **Performance**: Benchmark results

## 🔍 Memory Analysis Tools Included

### Dr. Memory
- **Purpose**: Detects memory leaks, buffer overflows, use-after-free
- **Location**: `C:\Program Files (x86)\Dr. Memory\`
- **Usage**: Automated via validation scripts

### Application Verifier
- **Purpose**: Windows-native heap verification
- **Location**: Built into Windows SDK
- **Usage**: Manual configuration for specific apps

### CRT Debug Heap
- **Purpose**: Debug heap for memory leak detection
- **Location**: Built into MSVC runtime
- **Usage**: Compile-time flags for debug builds

## 📈 Performance Expectations

| Component | Expected Performance |
|-----------|---------------------|
| Build Time | 5-15 minutes (depending on VM specs) |
| Unit Tests | 2-5 minutes |
| Memory Analysis | 1-2 minutes per application |
| Sample Apps | <30 seconds each |

## 🐛 Common Issues & Solutions

### Issue: Meson 1.8.0 Compatibility
**Error**: CMake submodule failures with `/WX` parameters
**Solution**: Use Meson 1.6.1 (included in setup script)

### Issue: Visual Studio Environment
**Error**: MSVC compiler not found
**Solution**: Run `setup_env.bat` or manually call `vcvars64.bat`

### Issue: OpenBLAS Dependencies
**Error**: BLAS library not found
**Solution**: Verify `CMAKE_PREFIX_PATH` includes OpenBLAS directory

### Issue: Memory Test Timeouts
**Error**: Dr. Memory tests timeout
**Solution**: Increase timeout values in validation script

## 📁 Generated Deliverables

After successful completion, you'll have:

```
C:\nntrainer-dev\
├── nntrainer\                  # Source code
├── builddir\                   # Build artifacts
├── openblas\                   # BLAS library
├── vcpkg\                      # Package manager
├── setup_env.bat              # Environment setup
├── run_memory_tests.ps1       # Memory testing
└── validation_report_*\       # Test results
    ├── validation_report.md   # Comprehensive report
    ├── validation.log         # Detailed logs
    ├── memory_*/              # Dr. Memory results
    └── *_output.txt           # Application outputs
```

## 🎯 Success Criteria Checklist

- [ ] ✅ Windows VM fully configured with development tools
- [ ] ✅ NNTrainer builds successfully (both MSVC and Clang)
- [ ] ✅ All unit tests pass
- [ ] ✅ Sample applications run without crashes
- [ ] ✅ Memory analysis tools operational
- [ ] ✅ No critical memory leaks detected
- [ ] ✅ Performance within acceptable ranges
- [ ] ✅ Comprehensive test reports generated

## ⚡ Advanced Usage

### Custom Build Configurations
```powershell
# Enable specific features
meson setup --native-file windows-native.ini builddir -Denable-cublas=true -Denable-fp16=true

# Debug build with symbols
meson setup --native-file windows-native.ini builddir --buildtype=debug

# Release build optimized
meson setup --native-file windows-native.ini builddir --buildtype=release
```

### Memory Testing Customization
```powershell
# Test specific applications only
.\validate_build.ps1 -BuildDir builddir -SkipPerformanceTests

# Extended memory analysis
$apps = @("builddir\Applications\MNIST\mnist_main.exe")
foreach ($app in $apps) {
    & "C:\Program Files (x86)\Dr. Memory\bin64\drmemory.exe" -logdir "detailed_memory_analysis" -light -- $app
}
```

### CI/CD Integration
```powershell
# Automated build and test pipeline
$ErrorActionPreference = "Stop"

# Setup
.\setup_windows_environment.ps1 -SkipVisualStudio

# Build
cd C:\nntrainer-dev\nntrainer
meson setup --native-file windows-native.ini builddir
meson compile -C builddir

# Test and validate
.\validate_build.ps1 -BuildDir builddir

# Exit with appropriate code for CI
exit $LASTEXITCODE
```

## 📞 Support

If you encounter issues:

1. **Check logs**: Review `validation.log` for detailed error messages
2. **Verify environment**: Ensure all tools are in PATH
3. **Memory issues**: Check Dr. Memory logs in report directories
4. **Build failures**: Verify Visual Studio Build Tools installation

## 🏁 Conclusion

This setup provides a complete Windows-native development and testing environment for NNTrainer with:
- ✅ Full build system automation
- ✅ Comprehensive testing framework
- ✅ Advanced memory analysis tools
- ✅ Detailed reporting and logging
- ✅ Production-ready binaries

Total setup time: **~30 minutes** automated + manual VS installation
Total validation time: **~10-15 minutes** for complete test suite