# NNTrainer Windows Build Validation Script
# This script validates the NNTrainer build and runs comprehensive tests

param(
    [Parameter(Mandatory=$true)]
    [string]$BuildDir,
    [string]$ReportDir = "validation_report",
    [switch]$SkipMemoryTests,
    [switch]$SkipPerformanceTests,
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# Create report directory
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportPath = "$ReportDir`_$timestamp"
New-Item -ItemType Directory -Path $reportPath -Force | Out-Null

$reportFile = Join-Path $reportPath "validation_report.md"
$logFile = Join-Path $reportPath "validation.log"

function Write-Report {
    param([string]$Message, [string]$Level = "INFO")
    $logEntry = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Level] $Message"
    Write-Host $logEntry
    Add-Content -Path $logFile -Value $logEntry
}

function Write-ReportSection {
    param([string]$Title, [string]$Content)
    Add-Content -Path $reportFile -Value "`n## $Title`n"
    Add-Content -Path $reportFile -Value $Content
}

# Initialize report
$reportHeader = @"
# NNTrainer Windows Build Validation Report

**Generated:** $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  
**Build Directory:** $BuildDir  
**Report Directory:** $reportPath  

---

"@
Add-Content -Path $reportFile -Value $reportHeader

Write-Report "Starting NNTrainer Windows Build Validation" "INFO"

# 1. Environment Check
Write-Report "Checking build environment..." "INFO"
$envCheck = @()

# Check Visual Studio
$vsPath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools"
$vsFound = Test-Path $vsPath
$envCheck += "Visual Studio Build Tools 2022: $(if($vsFound) {'✅ Found'} else {'❌ Missing'})"

# Check CMake
$cmakeVersion = try { (cmake --version | Select-Object -First 1).Split(' ')[2] } catch { "Not found" }
$envCheck += "CMake: $cmakeVersion"

# Check Python
$pythonVersion = try { (python --version 2>&1).Split(' ')[1] } catch { "Not found" }
$envCheck += "Python: $pythonVersion"

# Check Meson
$mesonVersion = try { (meson --version) } catch { "Not found" }
$envCheck += "Meson: $mesonVersion"

Write-ReportSection "Environment Check" ($envCheck -join "`n")

# 2. Build Directory Validation
Write-Report "Validating build directory..." "INFO"
$buildValidation = @()

if (!(Test-Path $BuildDir)) {
    Write-Report "Build directory not found: $BuildDir" "ERROR"
    $buildValidation += "❌ Build directory not found"
    Write-ReportSection "Build Validation" ($buildValidation -join "`n")
    exit 1
}

$buildValidation += "✅ Build directory exists: $BuildDir"

# Check for essential build artifacts
$artifacts = @{
    "NNTrainer Library" = @("$BuildDir\nntrainer\*.dll", "$BuildDir\nntrainer\*.lib")
    "API Headers" = @("$BuildDir\nntrainer\*.h")
    "Unit Tests" = @("$BuildDir\test\unittest\*.exe")
    "Sample Applications" = @(
        "$BuildDir\Applications\MNIST\*.exe",
        "$BuildDir\Applications\SimpleFC\*.exe",
        "$BuildDir\Applications\LogisticRegression\*.exe"
    )
}

foreach ($category in $artifacts.Keys) {
    $found = $false
    foreach ($pattern in $artifacts[$category]) {
        if (Get-ChildItem $pattern -ErrorAction SilentlyContinue) {
            $found = $true
            break
        }
    }
    $buildValidation += "$category: $(if($found) {'✅ Found'} else {'❌ Missing'})"
}

Write-ReportSection "Build Validation" ($buildValidation -join "`n")

# 3. Unit Tests Execution
Write-Report "Running unit tests..." "INFO"
$testStartTime = Get-Date

$testProcess = Start-Process -FilePath "meson" -ArgumentList "test", "-C", $BuildDir, "--verbose", "--timeout-multiplier", "2" -PassThru -RedirectStandardOutput "$reportPath\unittest_output.txt" -RedirectStandardError "$reportPath\unittest_errors.txt" -WindowStyle Hidden

$testCompleted = $testProcess.WaitForExit($TimeoutSeconds * 1000)
$testEndTime = Get-Date
$testDuration = ($testEndTime - $testStartTime).TotalSeconds

$testResults = @()
$testResults += "Duration: $([math]::Round($testDuration, 2)) seconds"
$testResults += "Timeout: $(if($testCompleted) {'No'} else {'Yes - terminated after ' + $TimeoutSeconds + ' seconds'})"

if ($testCompleted) {
    $testResults += "Exit Code: $($testProcess.ExitCode)"
    $testStatus = if ($testProcess.ExitCode -eq 0) { "✅ PASSED" } else { "❌ FAILED" }
} else {
    $testProcess.Kill()
    $testResults += "Exit Code: Timeout"
    $testStatus = "❌ TIMEOUT"
}

$testResults += "Overall Status: $testStatus"

# Parse test output for detailed results
$testOutput = Get-Content "$reportPath\unittest_output.txt" -ErrorAction SilentlyContinue
$testSummary = $testOutput | Where-Object { $_ -match "test.*Ok|FAIL|ERROR" } | Select-Object -Last 10

if ($testSummary) {
    $testResults += "`nTest Summary:`n" + ($testSummary -join "`n")
}

Write-ReportSection "Unit Tests" ($testResults -join "`n")

# 4. Sample Applications Testing
Write-Report "Testing sample applications..." "INFO"
$appTests = @()

$sampleApps = @(
    @{Name="MNIST"; Path="$BuildDir\Applications\MNIST\mnist_main.exe"; Args=@()},
    @{Name="SimpleFC"; Path="$BuildDir\Applications\SimpleFC\simplefc_main.exe"; Args=@()},
    @{Name="LogisticRegression"; Path="$BuildDir\Applications\LogisticRegression\logistic_main.exe"; Args=@()}
)

foreach ($app in $sampleApps) {
    if (Test-Path $app.Path) {
        Write-Report "Testing $($app.Name)..." "INFO"
        
        $appStartTime = Get-Date
        $appProcess = Start-Process -FilePath $app.Path -ArgumentList $app.Args -PassThru -RedirectStandardOutput "$reportPath\$($app.Name)_output.txt" -RedirectStandardError "$reportPath\$($app.Name)_errors.txt" -WindowStyle Hidden
        
        $appCompleted = $appProcess.WaitForExit(60000) # 60 second timeout
        $appEndTime = Get-Date
        $appDuration = ($appEndTime - $appStartTime).TotalSeconds
        
        if ($appCompleted) {
            $appStatus = if ($appProcess.ExitCode -eq 0) { "✅ PASSED" } else { "❌ FAILED (Exit: $($appProcess.ExitCode))" }
        } else {
            $appProcess.Kill()
            $appStatus = "❌ TIMEOUT"
        }
        
        $appTests += "$($app.Name): $appStatus (Duration: $([math]::Round($appDuration, 2))s)"
    } else {
        $appTests += "$($app.Name): ❌ NOT FOUND"
    }
}

Write-ReportSection "Sample Applications" ($appTests -join "`n")

# 5. Memory Testing (if not skipped)
if (!$SkipMemoryTests) {
    Write-Report "Running memory tests..." "INFO"
    $memoryTests = @()
    
    $drmemoryPath = "${env:ProgramFiles(x86)}\Dr. Memory\bin64\drmemory.exe"
    
    if (Test-Path $drmemoryPath) {
        foreach ($app in $sampleApps) {
            if (Test-Path $app.Path) {
                Write-Report "Memory testing $($app.Name)..." "INFO"
                
                $memLogDir = Join-Path $reportPath "memory_$($app.Name)"
                $memProcess = Start-Process -FilePath $drmemoryPath -ArgumentList "-logdir", $memLogDir, "-brief", "--", $app.Path -PassThru -WindowStyle Hidden
                
                $memCompleted = $memProcess.WaitForExit(120000) # 2 minute timeout
                
                if ($memCompleted) {
                    # Parse Dr. Memory results
                    $memLog = Get-ChildItem "$memLogDir\*.txt" | Get-Content -ErrorAction SilentlyContinue
                    $errors = ($memLog | Where-Object { $_ -match "Error|LEAK|UNINITIALIZED" }).Count
                    $warnings = ($memLog | Where-Object { $_ -match "Warning|POSSIBLE" }).Count
                    
                    $memStatus = if ($errors -eq 0) { "✅ CLEAN" } else { "❌ $errors errors, $warnings warnings" }
                } else {
                    $memProcess.Kill()
                    $memStatus = "❌ TIMEOUT"
                }
                
                $memoryTests += "$($app.Name): $memStatus"
            }
        }
    } else {
        $memoryTests += "❌ Dr. Memory not found - skipping memory tests"
    }
    
    Write-ReportSection "Memory Tests" ($memoryTests -join "`n")
}

# 6. Performance Benchmarks (if not skipped)
if (!$SkipPerformanceTests) {
    Write-Report "Running performance benchmarks..." "INFO"
    $perfTests = @()
    
    # Look for benchmark executables
    $benchmarkExes = Get-ChildItem "$BuildDir\*benchmark*.exe" -Recurse -ErrorAction SilentlyContinue
    
    if ($benchmarkExes) {
        foreach ($benchmark in $benchmarkExes) {
            Write-Report "Running benchmark: $($benchmark.Name)..." "INFO"
            
            $perfStartTime = Get-Date
            $perfProcess = Start-Process -FilePath $benchmark.FullName -PassThru -RedirectStandardOutput "$reportPath\$($benchmark.BaseName)_perf.txt" -WindowStyle Hidden
            
            $perfCompleted = $perfProcess.WaitForExit(180000) # 3 minute timeout
            $perfEndTime = Get-Date
            $perfDuration = ($perfEndTime - $perfStartTime).TotalSeconds
            
            if ($perfCompleted) {
                $perfStatus = if ($perfProcess.ExitCode -eq 0) { "✅ COMPLETED" } else { "❌ FAILED" }
            } else {
                $perfProcess.Kill()
                $perfStatus = "❌ TIMEOUT"
            }
            
            $perfTests += "$($benchmark.BaseName): $perfStatus (Duration: $([math]::Round($perfDuration, 2))s)"
        }
    } else {
        $perfTests += "ℹ️ No benchmark executables found"
    }
    
    Write-ReportSection "Performance Benchmarks" ($perfTests -join "`n")
}

# 7. System Information
Write-Report "Collecting system information..." "INFO"
$sysInfo = @()

$sysInfo += "**Operating System:** $(Get-WmiObject -Class Win32_OperatingSystem | ForEach-Object { "$($_.Caption) $($_.Version)" })"
$sysInfo += "**Processor:** $(Get-WmiObject -Class Win32_Processor | ForEach-Object { $_.Name })"
$sysInfo += "**Total RAM:** $([math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)) GB"
$sysInfo += "**Available RAM:** $([math]::Round((Get-WmiObject -Class Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2)) MB"

Write-ReportSection "System Information" ($sysInfo -join "`n")

# 8. Generate Summary
Write-Report "Generating validation summary..." "INFO"

$overallStatus = "✅ PASSED"
$issues = @()

if (!$vsFound) { $issues += "Visual Studio Build Tools missing"; $overallStatus = "❌ FAILED" }
if ($testStatus -ne "✅ PASSED") { $issues += "Unit tests failed"; $overallStatus = "❌ FAILED" }
if ($appTests | Where-Object { $_ -match "❌" }) { $issues += "Sample application failures"; $overallStatus = "⚠️ PARTIAL" }

$summary = @()
$summary += "**Overall Status:** $overallStatus"
if ($issues) {
    $summary += "**Issues Found:**"
    $summary += ($issues | ForEach-Object { "- $_" })
}
$summary += ""
$summary += "**Validation completed at:** $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$summary += "**Total duration:** $([math]::Round(((Get-Date) - $testStartTime).TotalMinutes, 2)) minutes"

Write-ReportSection "Validation Summary" ($summary -join "`n")

# Final log entries
Write-Report "Validation completed with status: $overallStatus" "INFO"
Write-Report "Report saved to: $reportFile" "INFO"
Write-Report "Log saved to: $logFile" "INFO"

# Display final summary
Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "NNTrainer Windows Build Validation Complete" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Status: $overallStatus"
Write-Host "Report: $reportFile"
Write-Host "Logs: $logFile"

if ($issues) {
    Write-Host "`nIssues found:" -ForegroundColor Yellow
    foreach ($issue in $issues) {
        Write-Host "- $issue" -ForegroundColor Red
    }
}

# Return appropriate exit code
if ($overallStatus -eq "✅ PASSED") {
    exit 0
} elseif ($overallStatus -eq "⚠️ PARTIAL") {
    exit 1
} else {
    exit 2
}