# monitor.ps1
# Prints: GPU util + GPU memory, CPU util, RAM used% every 3 seconds
# Requires: nvidia-smi available (NVIDIA driver installed)

$intervalSec = 3

function Get-CpuUsagePercent {
    # Uses the built-in perf counter; returns average CPU across all cores
    $v = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples[0].CookedValue
    [math]::Round($v, 1)
}

function Get-RamUsage {
    # Use CIM for total/available memory
    $os = Get-CimInstance Win32_OperatingSystem
    $totalBytes = [double]$os.TotalVisibleMemorySize * 1KB
    $freeBytes  = [double]$os.FreePhysicalMemory      * 1KB
    $usedBytes  = $totalBytes - $freeBytes
    $usedPct    = ($usedBytes / $totalBytes) * 100.0

    [pscustomobject]@{
        TotalGB = [math]::Round($totalBytes / 1GB, 2)
        UsedGB  = [math]::Round($usedBytes  / 1GB, 2)
        UsedPct = [math]::Round($usedPct, 1)
    }
}

function Get-GpuStats {
    # Query multiple GPUs; output objects
    $nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvsmi) { return $null }

    $args = @(
        '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu'
        '--format=csv,noheader,nounits'
    )

    $lines = & nvidia-smi @args 2>$null
    if (-not $lines) { return $null }

    $lines | ForEach-Object {
        $p = $_.Split(',') | ForEach-Object { $_.Trim() }
        [pscustomobject]@{
            Index        = [int]$p[0]
            Name         = $p[1]
            GpuUtilPct   = [int]$p[2]
            MemUtilPct   = [int]$p[3]
            MemUsedMiB   = [int]$p[4]
            MemTotalMiB  = [int]$p[5]
            TempC        = [int]$p[6]
        }
    }
}

Write-Host "Monitoring every $intervalSec seconds. Press Ctrl+C to stop." -ForegroundColor Cyan

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    $cpuPct = Get-CpuUsagePercent
    $ram = Get-RamUsage
    $gpus = Get-GpuStats

    Clear-Host
    Write-Host "[$ts]" -ForegroundColor Yellow
    Write-Host ("CPU: {0}%    RAM: {1}% ({2} / {3} GB)" -f $cpuPct, $ram.UsedPct, $ram.UsedGB, $ram.TotalGB)

    if (-not $gpus) {
        Write-Host "GPU: nvidia-smi not found or no NVIDIA GPU detected." -ForegroundColor Red
    } else {
        Write-Host ""
        Write-Host "NVIDIA GPU(s):"
        $gpus | Format-Table -AutoSize `
            Index, Name, TempC, `
            @{Name="GPU%"; Expression={$_.GpuUtilPct}}, `
            @{Name="Mem%"; Expression={$_.MemUtilPct}}, `
            @{Name="MemUsed(MiB)"; Expression={$_.MemUsedMiB}}, `
            @{Name="MemTotal(MiB)"; Expression={$_.MemTotalMiB}}
    }

    Start-Sleep -Seconds $intervalSec
}
