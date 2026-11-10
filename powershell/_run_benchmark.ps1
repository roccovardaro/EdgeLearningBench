[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [string[]]$Macs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-Python {
    param([string]$ProjectRoot)

    $isWin = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT

    if ((Test-Path "Env:VIRTUAL_ENV") -and (-not [string]::IsNullOrWhiteSpace($env:VIRTUAL_ENV))) {
        $py = if ($isWin) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { Join-Path $env:VIRTUAL_ENV "bin/python" }
        if (Test-Path $py -PathType Leaf) { return $py }
    }

    if ((Test-Path "Env:CONDA_PREFIX") -and (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX))) {
        $py = if ($isWin) { Join-Path $env:CONDA_PREFIX "python.exe" } else { Join-Path $env:CONDA_PREFIX "bin/python" }
        if (Test-Path $py -PathType Leaf) { return $py }
    }

    $candidates = if ($isWin) {
        @((Join-Path $ProjectRoot ".venv\Scripts\python.exe"), (Join-Path $ProjectRoot "venv\Scripts\python.exe"))
    } else {
        @((Join-Path $ProjectRoot ".venv/bin/python"), (Join-Path $ProjectRoot "venv/bin/python"))
    }
    foreach ($py in $candidates) {
        if (Test-Path $py -PathType Leaf) { return $py }
    }

    foreach ($name in @("python", "python3", "py")) {
        $cmd = Get-Command -Name $name -TotalCount 1 -ErrorAction SilentlyContinue
        if ($null -ne $cmd) { return $cmd.Source }
    }

    throw "Python non trovato. Attiva un virtual environment o aggiungi python al PATH."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Resolve-Python -ProjectRoot $projectRoot
$resultsDir = Join-Path $projectRoot "results"

Push-Location $projectRoot
try {
    for ($i = 0; $i -lt $Macs.Count; $i++) {
        $level = "B{0}" -f ($i + 1)
        $mac = $Macs[$i]

        Write-Host ("`n=== {0} | {1} (--mac {2}) ===" -f $Dataset, $level, $mac) -ForegroundColor Cyan

        $before = @{}
        if (Test-Path $resultsDir -PathType Container) {
            foreach ($f in (Get-ChildItem -Path $resultsDir -File)) {
                $before[$f.Name] = $f.LastWriteTimeUtc
            }
        }

        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $pythonExe "run_all.py" "--dataset" $Dataset "--mac" $mac
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $prevEap
        }

        if ($exitCode -ne 0) {
            throw ("Errore su {0} ({1}, --mac {2}): exit code {3}" -f $Dataset, $level, $mac, $exitCode)
        }

        # Sposta i file generati per evitare che *_budget_summary.csv venga sovrascritto dal livello successivo
        if (Test-Path $resultsDir -PathType Container) {
            $targetDir = Join-Path (Join-Path $resultsDir $Dataset) ("{0}_{1}" -f $level, $mac)
            foreach ($f in (Get-ChildItem -Path $resultsDir -File)) {
                if ($f.Name -eq ".DS_Store") { continue }
                if ((-not $before.ContainsKey($f.Name)) -or ($f.LastWriteTimeUtc -gt $before[$f.Name])) {
                    if (-not (Test-Path $targetDir -PathType Container)) {
                        New-Item -Path $targetDir -ItemType Directory -Force | Out-Null
                    }
                    Move-Item -Path $f.FullName -Destination (Join-Path $targetDir $f.Name) -Force
                }
            }
        }
    }
} finally {
    Pop-Location
}
