& (Join-Path $PSScriptRoot "_run_benchmark.ps1") -Dataset "cifar10" -Macs @(
    "1.0e12",  # B1 (1.0 TMAC)
    "5.0e12",  # B2 (5.0 TMAC)
    "2.0e13",  # B3 (20 TMAC)
    "5.0e13",  # B4 (50 TMAC)
    "1.5e14"   # B5 (150 TMAC)
)
