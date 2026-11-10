& (Join-Path $PSScriptRoot "_run_benchmark.ps1") -Dataset "breast_cancer" -Macs @(
    "1.0e7",   # B1 (10 MMAC)
    "5.0e7",   # B2 (50 MMAC)
    "2.0e8",   # B3 (200 MMAC)
    "5.0e8",   # B4 (500 MMAC)
    "1.5e9"    # B5 (1.5 GMAC)
)
