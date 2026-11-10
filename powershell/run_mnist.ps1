& (Join-Path $PSScriptRoot "_run_benchmark.ps1") -Dataset "mnist" -Macs @(
    "3.0e11",  # B1 (300 GMAC)
    "1.5e12",  # B2 (1.5 TMAC)
    "5.0e12",  # B3 (5.0 TMAC)
    "2.0e13",  # B4 (20 TMAC)
    "8.0e13"   # B5 (80 TMAC)
)
