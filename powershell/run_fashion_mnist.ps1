& (Join-Path $PSScriptRoot "_run_benchmark.ps1") -Dataset "fashion_mnist" -Macs @(
    "5.0e11",  # B1 (500 GMAC)
    "2.0e12",  # B2 (2.0 TMAC)
    "6.0e12",  # B3 (6.0 TMAC)
    "2.5e13",  # B4 (25 TMAC)
    "1.0e14"   # B5 (100 TMAC)
)
