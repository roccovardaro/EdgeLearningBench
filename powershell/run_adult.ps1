& (Join-Path $PSScriptRoot "_run_benchmark.ps1") -Dataset "adult" -Macs @(
    "5.0e9",   # B1 (5 GMAC)
    "1.5e10",  # B2 (15 GMAC)
    "4.5e10",  # B3 (45 GMAC)
    "1.0e11",  # B4 (100 GMAC)
    "3.0e11"   # B5 (300 GMAC)
)
