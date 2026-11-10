& (Join-Path $PSScriptRoot "_run_benchmark.ps1") -Dataset "wine_quality" -Macs @(
    "1.0e8",   # B1 (100 MMAC)
    "3.0e8",   # B2 (300 MMAC)
    "1.0e9",   # B3 (1.0 GMAC)
    "2.5e9",   # B4 (2.5 GMAC)
    "4.0e9"    # B5 (4.0 GMAC)
)
