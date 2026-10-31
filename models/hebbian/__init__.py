"""
Modulo Hebbian Learning: Contrastive Hebbian Learning (CHL).

Implementa il paradigma bio-ispirato Contrastive Hebbian Learning end-to-end
con dinamiche ricorrenti e fasi libera / bloccata (zero backpropagation).
"""

from models.hebbian.chl import CHLNet, build_chl, chl_run

__all__ = [
    "CHLNet",
    "build_chl",
    "chl_run",
]


