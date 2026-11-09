"""
Script per generare l'Albero Gerarchico Tassonomico con Paradigmi Generali di Learning
- Learning unificato a livello di paradigma generale:
    * BPTT (Backpropagation Through Time): LTC e SNN (Surrogate GD)
    * BP (Backpropagation Statica): KAN e SNN (Conversion)
    * Regole Locali: FF, CHL, SNN (STDP)
- Profilo Operativo: MAC vs AC
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp"

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def draw_general_tree(output_prefix="tree_taxonomy"):
    fig, ax = plt.subplots(figsize=(20, 13.5), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 13.5)
    ax.axis("off")

    # Sfondo
    bg = patches.FancyBboxPatch(
        (0.3, 0.3), 19.4, 12.9,
        boxstyle="round,pad=0.2,rounding_size=0.4",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=2.5, zorder=0
    )
    ax.add_patch(bg)

    # Titolo
    ax.text(
        10.0, 12.9, "ALBERO GERARCHICO: PARADIGMI GENERALI (EdgeLearningBench)",
        fontsize=20, fontweight="bold", color="#0F172A", ha="center", va="center"
    )

    # Paradigmi con BPTT esplicito e categorie generali
    paradigms = [
        {
            "name": "KAN",
            "border": "#475569", "bg": "#FFFFFF", "text": "#1E293B",
            "dims": [
                ("CALCOLO", "KAT", "#1D4ED8", "#EFF6FF", "#3B82F6"),
                ("LEARNING", "BP", "#9F1239", "#FFF1F2", "#F43F5E"),
                ("PROFILO", "MAC", "#065F46", "#ECFDF5", "#10B981")
            ]
        },
        {
            "name": "FORWARD-FORWARD (FF)",
            "border": "#BE123C", "bg": "#FFF1F2", "text": "#9F1239",
            "dims": [
                ("CALCOLO", "UAT", "#1D4ED8", "#EFF6FF", "#3B82F6"),
                ("LEARNING", "Regole Locali", "#9F1239", "#FFF1F2", "#F43F5E"),
                ("PROFILO", "MAC", "#065F46", "#ECFDF5", "#10B981")
            ]
        },
        {
            "name": "LTC",
            "border": "#1D4ED8", "bg": "#EFF6FF", "text": "#1E40AF",
            "dims": [
                ("CALCOLO", "Neurone LTC (ODE)", "#1D4ED8", "#EFF6FF", "#3B82F6"),
                ("LEARNING", "BPTT", "#9F1239", "#FFF1F2", "#F43F5E"),
                ("PROFILO", "MAC", "#065F46", "#ECFDF5", "#10B981")
            ]
        },
        {
            "name": "SNN (Surrogate GD)",
            "border": "#0D9488", "bg": "#F0FDFA", "text": "#0F766E",
            "dims": [
                ("CALCOLO", "Event-Driven SNN", "#1D4ED8", "#EFF6FF", "#3B82F6"),
                ("LEARNING", "BPTT", "#9F1239", "#FFF1F2", "#F43F5E"),
                ("PROFILO", "AC", "#065F46", "#ECFDF5", "#10B981")
            ]
        },
        {
            "name": "SNN (Conversion)",
            "border": "#0D9488", "bg": "#F0FDFA", "text": "#0F766E",
            "dims": [
                ("CALCOLO", "Event-Driven SNN", "#1D4ED8", "#EFF6FF", "#3B82F6"),
                ("LEARNING", "BP", "#9F1239", "#FFF1F2", "#F43F5E"),
                ("PROFILO", "AC", "#065F46", "#ECFDF5", "#10B981")
            ]
        },
        {
            "name": "SNN (STDP)",
            "border": "#D97706", "bg": "#FEF3C7", "text": "#92400E",
            "dims": [
                ("CALCOLO", "Event-Driven SNN", "#1D4ED8", "#EFF6FF", "#3B82F6"),
                ("LEARNING", "Regole Locali", "#9F1239", "#FFF1F2", "#F43F5E"),
                ("PROFILO", "AC", "#065F46", "#ECFDF5", "#10B981")
            ]
        },
        {
            "name": "CHL",
            "border": "#7E22CE", "bg": "#FAF5FF", "text": "#6B21A8",
            "dims": [
                ("CALCOLO", "Equilibrio Ricorrente", "#1D4ED8", "#EFF6FF", "#3B82F6"),
                ("LEARNING", "Regole Locali", "#9F1239", "#FFF1F2", "#F43F5E"),
                ("PROFILO", "MAC", "#065F46", "#ECFDF5", "#10B981")
            ]
        }
    ]

    # Coordinate X
    x_root = 1.2
    x_model = 4.8
    x_dim = 9.8
    x_feat = 15.2

    y_starts = [11.3, 9.6, 7.9, 6.2, 4.5, 2.8, 1.1]
    y_root = 6.2

    def draw_bezier(p1, p2, color="#94A3B8", lw=1.8):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        verts = [(x1, y1), (x1 + dx * 0.45, y1), (x1 + dx * 0.45, y2), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        patch = patches.PathPatch(Path(verts, codes), facecolor="none", edgecolor=color, lw=lw, alpha=0.85, zorder=2)
        ax.add_patch(patch)

    ax.text(
        x_root, y_root, "TASSONOMIA\nDEI PARADIGMI",
        fontsize=10.5, fontweight="bold", color="#0F172A", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#E2E8F0", edgecolor="#64748B", lw=2.4, zorder=5)
    )

    for idx, p in enumerate(paradigms):
        y_c = y_starts[idx]
        y_sub = [y_c + 0.36, y_c, y_c - 0.36]

        draw_bezier((x_root + 1.2, y_root), (x_model - 1.8, y_c), color=p["border"], lw=2.0)

        ax.text(
            x_model, y_c, p["name"],
            fontsize=10.0, fontweight="bold", color=p["text"], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.42", facecolor=p["bg"], edgecolor=p["border"], lw=2.4, zorder=5)
        )

        for d_idx, (d_name, feat, t_col, bg_col, b_col) in enumerate(p["dims"]):
            y_r = y_sub[d_idx]

            draw_bezier((x_model + 1.8, y_c), (x_dim - 1.5, y_r), color=b_col, lw=1.5)

            ax.text(
                x_dim, y_r, d_name,
                fontsize=8.5, fontweight="bold", color=t_col, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.28", facecolor=bg_col, edgecolor=b_col, lw=1.6, zorder=5)
            )

            draw_bezier((x_dim + 1.5, y_r), (x_feat - 2.5, y_r), color="#CBD5E1", lw=1.3)

            ax.text(
                x_feat, y_r, feat,
                fontsize=9.2, fontweight="bold", color="#1E293B", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFFFFF", edgecolor=b_col, lw=1.6, zorder=5)
            )

    # Intestazioni Colonne
    ax.text(x_root, 12.3, "RADICE", fontsize=10.5, fontweight="bold", color="#64748B", ha="center")
    ax.text(x_model, 12.3, "PARADIGMA", fontsize=10.5, fontweight="bold", color="#64748B", ha="center")
    ax.text(x_dim, 12.3, "DIMENSIONE", fontsize=10.5, fontweight="bold", color="#64748B", ha="center")
    ax.text(x_feat, 12.3, "PARADIGMA GENERALE", fontsize=10.5, fontweight="bold", color="#64748B", ha="center")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{output_prefix}.svg", bbox_inches="tight")
    plt.close()
    print("Albero con BPTT Generale generato con successo!")

if __name__ == "__main__":
    draw_general_tree()
