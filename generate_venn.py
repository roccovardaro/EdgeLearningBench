"""
Script per generare il Diagramma di Venn Tassonomico con Paradigmi Generali di Learning
- BPTT condiviso da LTC e SNN (Surrogate GD)
- BP condiviso da KAN e SNN (Conversion)
- Regole Locali condiviso da FF, CHL e SNN (STDP)
- Profilo Operativo: MAC vs AC
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp"

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_general_venn(output_prefix="venn_taxonomy"):
    fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 15)
    ax.axis("off")

    # Sfondo Canvas
    bg = patches.FancyBboxPatch(
        (0.4, 0.4), 19.2, 14.2,
        boxstyle="round,pad=0.2,rounding_size=0.4",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=2.5, zorder=0
    )
    ax.add_patch(bg)

    # Titolo
    ax.text(
        10.0, 14.35, "TASSONOMIA TRIASSIALE: PARADIGMI GENERALI (EdgeLearningBench)",
        fontsize=19, fontweight="bold", color="#0F172A", ha="center", va="center"
    )

    # Geometria dei 3 Cerchi
    R = 4.35
    cA = (10.0, 9.2)   # Calcolo (Top)
    cB = (6.8, 5.4)    # Learning (Bottom-Left)
    cC = (13.2, 5.4)   # Profilo (Bottom-Right)

    col_A = "#2563EB"  # Royal Blue (Calcolo)
    col_B = "#E11D48"  # Crimson (Learning)
    col_C = "#059669"  # Emerald (Profilo)

    for center, color in [(cA, col_A), (cB, col_B), (cC, col_C)]:
        ax.add_patch(plt.Circle(center, radius=R, fill=False, edgecolor=color, lw=6.0, alpha=0.15, zorder=1))
        ax.add_patch(plt.Circle(center, radius=R, fill=False, edgecolor=color, lw=3.0, alpha=0.75, zorder=1))
        ax.add_patch(plt.Circle(center, radius=R, facecolor=color, alpha=0.10, zorder=1))

    # Banner Titoli delle Macro-Dimensioni
    ax.text(
        10.0, 13.7, "PARADIGMA DI CALCOLO",
        fontsize=12.5, fontweight="bold", color="#1D4ED8", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#EFF6FF", edgecolor="#3B82F6", lw=2.2, alpha=0.98, zorder=15)
    )

    ax.text(
        3.2, 1.4, "PARADIGMA DI LEARNING",
        fontsize=12.0, fontweight="bold", color="#9F1239", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFF1F2", edgecolor="#F43F5E", lw=2.2, alpha=0.98, zorder=15)
    )

    ax.text(
        16.8, 1.4, "PROFILO OPERATIVO (MAC vs AC)",
        fontsize=12.0, fontweight="bold", color="#065F46", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ECFDF5", edgecolor="#10B981", lw=2.2, alpha=0.98, zorder=15)
    )

    # Sottocerchi Colorati
    def draw_subcircle(x, y, r, label, border_color, fill_color, is_dashed=False, is_ellipse=False, h=0):
        if is_ellipse:
            patch = patches.Ellipse(
                (x, y), width=r*2, height=h*2,
                facecolor=fill_color, edgecolor=border_color, lw=2.2,
                linestyle="--" if is_dashed else "-", alpha=0.45, zorder=3
            )
        else:
            patch = patches.Circle(
                (x, y), radius=r,
                facecolor=fill_color, edgecolor=border_color, lw=2.2,
                linestyle="--" if is_dashed else "-", alpha=0.45, zorder=3
            )
        ax.add_patch(patch)

        label_y = y + (h if is_ellipse else r) - 0.28
        ax.text(
            x, label_y, label,
            fontsize=8.5, fontweight="bold", color=border_color, ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=border_color, lw=1.2, alpha=0.95)
        )

    draw_subcircle(x=2.9, y=10.5, r=1.75, label="KAT  •  MAC", border_color="#64748B", fill_color="#E2E8F0", is_dashed=True)
    draw_subcircle(x=10.0, y=11.5, r=1.35, label="Neurone LTC (ODE)  •  BPTT", border_color="#0284C7", fill_color="#BAE6FD")
    draw_subcircle(x=3.8, y=5.3, r=1.65, label="Regole Locali  •  MAC", border_color="#BE123C", fill_color="#FECDD3")
    draw_subcircle(x=7.1, y=8.2, r=1.35, label="Regole Locali  •  MAC", border_color="#7E22CE", fill_color="#E9D5FF")
    draw_subcircle(x=13.2, y=8.2, r=1.65, h=1.85, label="Event-Driven SNN  •  AC", border_color="#0D9488", fill_color="#CCFBF1", is_ellipse=True)
    draw_subcircle(x=10.0, y=6.2, r=1.35, label="★ Regole Locali  •  AC ★", border_color="#D97706", fill_color="#FDE68A")

    # Badge dei Paradigmi
    def draw_badge(x, y, name, calcolo, learning, profilo, border_col, bg_col="#FFFFFF", is_gold=False):
        shadow = patches.FancyBboxPatch(
            (x - 1.75 + 0.05, y - 0.55 - 0.05), 3.5, 1.1,
            boxstyle="round,pad=0,rounding_size=0.22",
            facecolor="#0F172A", alpha=0.08, zorder=5
        )
        ax.add_patch(shadow)

        card = patches.FancyBboxPatch(
            (x - 1.75, y - 0.55), 3.5, 1.1,
            boxstyle="round,pad=0,rounding_size=0.22",
            facecolor=bg_col, edgecolor=border_col, lw=2.2 if not is_gold else 3.2, zorder=6
        )
        ax.add_patch(card)

        h_bar = patches.FancyBboxPatch(
            (x - 1.75, y + 0.55 - 0.36), 3.5, 0.36,
            boxstyle="round,pad=0,rounding_size=0.22",
            facecolor=border_col if not is_gold else "#D97706", edgecolor="none", zorder=7
        )
        ax.add_patch(h_bar)

        ax.text(x, y + 0.37, name, fontsize=10.0, fontweight="bold", color="#FFFFFF", ha="center", va="center", zorder=8)
        ax.text(x, y + 0.02, f"Calcolo: {calcolo}", fontsize=8.0, fontweight="bold", color="#1D4ED8", ha="center", va="center", zorder=8)
        ax.text(x, y - 0.20, f"Learning: {learning}", fontsize=8.2, fontweight="bold", color="#BE123C", ha="center", va="center", zorder=8)
        ax.text(x, y - 0.40, f"Profilo: {profilo}", fontsize=8.2, fontweight="bold", color="#047857", ha="center", va="center", zorder=8)

    # 1. KAN
    draw_badge(2.9, 10.4, "KAN", calcolo="KAT", learning="BP", profilo="MAC", border_col="#475569")

    # 2. LTC (BPTT su ODE)
    draw_badge(10.0, 11.4, "LTC", calcolo="Neurone LTC (ODE)", learning="BPTT", profilo="MAC", border_col="#0284C7", bg_col="#EFF6FF")

    # 3. FF
    draw_badge(3.8, 5.2, "FF", calcolo="UAT", learning="Regole Locali", profilo="MAC", border_col="#BE123C", bg_col="#FFF1F2")

    # 4. CHL
    draw_badge(7.1, 8.1, "CHL", calcolo="Equilibrio Ricorrente", learning="Regole Locali", profilo="MAC", border_col="#7E22CE", bg_col="#FAF5FF")

    # 5. SNN Surrogate GD (BPTT su Spike)
    draw_badge(13.2, 8.9, "SNN (Surrogate GD)", calcolo="Event-Driven SNN", learning="BPTT", profilo="AC", border_col="#0D9488", bg_col="#F0FDFA")

    # 6. SNN Conversion (BP su ANN)
    draw_badge(13.2, 7.5, "SNN (Conversion)", calcolo="Event-Driven SNN", learning="BP", profilo="AC", border_col="#0D9488", bg_col="#F0FDFA")

    # 7. SNN STDP
    draw_badge(10.0, 6.1, "★ SNN (STDP) ★", calcolo="Event-Driven SNN", learning="Regole Locali", profilo="AC", border_col="#D97706", bg_col="#FFFBEB", is_gold=True)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{output_prefix}.svg", bbox_inches="tight")
    plt.close()
    print("Venn con Paradigmi Generali (BPTT condiviso) generato con successo!")

if __name__ == "__main__":
    draw_general_venn()
