"""
Script per generare l'Albero di Convergenza Tassonomica (Dimension Convergence Network)
Flusso:
[1. Macro-Dimensioni] -> [2. Classi delle Dimensioni] -> [3. Paradigmi (con 3 frecce convergenti)]
Esempio:
FF riceve 3 frecce:
- Blu da UAT (Calcolo)
- Rossa da Regole Locali (Learning)
- Verde da MAC (Profilo)
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp"

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def draw_convergence_tree(output_prefix="tree_by_dimensions"):
    fig, ax = plt.subplots(figsize=(23, 14.5), dpi=300)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 14.5)
    ax.axis("off")

    # Sfondo Canvas
    bg = patches.FancyBboxPatch(
        (0.3, 0.3), 22.4, 13.9,
        boxstyle="round,pad=0.2,rounding_size=0.4",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=2.5, zorder=0
    )
    ax.add_patch(bg)

    # Titolo
    ax.text(
        11.5, 14.0, "RETE DI CONVERGENZA TASSONOMICA: DIMENSIONI → CLASSI → PARADIGMI",
        fontsize=19, fontweight="bold", color="#0F172A", ha="center", va="center"
    )

    # ── 1. Coordinate X delle 3 Colonne ──────────────────────────────────────
    x_dim = 2.8       # Colonna 1: Macro-Dimensioni
    x_class = 9.2     # Colonna 2: Classi delle Dimensioni
    x_model = 19.0    # Colonna 3: Paradigmi (I 7 Modelli elencati una sola volta)

    # ── 2. Posizioni Y delle Macro-Dimensioni ────────────────────────────────
    dims_pos = {
        "calcolo": {"y": 10.7, "name": "1. PARADIGMA DI CALCOLO", "color": "#1D4ED8", "bg": "#EFF6FF", "border": "#3B82F6"},
        "learning": {"y": 6.3, "name": "2. PARADIGMA DI LEARNING", "color": "#9F1239", "bg": "#FFF1F2", "border": "#F43F5E"},
        "profilo": {"y": 2.2, "name": "3. PROFILO OPERATIVO", "color": "#065F46", "bg": "#ECFDF5", "border": "#10B981"}
    }

    # ── 3. Posizioni Y delle Classi (Colonna Centrale) ───────────────────────
    classes_pos = {
        # Calcolo (5 classi)
        "kat": {"y": 12.3, "name": "KAT", "dim": "calcolo", "color": "#1D4ED8", "border": "#3B82F6", "bg": "#EFF6FF"},
        "uat": {"y": 11.5, "name": "UAT", "dim": "calcolo", "color": "#1D4ED8", "border": "#3B82F6", "bg": "#EFF6FF"},
        "ltc": {"y": 10.7, "name": "Neurone LTC (ODE)", "dim": "calcolo", "color": "#1D4ED8", "border": "#3B82F6", "bg": "#EFF6FF"},
        "snn": {"y": 9.9, "name": "Event-Driven SNN", "dim": "calcolo", "color": "#1D4ED8", "border": "#3B82F6", "bg": "#EFF6FF"},
        "eq":  {"y": 9.1, "name": "Equilibrio Ricorrente", "dim": "calcolo", "color": "#1D4ED8", "border": "#3B82F6", "bg": "#EFF6FF"},
        # Learning (3 classi)
        "bp":     {"y": 7.3, "name": "BP", "dim": "learning", "color": "#9F1239", "border": "#F43F5E", "bg": "#FFF1F2"},
        "bptt":   {"y": 6.3, "name": "BPTT", "dim": "learning", "color": "#9F1239", "border": "#F43F5E", "bg": "#FFF1F2"},
        "loc":    {"y": 5.3, "name": "Regole Locali", "dim": "learning", "color": "#9F1239", "border": "#F43F5E", "bg": "#FFF1F2"},
        # Profilo (2 classi)
        "mac": {"y": 2.8, "name": "MAC", "dim": "profilo", "color": "#065F46", "border": "#10B981", "bg": "#ECFDF5"},
        "ac":  {"y": 1.6, "name": "AC", "dim": "profilo", "color": "#065F46", "border": "#10B981", "bg": "#ECFDF5"}
    }

    # ── 4. Posizioni Y dei 7 Paradigmi (Colonna Destra) ──────────────────────
    models_pos = {
        "kan":      {"y": 12.0, "name": "KAN", "calcolo": "kat", "learning": "bp", "profilo": "mac", "border": "#475569", "bg": "#FFFFFF"},
        "ff":       {"y": 10.3, "name": "FORWARD-FORWARD (FF)", "calcolo": "uat", "learning": "loc", "profilo": "mac", "border": "#BE123C", "bg": "#FFF1F2"},
        "ltc":      {"y": 8.6, "name": "LTC", "calcolo": "ltc", "learning": "bptt", "profilo": "mac", "border": "#1D4ED8", "bg": "#EFF6FF"},
        "snn_sg":   {"y": 6.9, "name": "SNN (Surrogate GD)", "calcolo": "snn", "learning": "bptt", "profilo": "ac", "border": "#0D9488", "bg": "#F0FDFA"},
        "snn_conv": {"y": 5.2, "name": "SNN (Conversion)", "calcolo": "snn", "learning": "bp", "profilo": "ac", "border": "#0D9488", "bg": "#F0FDFA"},
        "snn_stdp": {"y": 3.5, "name": "★ SNN (STDP) ★", "calcolo": "snn", "learning": "loc", "profilo": "ac", "border": "#D97706", "bg": "#FEF3C7", "is_gold": True},
        "chl":      {"y": 1.8, "name": "CHL", "calcolo": "eq", "learning": "loc", "profilo": "mac", "border": "#7E22CE", "bg": "#FAF5FF"}
    }

    # Funzione per curve Bezier con freccia terminale
    def draw_connecting_arrow(p1, p2, color, lw=1.6, alpha=0.65):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        verts = [(x1, y1), (x1 + dx * 0.45, y1), (x1 + dx * 0.45, y2), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        patch = patches.PathPatch(Path(verts, codes), facecolor="none", edgecolor=color, lw=lw, alpha=alpha, zorder=2)
        ax.add_patch(patch)
        # Punta della freccia a destra
        ax.annotate(
            "", xy=(x2, y2), xytext=(x2 - 0.25, y2),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=11), zorder=3
        )

    # ── 5. Disegno Collegamenti: Dimensione -> Classi ────────────────────────
    for c_id, c in classes_pos.items():
        dim = dims_pos[c["dim"]]
        draw_connecting_arrow((x_dim + 2.0, dim["y"]), (x_class - 1.6, c["y"]), color=dim["border"], lw=1.8, alpha=0.75)

    # ── 6. Disegno Collegamenti: Classi -> Paradigmi (Le 3 Frecce per Modello) ──
    for m_id, m in models_pos.items():
        # 1. Freccia da Calcolo (Blu)
        c_calc = classes_pos[m["calcolo"]]
        draw_connecting_arrow((x_class + 1.6, c_calc["y"]), (x_model - 2.2, m["y"] + 0.18), color="#2563EB", lw=1.8, alpha=0.7)

        # 2. Freccia da Learning (Rosso)
        c_learn = classes_pos[m["learning"]]
        draw_connecting_arrow((x_class + 1.6, c_learn["y"]), (x_model - 2.2, m["y"]), color="#E11D48", lw=1.8, alpha=0.7)

        # 3. Freccia da Profilo (Verde)
        c_prof = classes_pos[m["profilo"]]
        draw_connecting_arrow((x_class + 1.6, c_prof["y"]), (x_model - 2.2, m["y"] - 0.18), color="#059669", lw=1.8, alpha=0.7)

    # ── 7. Disegno Nodi delle 3 Colonne ──────────────────────────────────────
    # Colonna 1: Macro-Dimensioni
    for d_id, d in dims_pos.items():
        ax.text(
            x_dim, d["y"], d["name"],
            fontsize=10.5, fontweight="bold", color=d["color"], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=d["bg"], edgecolor=d["border"], lw=2.4, zorder=5)
        )

    # Colonna 2: Classi delle Dimensioni
    for c_id, c in classes_pos.items():
        ax.text(
            x_class, c["y"], c["name"],
            fontsize=9.2, fontweight="bold", color=c["color"], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.32", facecolor=c["bg"], edgecolor=c["border"], lw=1.8, zorder=5)
        )

    # Colonna 3: I 7 Paradigmi
    for m_id, m in models_pos.items():
        is_gold = m.get("is_gold", False)
        # Ombra
        shadow = patches.FancyBboxPatch(
            (x_model - 2.15 + 0.06, m["y"] - 0.35 - 0.06), 4.3, 0.70,
            boxstyle="round,pad=0.08,rounding_size=0.2", facecolor="#0F172A", alpha=0.08, zorder=6
        )
        ax.add_patch(shadow)

        # Badge Modello
        badge = patches.FancyBboxPatch(
            (x_model - 2.15, m["y"] - 0.35), 4.3, 0.70,
            boxstyle="round,pad=0.08,rounding_size=0.2",
            facecolor=m["bg"], edgecolor=m["border"], lw=2.4 if not is_gold else 3.2, zorder=7
        )
        ax.add_patch(badge)

        ax.text(
            x_model, m["y"], m["name"],
            fontsize=10.0, fontweight="bold", color="#0F172A", ha="center", va="center", zorder=8
        )

    # Intestazioni Colonne in Alto
    ax.text(x_dim, 13.0, "COLONNA 1: MACRO-DIMENSIONI", fontsize=10.5, fontweight="bold", color="#64748B", ha="center")
    ax.text(x_class, 13.0, "COLONNA 2: CLASSI TEORICHE", fontsize=10.5, fontweight="bold", color="#64748B", ha="center")
    ax.text(x_model, 13.0, "COLONNA 3: PARADIGMI (CONVERGENZA 3x)", fontsize=10.5, fontweight="bold", color="#64748B", ha="center")

    # Legenda Frecce in Basso
    ax.text(
        11.5, 0.7,
        "LEGENDA FRECCE:  [Blu] Freccia da Calcolo (KAT/UAT/LTC/SNN/Eq)   •   [Rosso] Freccia da Learning (BP/BPTT/Locali)   •   [Verde] Freccia da Profilo (MAC/AC)",
        fontsize=9.2, fontweight="bold", color="#475569", ha="center", va="center"
    )

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{output_prefix}.svg", bbox_inches="tight")
    plt.close()
    print("Rete di Convergenza (Albero con 3 frecce convergenti) generata con successo!")

if __name__ == "__main__":
    draw_convergence_tree()
