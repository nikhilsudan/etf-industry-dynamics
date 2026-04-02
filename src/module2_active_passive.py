import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit
from pathlib import Path
from src.data_loader import load_etf_launches

FIGURES = Path(__file__).resolve().parent.parent / "outputs" / "figures"
TABLES  = Path(__file__).resolve().parent.parent / "outputs" / "tables"


# ── Logistic S-curve model ───────────────────────────────────────
def logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))


def run():
    df = load_etf_launches()
    t  = np.arange(len(df))

    active_share = df["active_share"].values
    active_cum   = df["active_launches"].cumsum().values
    passive_cum  = df["passive_launches"].cumsum().values

    # ── Fit logistic curve to active share ───────────────────────
    popt, _ = curve_fit(
        logistic, t, active_share,
        p0=[0.6, 0.15, 10],
        bounds=([0.4, 0.01, -5], [1.0, 5.0, 20]),
        maxfev=10000
    )
    L_fit, k_fit, t0_fit = popt
    fitted_share = logistic(t, L_fit, k_fit, t0_fit)

    # ── Inflection point label ───────────────────────────────────
    inflection_idx = int(round(t0_fit))
    inflection_idx = max(0, min(inflection_idx, len(df) - 1))

    if t0_fit < 0:
        inflection_label = "pre-2022 (model estimate)"
    elif inflection_idx >= len(df) - 1:
        inflection_label = "post-2025 (model estimate)"
    else:
        inflection_label = df["period"].iloc[inflection_idx]

    # ── Adoption stage interpretation ────────────────────────────
    current_share = active_share[-1]
    pct_of_ceiling = (current_share / L_fit) * 100
    if pct_of_ceiling < 50:
        stage = "Early Growth"
    elif pct_of_ceiling < 85:
        stage = "Accelerating (Past Inflection)"
    else:
        stage = "Maturing / Approaching Saturation"

    print(f"\n── Module 2: Active vs Passive ETF Shift ──")
    print(f"S-curve ceiling (L):           {L_fit*100:.1f}%")
    print(f"Growth rate (k):               {k_fit:.4f}")
    print(f"Inflection point:              {inflection_label}")
    print(f"Current active share:          {current_share*100:.1f}%")
    print(f"% of ceiling reached:          {pct_of_ceiling:.1f}%")
    print(f"Adoption stage:                {stage}")

    # ── Save summary table ───────────────────────────────────────
    out_df = df[["period", "passive_launches",
                 "active_launches", "total_launches",
                 "active_share"]].copy()
    out_df["fitted_active_share"] = fitted_share
    out_df.to_csv(TABLES / "module2_summary.csv", index=False)

    # ════════════════════════════════════════════════════════════
    # FIGURE 2
    # ════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")

    colors = {
        "active":  "#c084fc",
        "passive": "#60a5fa",
        "fit":     "#ffffff",
        "inflect": "#f0c040"
    }
    labels = df["period"].tolist()

    # ── Panel A — Stacked bar: launches per quarter ──────────────
    ax = axes[0]
    ax.bar(t, df["passive_launches"], color=colors["passive"],
           alpha=0.85, label="Passive Launches")
    ax.bar(t, df["active_launches"],  color=colors["active"],
           alpha=0.85, label="Active Launches",
           bottom=df["passive_launches"])
    ax.set_xticks(t)
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       fontsize=8, color="white")
    ax.set_ylabel("Number of ETF Launches", color="white")
    ax.set_title("Panel A — Global ETF Launches per Quarter: Active vs Passive",
                 color="white", fontsize=11, pad=10)
    ax.legend(framealpha=0, labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")

    # ── Panel B — Active share + S-curve fit ─────────────────────
    ax = axes[1]
    ax.scatter(t, active_share * 100, color=colors["active"],
               zorder=5, s=60, label="Observed Active Share (%)")
    ax.plot(t, fitted_share * 100, color=colors["fit"],
            linewidth=2, linestyle="--",
            label=f"Logistic Fit  L={L_fit*100:.0f}%  k={k_fit:.3f}")

    # Only draw inflection line if it falls within data range
    if 0 < t0_fit < len(df) - 1:
        ax.axvline(t0_fit, color=colors["inflect"], linewidth=1.2,
                   linestyle=":", alpha=0.8,
                   label=f"Inflection → {inflection_label}")
    else:
        # Add a text note instead
        ax.text(0.02, 0.95,
                f"Inflection point: {inflection_label}\n"
                f"Adoption stage: {stage}\n"
                f"At {pct_of_ceiling:.0f}% of ceiling",
                transform=ax.transAxes, fontsize=8,
                color=colors["inflect"], verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#1a1a2e", edgecolor="#f0c040",
                          alpha=0.8))

    ax.axhline(L_fit * 100, color=colors["passive"], linewidth=0.8,
               linestyle=":", alpha=0.6,
               label=f"Model ceiling: {L_fit*100:.0f}%")
    ax.set_xticks(t)
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       fontsize=8, color="white")
    ax.set_ylabel("Active Share of Launches (%)", color="white")
    ax.set_title("Panel B — Active ETF Adoption: Logistic S-Curve Fit",
                 color="white", fontsize=11, pad=10)
    ax.legend(framealpha=0, labelcolor="white", fontsize=8,
              loc="lower right")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    # ── Panel C — Cumulative launches ────────────────────────────
    ax = axes[2]
    ax.plot(t, active_cum,  color=colors["active"],
            linewidth=2.5, label="Cumulative Active Launches")
    ax.plot(t, passive_cum, color=colors["passive"],
            linewidth=2.5, label="Cumulative Passive Launches")
    ax.fill_between(t, active_cum, passive_cum,
                    where=active_cum >= passive_cum,
                    alpha=0.15, color=colors["active"],
                    label="Active > Passive zone")
    ax.fill_between(t, active_cum, passive_cum,
                    where=active_cum < passive_cum,
                    alpha=0.10, color=colors["passive"],
                    label="Passive > Active zone")
    ax.set_xticks(t)
    ax.set_xticklabels(labels, rotation=45, ha="right",
                       fontsize=8, color="white")
    ax.set_ylabel("Cumulative Launches", color="white")
    ax.set_title("Panel C — Cumulative Active vs Passive ETF Launches",
                 color="white", fontsize=11, pad=10)
    ax.legend(framealpha=0, labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")

    fig.suptitle(
        "Module 2 — Active ETF Adoption: S-Curve Analysis (2022–2025)\n"
        "Inspired by Jane Street Global ETF Landscape Report, Dec 2025",
        color="white", fontsize=13, y=1.01, fontweight="bold"
    )

    plt.tight_layout()
    out = FIGURES / "module2_active_passive.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Figure saved → {out}")


if __name__ == "__main__":
    run()