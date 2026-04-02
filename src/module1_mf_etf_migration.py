import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit
from pathlib import Path
from src.data_loader import load_etf_flows, load_mf_flows

FIGURES = Path(__file__).resolve().parent.parent / "outputs" / "figures"
TABLES = Path(__file__).resolve().parent.parent / "outputs" / "tables"


# ── Exponential model ────────────────────────────────────────────
def exponential_model(t, A, r):
    return A * np.exp(r * t)


# ── Main analysis ────────────────────────────────────────────────
def run():
    etf = load_etf_flows()
    mf = load_mf_flows()

    # Merge on date
    df = pd.merge(etf, mf, on="date", suffixes=("_etf", "_mf"))
    df["t"] = np.arange(len(df))  # numeric time index for curve fitting

    # ── Cumulative flows (to show AUM-like growth trajectory) ────
    df["etf_cumulative"] = df["etf_net_flows_billions"].cumsum()
    df["mf_cumulative"]  = df["mf_net_flows_billions"].cumsum()

    # ── Exponential fit on cumulative ETF flows ──────────────────
    etf_cum = df["etf_cumulative"].values
    t       = df["t"].values

    popt, _ = curve_fit(exponential_model, t, etf_cum,
                        p0=[etf_cum[0], 0.05], maxfev=5000)
    A_fit, r_fit = popt
    etf_fitted = exponential_model(t, A_fit, r_fit)

    print(f"\n── Module 1: MF → ETF Migration ──")
    print(f"Exponential growth rate (r): {r_fit:.4f} per quarter")
    print(f"Doubling time: {np.log(2)/r_fit:.1f} quarters")

    # ── Rolling 4-quarter correlation ────────────────────────────
    df["rolling_corr"] = (
        df["etf_net_flows_billions"]
        .rolling(4)
        .corr(df["mf_net_flows_billions"])
    )

    print(f"Average rolling correlation (ETF vs MF flows): "
          f"{df['rolling_corr'].mean():.3f}")

    # ── Save summary table ───────────────────────────────────────
    summary = df[["date", "etf_net_flows_billions",
                  "mf_net_flows_billions", "rolling_corr"]].copy()
    summary.to_csv(TABLES / "module1_summary.csv", index=False)

    # ════════════════════════════════════════════════════════════
    # FIGURE 1 — Flow Divergence
    # ════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")

    colors = {"etf": "#00d4aa", "mf": "#ff6b6b", "corr": "#f0c040"}
    quarters = df["date"].dt.to_period("Q").astype(str)

    # Panel A — Quarterly flows bar chart
    ax = axes[0]
    x = np.arange(len(df))
    w = 0.4
    ax.bar(x - w/2, df["etf_net_flows_billions"],
           width=w, color=colors["etf"], alpha=0.85, label="ETF Net Flows")
    ax.bar(x + w/2, df["mf_net_flows_billions"],
           width=w, color=colors["mf"],  alpha=0.85, label="MF Net Flows")
    ax.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=45, ha="right",
                       fontsize=7, color="white")
    ax.set_ylabel("Net Flows ($B)", color="white")
    ax.set_title("Panel A — Quarterly Net Flows: ETF vs Mutual Fund",
                 color="white", fontsize=11, pad=10)
    ax.legend(framealpha=0, labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0fB'))

    # Panel B — Cumulative flows + exponential fit
    ax = axes[1]
    ax.plot(x, df["etf_cumulative"], color=colors["etf"],
            linewidth=2, label="ETF Cumulative Flows")
    ax.plot(x, etf_fitted, color="white", linewidth=1.4,
            linestyle="--", alpha=0.7,
            label=f"Exponential Fit  r={r_fit:.3f}/qtr")
    ax.plot(x, df["mf_cumulative"], color=colors["mf"],
            linewidth=2, label="MF Cumulative Flows")
    ax.fill_between(x, df["etf_cumulative"], df["mf_cumulative"],
                    alpha=0.08, color=colors["etf"])
    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=45, ha="right",
                       fontsize=7, color="white")
    ax.set_ylabel("Cumulative Flows ($B)", color="white")
    ax.set_title("Panel B — Cumulative Flow Divergence + Exponential Growth Fit",
                 color="white", fontsize=11, pad=10)
    ax.legend(framealpha=0, labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0fB'))

    # Panel C — Rolling 4-quarter correlation
    ax = axes[2]
    ax.plot(x, df["rolling_corr"], color=colors["corr"],
            linewidth=2, label="4-Quarter Rolling Correlation")
    ax.axhline(0,  color="white", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.axhline(-1, color="#ff6b6b", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.fill_between(x, df["rolling_corr"], 0,
                    where=df["rolling_corr"] < 0,
                    color=colors["mf"], alpha=0.15,
                    label="Negative correlation zone")
    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=45, ha="right",
                       fontsize=7, color="white")
    ax.set_ylabel("Correlation Coefficient", color="white")
    ax.set_ylim(-1.2, 1.2)
    ax.set_title("Panel C — Rolling Correlation: ETF vs MF Flows",
                 color="white", fontsize=11, pad=10)
    ax.legend(framealpha=0, labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")

    # Global title
    fig.suptitle(
        "Module 1 — The Mutual Fund → ETF Migration (2020–2025)\n"
        "Inspired by Jane Street Global ETF Landscape Report, Dec 2025",
        color="white", fontsize=13, y=1.01, fontweight="bold"
    )

    plt.tight_layout()
    out = FIGURES / "module1_mf_etf_migration.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Figure saved → {out}")


if __name__ == "__main__":
    run()