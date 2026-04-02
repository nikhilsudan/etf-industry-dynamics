import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
from src.data_loader import load_crypto_etp

FIGURES = Path(__file__).resolve().parent.parent / "outputs" / "figures"
TABLES  = Path(__file__).resolve().parent.parent / "outputs" / "tables"

# ── Regulatory event dates ───────────────────────────────────────
EVENTS = {
    "Spot BTC\nApproved":      "2024-01-10",
    "Spot ETH\nApproved":      "2024-05-23",
    "In-Kind\nRedemptions":    "2025-07-22",
    "Generic Listing\nStandards": "2025-09-17",
}


def run():
    df = load_crypto_etp()

    # ── Aggregate across all tickers by date ────────────────────
    daily = (
        df.groupby("Date")
          .agg(total_volume=("Volume", "sum"),
               avg_close=("Close", "mean"))
          .reset_index()
          .sort_values("Date")
    )
    daily["log_volume"]     = np.log1p(daily["total_volume"])
    daily["rolling_vol_7"]  = daily["avg_close"].pct_change().rolling(7).std()
    daily["rolling_vol_30"] = daily["avg_close"].pct_change().rolling(30).std()
    daily["return_1d"]      = daily["avg_close"].pct_change()
    daily["cum_return"]     = (1 + daily["return_1d"]).cumprod()

    # ── Event study ──────────────────────────────────────────────
    window = 30  # days pre and post each event
    event_results = []

    for label, date_str in EVENTS.items():
        event_date = pd.Timestamp(date_str)
        if event_date not in daily["Date"].values:
            # Find nearest trading date
            idx = (daily["Date"] - event_date).abs().idxmin()
            event_date = daily["Date"].iloc[idx]

        idx = daily[daily["Date"] == event_date].index[0]
        pre_start  = max(0, idx - window)
        post_end   = min(len(daily) - 1, idx + window)

        pre_data  = daily.iloc[pre_start:idx]["log_volume"]
        post_data = daily.iloc[idx:post_end]["log_volume"]

        if len(pre_data) < 5 or len(post_data) < 5:
            continue

        pre_mean  = pre_data.mean()
        post_mean = post_data.mean()
        abnormal  = post_mean - pre_mean

        t_stat, p_value = stats.ttest_ind(pre_data, post_data)

        event_results.append({
            "event":        label.replace("\n", " "),
            "event_date":   date_str,
            "pre_mean_vol": pre_mean,
            "post_mean_vol": post_mean,
            "abnormal_vol": abnormal,
            "t_statistic":  t_stat,
            "p_value":      p_value,
            "significant":  p_value < 0.05
        })

        print(f"\nEvent: {label.replace(chr(10), ' ')}")
        print(f"  Pre-event avg log volume:  {pre_mean:.4f}")
        print(f"  Post-event avg log volume: {post_mean:.4f}")
        print(f"  Abnormal volume:           {abnormal:+.4f}")
        print(f"  t-statistic:               {t_stat:.4f}")
        print(f"  p-value:                   {p_value:.4f} "
              f"{'*** SIGNIFICANT' if p_value < 0.05 else '(not significant)'}")

    results_df = pd.DataFrame(event_results)
    results_df.to_csv(TABLES / "module3_event_study.csv", index=False)

    # ── Random Forest: High vs Low inflow regime ─────────────────
    rf_df = daily.dropna(subset=["rolling_vol_7", "rolling_vol_30",
                                  "return_1d", "log_volume"]).copy()

    # Label: 1 = high volume day (above median), 0 = low volume day
    median_vol = rf_df["log_volume"].median()
    rf_df["label"] = (rf_df["log_volume"] > median_vol).astype(int)

    features = ["rolling_vol_7", "rolling_vol_30", "return_1d"]
    X = rf_df[features].values
    y = rf_df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, shuffle=False
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                  max_depth=4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    importances = clf.feature_importances_
    feature_names = ["7d Volatility", "30d Volatility", "1d Return"]

    print(f"\n── Random Forest: Volume Regime Classifier ──")
    print(classification_report(y_test, y_pred,
                                 target_names=["Low Volume", "High Volume"]))
    print("Feature importances:")
    for name, imp in zip(feature_names, importances):
        print(f"  {name}: {imp:.4f}")

    # ════════════════════════════════════════════════════════════
    # FIGURE 3
    # ════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(4, 1, figsize=(14, 20))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")

    colors = {
        "price":    "#00d4aa",
        "volume":   "#60a5fa",
        "vol7":     "#f0c040",
        "vol30":    "#ff6b6b",
        "event":    "#c084fc",
        "rf_high":  "#00d4aa",
        "rf_low":   "#ff6b6b",
    }

    event_dates = [pd.Timestamp(d) for d in EVENTS.values()]
    event_labels = list(EVENTS.keys())

    def draw_events(ax):
        for date, label in zip(event_dates, event_labels):
            if daily["Date"].min() <= date <= daily["Date"].max():
                ax.axvline(date, color=colors["event"],
                           linewidth=1.0, linestyle="--", alpha=0.7)
                ax.text(date, ax.get_ylim()[1] * 0.92,
                        label, color=colors["event"],
                        fontsize=6.5, ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="#1a1a2e",
                                  edgecolor=colors["event"],
                                  alpha=0.7))

    # Panel A — Cumulative price return
    ax = axes[0]
    ax.plot(daily["Date"], daily["cum_return"],
            color=colors["price"], linewidth=1.8)
    ax.fill_between(daily["Date"], daily["cum_return"], 1,
                    where=daily["cum_return"] >= 1,
                    alpha=0.12, color=colors["price"])
    ax.axhline(1, color="white", linewidth=0.6,
               linestyle="--", alpha=0.4)
    ax.set_ylabel("Cumulative Return (base=1)", color="white")
    ax.set_title("Panel A — Crypto ETP Cumulative Price Return",
                 color="white", fontsize=11, pad=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")
    draw_events(ax)

    # Panel B — Log volume with event lines
    ax = axes[1]
    ax.bar(daily["Date"], daily["log_volume"],
           color=colors["volume"], alpha=0.6, width=1)
    ax.set_ylabel("Log Volume", color="white")
    ax.set_title("Panel B — Aggregate Daily Log Volume + Regulatory Events",
                 color="white", fontsize=11, pad=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")
    draw_events(ax)

    # Panel C — Rolling volatility
    ax = axes[2]
    ax.plot(daily["Date"], daily["rolling_vol_7"] * 100,
            color=colors["vol7"], linewidth=1.5,
            label="7-day Rolling Volatility")
    ax.plot(daily["Date"], daily["rolling_vol_30"] * 100,
            color=colors["vol30"], linewidth=1.5,
            label="30-day Rolling Volatility")
    ax.set_ylabel("Volatility (%)", color="white")
    ax.set_title("Panel C — Rolling Volatility (7d and 30d)",
                 color="white", fontsize=11, pad=10)
    ax.legend(framealpha=0, labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")
    draw_events(ax)

    # Panel D — Random Forest feature importances
    ax = axes[3]
    bars = ax.barh(feature_names, importances,
                   color=[colors["vol7"], colors["vol30"], colors["price"]],
                   alpha=0.85, height=0.5)
    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{imp:.3f}", va="center", color="white", fontsize=10)
    ax.set_xlabel("Feature Importance", color="white")
    ax.set_title(
        "Panel D — Random Forest: Feature Importance for Volume Regime Classification\n"
        "(Which features best predict high vs low volume days?)",
        color="white", fontsize=11, pad=10
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")
    ax.set_xlim(0, max(importances) + 0.08)

    fig.suptitle(
        "Module 3 — Crypto ETP: Regulatory Events as Growth Catalysts (2024–2025)\n"
        "Inspired by Jane Street Global ETF Landscape Report, Dec 2025",
        color="white", fontsize=13, y=1.01, fontweight="bold"
    )

    plt.tight_layout()
    out = FIGURES / "module3_crypto_events.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nFigure saved → {out}")


if __name__ == "__main__":
    run()