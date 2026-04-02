import pandas as pd
import numpy as np
from pathlib import Path

# Base paths
DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_etf_flows():
    path = DATA_RAW / "ici_etf_flows_2020_2025.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_mf_flows():
    path = DATA_RAW / "ici_mf_flows_2020_2025.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_etf_launches():
    path = DATA_RAW / "etf_launches_2020_2025.csv"
    df = pd.read_csv(path)
    df["period"] = df["quarter"] + " " + df["year"].astype(str)
    df["active_share"] = df["active_launches"] / df["total_launches"]
    return df


def load_retirement_assets():
    path = DATA_RAW / "us_retirement_assets_2020_2025.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_crypto_etp():
    path = DATA_RAW / "crypto_etp_prices_volumes.csv"
    df = pd.read_csv(path)

    # Flatten any multiindex columns from yfinance
    df.columns = [str(c).strip() for c in df.columns]

    # Rename Price/Close variants
    if "Price" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Price": "Close"})

    # Parse date column regardless of name
    date_col = [c for c in df.columns if c.lower() in ["date", "datetime"]][0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Force numeric columns
    for col in ["Close", "Volume", "Open", "High", "Low"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Close"])
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    return df


def load_all():
    return {
        "etf_flows": load_etf_flows(),
        "mf_flows": load_mf_flows(),
        "etf_launches": load_etf_launches(),
        "retirement": load_retirement_assets(),
        "crypto": load_crypto_etp()
    }