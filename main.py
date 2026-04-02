"""
ETF Industry Dynamics: Growth, Migration, and Structural Shifts (2020-2025)
Inspired by Jane Street Global ETF Landscape Report, December 2025

Run this file to execute all three modules end to end.
Usage: python main.py
"""

import time
from src.module1_mf_etf_migration import run as run_module1
from src.module2_active_passive import run as run_module2
from src.module3_crypto_events import run as run_module3


def main():
    print("=" * 60)
    print("  ETF Industry Dynamics — Full Analysis Pipeline")
    print("  Inspired by Jane Street Global ETF Landscape 2026")
    print("=" * 60)

    print("\n[1/3] Running Module 1 — MF to ETF Migration...")
    t0 = time.time()
    run_module1()
    print(f"      Completed in {time.time() - t0:.1f}s")

    print("\n[2/3] Running Module 2 — Active vs Passive ETF Shift...")
    t0 = time.time()
    run_module2()
    print(f"      Completed in {time.time() - t0:.1f}s")

    print("\n[3/3] Running Module 3 — Crypto ETP Regulatory Events...")
    t0 = time.time()
    run_module3()
    print(f"      Completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 60)
    print("  All modules complete.")
    print("  Figures saved to:  outputs/figures/")
    print("  Tables saved to:   outputs/tables/")
    print("=" * 60)


if __name__ == "__main__":
    main()