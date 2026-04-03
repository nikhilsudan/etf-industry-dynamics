# Methodology Notes & Design Decisions

This document explains the key analytical choices made in each module,
the reasoning behind them, and where the analysis intentionally stops short.

---

## Module 1 — MF to ETF Migration

### Why exponential growth model?
ETF AUM growth is widely described as compounding in nature. The exponential
model `AUM(t) = A * e^(rt)` is the natural mathematical starting point for
any compounding process. Fitting this to cumulative flows rather than raw
quarterly flows smooths out quarter-to-quarter noise and reveals the
underlying trajectory. The growth rate r = 0.093 per quarter is interpretable
directly: it implies a doubling time of 7.5 quarters, which is a concrete and
communicable finding rather than an abstract coefficient.

### Why cumulative flows rather than AUM?
True AUM data requires a Bloomberg terminal or paid Morningstar subscription.
Cumulative net flows from public ICI data serve as a reasonable structural
proxy for the direction and magnitude of capital migration. The limitation is
acknowledged in the README.

### Why 4-quarter rolling correlation?
A 4-quarter window captures one full annual cycle, which is the natural
periodicity of institutional rebalancing and fund reporting. Shorter windows
would introduce too much noise; longer windows would obscure the structural
shift that begins post-2023. The finding that correlation turns negative
after 2023 is the most important result in this module and directly supports
the rotation narrative in the Jane Street report.

---

## Module 2 — Active ETF Adoption S-Curve

### Why logistic growth model?
Technology and product adoption processes are classically modelled using
logistic growth because they have natural ceilings. An ETF market cannot
have more than 100% active launches, and institutional inertia means passive
indexing will retain a structural floor. The logistic function
`f(t) = L / (1 + e^(-k(t - t0)))` is the solution to the differential
equation `df/dt = k * f * (1 - f/L)`, which models growth that slows as it
approaches a ceiling. This is a more honest model than linear extrapolation.

### Why not fit to flow share instead of launch share?
Launch data from the Jane Street report is more directly observable and
less subject to market price fluctuations than flow share. Flow share
is heavily influenced by short-term market sentiment; launch decisions
reflect longer-term strategic commitments by asset managers, which is
what the adoption curve should capture.

### Honest limitation of the S-curve fit
The inflection point estimated as pre-2022 reflects a genuine data constraint:
our dataset begins at 30% active share, meaning we observe only the later
portion of the S-curve. The model cannot see the early slow-growth phase.
This is reported transparently rather than hidden. The finding that active
ETF adoption is already at 87% of its estimated ceiling is arguably more
interesting than a clean S-curve would be, suggesting the structural shift
is maturing rather than accelerating.

---

## Module 3 — Crypto ETP Regulatory Events

### Why log volume rather than raw volume?
Trading volume in financial markets follows a log-normal distribution.
Raw volume comparisons are heavily skewed by outlier days. Log volume
stabilises variance and makes pre/post event comparisons via t-tests
more statistically valid, since the t-test assumes approximately normal
distributions in each group.

### Why a 30-day event window?
30 calendar days is a standard window in academic event studies for
regulatory approvals. It is long enough to capture sustained behavioural
change rather than a one-day spike, but short enough to avoid confounding
with unrelated macro events. A 7-day window would capture only the
immediate reaction; a 90-day window would pick up too many unrelated
market movements.

### Why two-sample t-test rather than paired t-test?
Pre and post event observations are independent time periods with
different market conditions. A paired t-test would require matching
individual days across the two windows, which is not appropriate here.
The two-sample t-test correctly treats the pre and post windows as two
separate samples.

### Interpreting the ETH approval result
The negative abnormal volume after Spot ETH approval (-0.24 log units,
p=0.013) does not mean the approval had a negative effect. It most likely
reflects capital rotating from Bitcoin ETPs into Ethereum ETPs rather than
new money entering the market. This is a financially coherent interpretation
consistent with observed BTC ETP volume declining modestly around the same
period. A more rigorous test would isolate each ticker separately, which is
flagged as a future extension.

### Why Random Forest for regime classification?
Random Forest is an appropriate first-pa