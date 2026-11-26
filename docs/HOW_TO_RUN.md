# How to Run QUANT-M1

## 1. Install
```bash
pip install -r quant_m1/requirements.txt
```

## 2. Run the test suite
```bash
python -m pytest tests -q
```

## 3. Exploratory simulation on free data (yfinance)

This runs the paper loop on yfinance data and lets the contribution trackers
adapt. **Exploratory only** — yfinance is not point-in-time, so results cannot
support edge claims.

```python
import yfinance as yf, numpy as np, tempfile
from quant_m1.system import build
from quant_m1.orchestrator.daemon import PaperDaemon
from quant_m1.meta import ContributionTracker, MarketState
from quant_m1.allocate import solve_allocation

# Download daily closes
data = yf.download(["SPY","QQQ","TLT"], start="2015-01-01", auto_adjust=False)["Close"]
data = data.dropna()

# Build a daemon with a mock venue
d = tempfile.mkdtemp()
da = PaperDaemon(d, cash=10_000_000)

# A simple adaptive tracker
tracker = ContributionTracker(half_life_obs=60.0, probation_obs=30)

# Walk forward day by day (simplified — real use builds MarketState per session)
print("Sessions to run:", len(data))
```

## 4. Run on Alpaca Paper (optional)

Set Colab Secrets `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (paper keys only).

```python
from quant_m1.venue_adapters.alpaca_paper import AlpacaPaperVenue
venue = AlpacaPaperVenue()          # hardcoded paper=True
events = venue.poll_fills()         # discover fills each session
```

## 5. What the results mean

- A positive paired differential on yfinance ≠ an edge (look-ahead bias).
- Treat every yfinance result as a system-functionality check, not alpha.
- For confirmatory results you need point-in-time data (see Round 9A spec).
