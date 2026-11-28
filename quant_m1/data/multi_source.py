"""Multi-source free-data ingest with a frozen, append-only archive.

Honest scope (see evidence/free_data_protocol.md):
  * Cross-provider DRIFT detection: if a provider restates a price you already
    froze, the archive records a DRIFT event and NEVER overwrites the old bar.
  * Self-stamped availability: every bar carries your own received_at (UTC).
    This is a weaker proxy for vendor known_at, and is labelled as such.
  * Delisted retention: symbols you have downloaded are never deleted.

This does NOT produce true PIT data, vendor known_at, or survivorship-free
universe membership. It is an EXPLORATORY-grade panel. The confirmatory
PIT_DATA_ACCEPTANCE_SPEC remains unmet and unchanged.

Real provider fetchers are thin, version-sensitive adapters (network + keys
required) and are NOT exercised by the unit tests. The tested core is the
archive / drift logic, which accepts injected fetchers.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ..core import Store, sha

# A fetcher returns [(date_str YYYY-MM-DD, close_float), ...] for one symbol.
Fetcher = Callable[[str], List[Tuple[str, float]]]


def _bar_hash(symbol: str, date: str, provider: str, close_cents: int) -> str:
    return sha({"symbol": symbol, "date": date, "provider": provider,
                "close_cents": close_cents})


class MultiSourceArchive:
    """Append-only, hash-chained archive of (symbol, date, provider) bars."""

    def __init__(self, path: str):
        self.store = Store(path)
        self.chain = self.store.load() or []
        self._verify()
        self.bars: Dict[Tuple[str, str, str], dict] = {}
        self.drift: List[dict] = []
        for rec in self.chain:
            p = rec["payload"]
            if rec["kind"] == "BAR":
                self.bars[(p["symbol"], p["date"], p["provider"])] = p
            elif rec["kind"] == "DRIFT":
                self.drift.append(p)

    def _verify(self) -> None:
        prev = "0" * 64
        for rec in self.chain:
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec.get("prev") != prev or rec.get("hash") != sha(body):
                raise ValueError("MULTI_SOURCE_CHAIN_INVALID")
            prev = rec["hash"]

    def _append(self, kind: str, payload: dict) -> None:
        prev = self.chain[-1]["hash"] if self.chain else "0" * 64
        rec = {"kind": kind, "payload": payload, "prev": prev}
        rec["hash"] = sha(rec)
        self.chain.append(rec)
        self.store.save(self.chain)

    def ingest(self, symbol: str, date: str, provider: str,
               close: float, received_at: str) -> str:
        """Return 'NEW', 'UNCHANGED', or 'DRIFT'. Never overwrites a frozen bar."""
        if not isinstance(close, (int, float)) or close != close or close <= 0:
            raise ValueError("close must be a positive finite number")
        close_cents = int(round(close * 100))
        key = (symbol, date, provider)
        h = _bar_hash(symbol, date, provider, close_cents)
        existing = self.bars.get(key)
        if existing is not None:
            if existing["hash"] == h:
                return "UNCHANGED"
            self._append("DRIFT", {
                "symbol": symbol, "date": date, "provider": provider,
                "old_cents": existing["close_cents"], "new_cents": close_cents,
                "old_received_at": existing["received_at"],
                "new_received_at": received_at,
            })
            self.drift.append(self.chain[-1]["payload"])
            return "DRIFT"
        payload = {"symbol": symbol, "date": date, "provider": provider,
                   "close_cents": close_cents, "received_at": received_at,
                   "hash": h}
        self._append("BAR", payload)
        self.bars[key] = payload
        return "NEW"

    def ingest_fetcher(self, symbol: str, provider: str,
                       fetcher: Fetcher, received_at: str) -> Dict[str, int]:
        counts = {"NEW": 0, "UNCHANGED": 0, "DRIFT": 0}
        for date, close in fetcher(symbol):
            counts[self.ingest(symbol, date, provider, close, received_at)] += 1
        return counts

    def symbols(self) -> List[str]:
        return sorted({k[0] for k in self.bars})

    def has_symbol(self, symbol: str) -> bool:
        """Delisted retention: once downloaded, a symbol is never removed."""
        return any(k[0] == symbol for k in self.bars)

    def close_cents(self, symbol: str, date: str, provider: str) -> Optional[int]:
        bar = self.bars.get((symbol, date, provider))
        return None if bar is None else bar["close_cents"]


# ---- Real provider adapters (untested here; require network + keys) --------

def yfinance_fetcher(symbol: str, start: str, end: str) -> Fetcher:
    def _fetch(sym: str) -> List[Tuple[str, float]]:
        import yfinance as yf
        df = yf.download(sym, start=start, end=end,
                         auto_adjust=False, progress=False)
        out = []
        for idx, row in df.iterrows():
            c = float(row["Close"])
            if c == c and c > 0:
                out.append((idx.strftime("%Y-%m-%d"), c))
        return out
    return _fetch


def alpaca_fetcher(symbol: str, start: str, end: str) -> Fetcher:
    """Version-sensitive adapter; verify against your alpaca-py version."""
    def _fetch(sym: str) -> List[Tuple[str, float]]:
        import os
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        client = StockHistoricalDataClient(
            os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
        req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                               start=start, end=end, adjustment="raw")
        df = client.get_stock_bars(req).df
        out = []
        for idx, row in df.iterrows():
            c = float(row["close"])
            if c == c and c > 0:
                out.append((idx[0].strftime("%Y-%m-%d")
                            if isinstance(idx, tuple) else idx.strftime("%Y-%m-%d"), c))
        return out
    return _fetch
