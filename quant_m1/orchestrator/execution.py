"""Execution wiring: target weights to authorized orders.

Converts allocator output into integer-share Intent objects, applies a per-asset
dust-trade filter, and dispatches them through the Kernel and Gateway.

The Kernel remains the final authorization gate. This module never bypasses it.
A rejected order does not halt dispatch of the remaining orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..allocate.engine import AllocationResult
from ..authority import Intent, Kernel, Gateway, Reject
from ..core import Ledger, Clock, Instrument
from ..flags import Flags


@dataclass
class ExecutionReport:
    intents_generated: int
    intents_submitted: int
    intents_rejected: int
    rejection_reasons: list[str]
    skipped_dust_trades: int

    def as_dict(self) -> dict:
        return {
            "intents_generated": self.intents_generated,
            "intents_submitted": self.intents_submitted,
            "intents_rejected": self.intents_rejected,
            "rejection_reasons": list(self.rejection_reasons),
            "skipped_dust_trades": self.skipped_dust_trades,
        }


def generate_intents(allocation: AllocationResult,
                     ledger: Ledger,
                     prices: Mapping[str, int],
                     instruments: Mapping[str, Instrument],
                     min_trade_bps: int = 10,
                     cost_bps: int = 10) -> tuple[list[Intent], int]:
    """Convert target weights into BUY/SELL Intents plus a dust-skip count.

    Applies a per-asset dust filter: if abs(delta notional) is below
    min_trade_bps of current NAV, the trade is skipped. Addresses the
    binary-band criticism by making the threshold per-asset in notional terms.

    Universe ordering is sorted(prices.keys()). The caller MUST ensure the
    allocation weight vector uses that same ordering.

    Returns:
        (intents, skipped_dust_trades)
    """
    if not allocation.solver_ok:
        return [], 0

    cash_equiv = ledger.state.cash_settled + ledger.state.receivable_total()
    pos_val = sum(q * ledger.state.marks[s]
                  for s, q in ledger.state.positions.items()
                  if s in ledger.state.marks)
    nav = cash_equiv + pos_val
    if nav <= 0:
        return [], 0

    min_trade_notional = (nav * min_trade_bps) // 10_000
    universe = sorted(prices.keys())
    if len(universe) != len(allocation.weights):
        raise ValueError("universe size mismatch with allocation weights")

    intents: list[Intent] = []
    skipped = 0

    for i, symbol in enumerate(universe):
        if symbol not in instruments or symbol not in ledger.state.marks:
            continue

        target_weight = float(allocation.weights[i])
        target_notional = int(target_weight * nav)
        price = prices[symbol]
        if price <= 0:
            continue

        target_shares = target_notional // price
        current_shares = ledger.state.positions.get(symbol, 0)
        delta_shares = target_shares - current_shares
        if delta_shares == 0:
            continue

        delta_notional = abs(delta_shares * price)
        if delta_notional < min_trade_notional:
            skipped += 1
            continue

        side = "BUY" if delta_shares > 0 else "SELL"
        intents.append(Intent(symbol, side, abs(delta_shares), price, cost_bps))

    return intents, skipped


def dispatch_intents(intents: list[Intent],
                     kernel: Kernel,
                     gateway: Gateway,
                     ledger: Ledger,
                     flags: Flags,
                     clock: Clock,
                     watchdog_stop: bool = False,
                     skipped_dust_trades: int = 0) -> ExecutionReport:
    """Authorize and submit intents. Kernel rejections are captured, not raised."""
    submitted = 0
    rejected = 0
    reasons: list[str] = []

    for intent in intents:
        try:
            tok = kernel.authorize(ledger, intent, flags, clock, watchdog_stop)
            gateway.submit(tok, intent)
            submitted += 1
        except Reject as e:
            rejected += 1
            reasons.append(f"{intent.symbol}:{e.code}")
        except Exception as e:
            rejected += 1
            reasons.append(f"{intent.symbol}:UNEXPECTED_{type(e).__name__}")

    return ExecutionReport(
        intents_generated=len(intents),
        intents_submitted=submitted,
        intents_rejected=rejected,
        rejection_reasons=reasons,
        skipped_dust_trades=skipped_dust_trades,
    )