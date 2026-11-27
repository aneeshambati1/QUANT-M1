"""ETF Universe Factory.

Creates valid Instrument objects with realistic metadata and updates the
mandate allowlist so Kernel.authorize() will accept the symbols.
"""
from __future__ import annotations
from typing import List, Mapping

def register_etf_universe(sx: dict, symbols: List[str]) -> None:
    """
    Registers ETF instruments with proper metadata and updates mandate allowlist.
    
    Args:
        sx: System state dict from system.build()
        symbols: List of ETF ticker symbols (e.g., ["SPY", "QQQ", "IEF"])
    """
    # 1. Import Instrument class (adjust path if your structure differs)
    try:
        from .meta.instrument import Instrument
    except ImportError:
        from quant_m1.meta.instrument import Instrument

    if "instruments" not in sx:
        sx["instruments"] = {}

    # 2. Create valid Instrument objects with realistic defaults
    for sym in symbols:
        if sym not in sx["instruments"]:
            sx["instruments"][sym] = Instrument(
                symbol=sym,
                product_class="EQUITY",      # Or "ETF" if your enum supports it
                adv=1_000_000_000,           # $1B ADV (liquid ETF default)
                beta_bps=10000,              # 1.0 beta
                holdings=0,                  # No initial holdings
                halted=False,                # Active trading
            )

    # 3. Update mandate allowlist so Kernel.authorize() accepts these symbols
    # The mandate is typically nested inside the kernel or config
    kernel = sx.get("kernel")
    if kernel and hasattr(kernel, "mandate"):
        mandate = kernel.mandate
        if hasattr(mandate, "allowlist"):
            # allowlist is typically a set
            if isinstance(mandate.allowlist, set):
                mandate.allowlist.update(set(symbols))
            elif isinstance(mandate.allowlist, (list, tuple)):
                mandate.allowlist = list(set(mandate.allowlist) | set(symbols))
