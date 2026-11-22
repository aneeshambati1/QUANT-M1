# scripts/run_backtest.py
from quant_m1.system import build
from quant_m1.alpha.controller import MomentumStrategy, MarketSnapshot
import tempfile

def run_backtest():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Build your existing OMS
        sx = build(tmpdir, cash=1_000_000, positions={}, marks={"SPY": 400})
        
        # 2. Initialize the Brain
        strategy = MomentumStrategy(symbol="SPY", lookback_days=5, target_weight=0.8, cost_bps=5)
        
        # 3. Mock Data Feed (Replace with real PIT Data later)
        mock_data_feed = [
            MarketSnapshot("2024-01-01", {"SPY": 400.0}, {"SPY": 1e6}),
            MarketSnapshot("2024-01-02", {"SPY": 402.0}, {"SPY": 1e6}),
            # ... add more days
        ]
        
        # 4. The Daily Loop
        for day in mock_data_feed:
            # Update marks in the ledger based on today's prices
            # (You will need to add a mark_to_market method to your Ledger/State)
            
            # Generate Intents
            intents = strategy.generate_intents(
                snapshot=day,
                current_positions=sx["ledger"].state.positions,
                cash_settled=sx["ledger"].state.cash_settled
            )
            
            # Feed Intents to your existing Kernel/Gateway
            for intent in intents:
                try:
                    tok = sx["kernel"].authorize(
                        sx["ledger"], intent, sx["flags"], sx["clock"], False
                    )
                    sx["gw"].submit(tok, intent)
                    # Process mock venue fills...
                except Exception as e:
                    print(f"Kernel rejected intent: {e}")

if __name__ == "__main__":
    run_backtest()
