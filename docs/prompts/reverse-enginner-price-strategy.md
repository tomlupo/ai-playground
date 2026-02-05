 claude -p "
 AUTONOMOUS MODE - Do not ask questions, make reasonable assumptions.

 Reverse engineer MRETF, MRMOM, and ETFSEAS strategies from Price Action Lab.
 Use cached data from data/samples/.

 For each strategy:
 1. Hypothesize the signal logic based on strategy type and metrics
 2. Implement in tools/reverse-engineer-strategies/{strategy_name}.py
 3. Backtest 2003-2024
 4. Generate performance tearsheet
 5. Save charts to output/

 Target metrics to match (within 30%):
 - MRETF: Sharpe 0.78, MDD -9.3%, CAGR 5.2%
 - MRMOM: Sharpe 1.15, MDD -16.8%, CAGR 10.3%
 - ETFSEAS: Sharpe 0.83, MDD -13.4%, CAGR 7.3%

 When uncertain about parameters, start with common defaults (RSI-14, SMA-20/50/200, etc.) and iterate.
 " --dangerously-skip-permissions --output-format stream-json