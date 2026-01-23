#!/usr/bin/env python3
"""
Basic usage examples for the trading framework.

Demonstrates:
1. Setting up different exchange connections
2. Paper trading with the simulator
3. Calibrating from real market data
4. Implementing simple strategies
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta

# Core imports
from trading import (
    TradingEngine,
    SimulatedExchange,
    BinanceExchange,
    Order,
    OrderSide,
    OrderType,
    AssetClass,
)
from trading.simulator import (
    SimulatorConfig,
    MarketModelConfig,
    MarketCalibrator,
    GBMModel,
)
from trading.strategy import Strategy, StrategyConfig, Signal
from trading.strategy.risk import RiskManager, RiskConfig, PercentEquitySizer
from trading.config import TradingConfig, ExchangeSetup, ExchangeType


# =============================================================================
# Example 1: Simple Paper Trading with Simulator
# =============================================================================

async def example_paper_trading():
    """
    Basic paper trading setup with the market simulator.

    This is the simplest way to get started - no API keys needed!
    """
    print("\n=== Example 1: Paper Trading ===\n")

    # Create trading engine
    engine = TradingEngine()

    # Add simulator with initial balance
    engine.add_simulated_exchange(
        initial_balance={"USD": 100000.0, "BTC": 0.0},
        volatility=0.60,  # 60% annual volatility (crypto-like)
        spread_pct=0.0005,  # 5 basis points spread
    )

    # Add trading instrument
    engine.add_instrument(
        symbol="BTCUSDT",
        initial_price=50000.0,
        asset_class=AssetClass.CRYPTO,
    )

    # Start engine
    await engine.start()

    # Get current price
    ticker = await engine.get_ticker("BTCUSDT")
    print(f"Current BTC price: ${ticker.mid_price:,.2f}")
    print(f"Spread: ${ticker.spread:,.2f} ({ticker.spread_bps:.1f} bps)")

    # Place a buy order
    order = await engine.submit_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=0.1,  # Buy 0.1 BTC
        order_type=OrderType.MARKET,
    )
    print(f"\nBuy order filled at ${order.average_fill_price:,.2f}")

    # Check position
    positions = await engine.get_positions()
    print(f"Positions: {[(p.symbol, float(p.quantity)) for p in positions]}")

    # Check equity
    equity = await engine.get_equity()
    print(f"Account equity: ${equity:,.2f}")

    # Close position
    await engine.close_position("BTCUSDT")
    print("Position closed")

    await engine.stop()


# =============================================================================
# Example 2: Multiple Asset Classes
# =============================================================================

async def example_multi_asset():
    """
    Trading multiple asset classes with different configurations.
    """
    print("\n=== Example 2: Multi-Asset Trading ===\n")

    engine = TradingEngine()

    # Add simulator
    sim = engine.add_simulated_exchange(
        initial_balance={"USD": 100000.0},
    )

    # Add crypto (high volatility)
    sim.add_instrument(
        symbol="BTCUSDT",
        initial_price=50000.0,
        model_config=MarketModelConfig(
            volatility=0.60,
            spread_pct=0.0005,
        ),
        asset_class=AssetClass.CRYPTO,
    )

    # Add equity (medium volatility)
    sim.add_instrument(
        symbol="AAPL",
        initial_price=175.0,
        model_config=MarketModelConfig(
            volatility=0.25,
            spread_pct=0.0001,
            drift=0.10,  # 10% expected annual return
        ),
        asset_class=AssetClass.EQUITY,
    )

    # Add forex (low volatility, mean-reverting)
    sim.add_instrument(
        symbol="EUR_USD",
        initial_price=1.0850,
        model_config=MarketModelConfig(
            volatility=0.08,
            spread_pct=0.00005,
            mean_reversion_speed=10.0,  # Strong mean reversion
            mean_reversion_level=1.0850,
        ),
        model_type="ou",  # Ornstein-Uhlenbeck model
        asset_class=AssetClass.FOREX,
    )

    await engine.start()

    # Get tickers for all instruments
    for symbol in ["BTCUSDT", "AAPL", "EUR_USD"]:
        ticker = await engine.get_ticker(symbol)
        print(f"{symbol}: {ticker.mid_price:.4f} (spread: {ticker.spread_bps:.1f} bps)")

    await engine.stop()


# =============================================================================
# Example 3: Calibrating from Real Data
# =============================================================================

async def example_calibration():
    """
    Calibrate market model from real exchange data.

    This makes your paper trading more realistic!
    """
    print("\n=== Example 3: Model Calibration ===\n")

    # Create calibrator
    calibrator = MarketCalibrator("BTCUSDT")

    # Simulate adding historical data
    # (In production, you'd fetch real data from an exchange)
    from trading.core.models import OHLCV
    import random

    price = 50000.0
    for i in range(100):
        change = random.gauss(0, price * 0.02)
        price = max(price + change, 1000)

        bar = OHLCV(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow() - timedelta(hours=100-i),
            open=Decimal(str(price)),
            high=Decimal(str(price * 1.01)),
            low=Decimal(str(price * 0.99)),
            close=Decimal(str(price + change/2)),
            volume=Decimal("1000"),
        )
        calibrator.add_bar(bar)

    # Perform calibration
    result = calibrator.calibrate()

    print(f"Calibration Results:")
    print(f"  Volatility: {result.volatility*100:.1f}%")
    print(f"  Drift: {result.drift*100:.1f}%")
    print(f"  Spread: {result.spread_pct*100:.3f}%")
    print(f"  Has Jumps: {result.has_jumps}")
    print(f"  Mean Reverting: {result.mean_reverting}")
    print(f"  Recommended Model: {result.recommended_model}")
    print(f"  R-squared: {result.r_squared:.3f}")

    # Use calibration in simulator
    engine = TradingEngine()
    engine.add_simulated_exchange()
    engine.add_instrument(
        symbol="BTCUSDT",
        initial_price=price,
        calibration=result,
    )

    await engine.start()
    ticker = await engine.get_ticker("BTCUSDT")
    print(f"\nCalibrated ticker: {ticker.mid_price:.2f}")
    await engine.stop()


# =============================================================================
# Example 4: Simple Moving Average Strategy
# =============================================================================

class SMAStrategy(Strategy):
    """
    Simple Moving Average Crossover Strategy.

    Goes long when fast MA crosses above slow MA.
    Goes short when fast MA crosses below slow MA.
    """

    def __init__(self, exchange, config=None, fast_period=10, slow_period=30):
        super().__init__(exchange, config or StrategyConfig(
            name="SMA_Crossover",
            symbols=["BTCUSDT"],
        ))
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prices: list[float] = []

    def on_tick(self, tick):
        # Update price history
        self.prices.append(float(tick.mid_price))
        self.prices = self.prices[-self.slow_period:]

        # Generate signal if we have enough data
        if len(self.prices) >= self.slow_period:
            signal = self.generate_signal(tick.symbol)
            if signal != Signal.HOLD:
                asyncio.create_task(self.execute_signal(
                    tick.symbol,
                    signal,
                    quantity=Decimal("0.01"),
                ))

    def on_bar(self, bar):
        # Same logic can be applied to bars
        pass

    def generate_signal(self, symbol: str) -> Signal:
        if len(self.prices) < self.slow_period:
            return Signal.HOLD

        fast_ma = sum(self.prices[-self.fast_period:]) / self.fast_period
        slow_ma = sum(self.prices) / len(self.prices)

        # Check previous MAs for crossover detection
        if len(self.prices) > self.slow_period:
            prev_prices = self.prices[:-1]
            prev_fast = sum(prev_prices[-self.fast_period:]) / self.fast_period
            prev_slow = sum(prev_prices) / len(prev_prices)

            # Bullish crossover
            if prev_fast <= prev_slow and fast_ma > slow_ma:
                return Signal.LONG

            # Bearish crossover
            if prev_fast >= prev_slow and fast_ma < slow_ma:
                return Signal.SHORT

        return Signal.HOLD


async def example_strategy():
    """
    Running a strategy with the trading engine.
    """
    print("\n=== Example 4: SMA Strategy ===\n")

    engine = TradingEngine()
    engine.add_simulated_exchange(
        initial_balance={"USD": 10000.0},
    )
    engine.add_instrument("BTCUSDT", 50000.0)

    await engine.start()

    # Create and add strategy
    exchange = engine.get_exchange()
    strategy = SMAStrategy(exchange, fast_period=5, slow_period=15)
    engine.add_strategy(strategy)

    # Run for a short time
    await strategy.start()

    # Simulate some price updates
    for _ in range(20):
        await asyncio.sleep(0.2)

    # Check results
    metrics = strategy.get_metrics()
    print(f"Strategy Metrics:")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Win Rate: {metrics.win_rate*100:.1f}%")

    await strategy.stop()
    await engine.stop()


# =============================================================================
# Example 5: Risk Management
# =============================================================================

async def example_risk_management():
    """
    Using the risk management system.
    """
    print("\n=== Example 5: Risk Management ===\n")

    # Configure risk limits
    risk_config = RiskConfig(
        max_position_pct=0.10,        # Max 10% per position
        max_total_exposure=0.50,      # Max 50% total exposure
        max_loss_per_trade_pct=0.02,  # Max 2% loss per trade
        daily_loss_limit_pct=0.05,    # Max 5% daily loss
        max_drawdown_pct=0.15,        # Max 15% drawdown
    )

    risk_manager = RiskManager(risk_config, Decimal("100000"))

    # Position sizer
    sizer = PercentEquitySizer(percent=0.02)  # 2% of equity per trade

    # Calculate position size
    equity = Decimal("100000")
    price = Decimal("50000")
    size = sizer.calculate_size(equity=equity, price=price)
    print(f"Position size for ${equity} equity at ${price}: {size:.4f} units")

    # Check if we can open a position
    can_open, reason = risk_manager.can_open_position(
        symbol="BTCUSDT",
        quantity=size,
        price=price,
        side=OrderSide.BUY,
    )
    print(f"Can open position: {can_open} ({reason or 'OK'})")

    # Get current risk metrics
    metrics = risk_manager.get_metrics()
    print(f"\nRisk Metrics:")
    print(f"  Current Drawdown: {metrics.current_drawdown*100:.2f}%")
    print(f"  VaR (95%): ${metrics.var_95:,.2f}")


# =============================================================================
# Example 6: Configuration File
# =============================================================================

def example_config():
    """
    Working with configuration files.
    """
    print("\n=== Example 6: Configuration ===\n")

    # Create config programmatically
    config = TradingConfig(
        mode="paper",
        symbols=["BTCUSDT", "ETHUSDT"],
        exchanges=[
            ExchangeSetup(
                type=ExchangeType.BINANCE,
                testnet=True,
            ),
        ],
        risk=RiskConfig(
            max_position_pct=0.05,
            daily_loss_limit_pct=0.02,
        ),
    )

    # Validate config
    issues = config.validate()
    if issues:
        print(f"Config issues: {issues}")
    else:
        print("Config is valid!")

    # Save to file
    from trading.config import save_config
    save_config(config, "/tmp/trading_config.yaml")
    print("Config saved to /tmp/trading_config.yaml")

    # Load from file
    from trading.config import load_config
    loaded = load_config("/tmp/trading_config.yaml")
    print(f"Loaded config mode: {loaded.mode}")
    print(f"Loaded symbols: {loaded.symbols}")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all examples."""
    await example_paper_trading()
    await example_multi_asset()
    await example_calibration()
    await example_strategy()
    await example_risk_management()
    example_config()

    print("\n=== All examples completed! ===\n")


if __name__ == "__main__":
    asyncio.run(main())
