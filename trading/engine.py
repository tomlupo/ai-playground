"""
Trading Engine - Main orchestrator for the trading framework.

Provides:
- Exchange connection management
- Strategy lifecycle management
- Event coordination
- Unified interface for all trading operations
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Callable, Any

from trading.core.models import Order, Position, Tick, OHLCV
from trading.core.enums import OrderSide, OrderType, AssetClass, DataResolution
from trading.core.events import EventEmitter, Event, EventType
from trading.exchanges.base import Exchange
from trading.exchanges.binance import BinanceExchange, BinanceConfig
from trading.exchanges.interactive_brokers import InteractiveBrokersExchange, IBConfig
from trading.exchanges.oanda import OandaExchange, OandaConfig
from trading.simulator import SimulatedExchange, SimulatorConfig, MarketModelConfig
from trading.simulator.calibration import MarketCalibrator, CalibrationResult
from trading.strategy.base import Strategy
from trading.strategy.risk import RiskManager, RiskConfig
from trading.config.settings import (
    TradingConfig,
    ExchangeSetup,
    ExchangeType,
    load_config,
)


class TradingEngine:
    """
    Main orchestrator for the trading framework.

    Manages:
    - Multiple exchange connections
    - Strategy execution
    - Risk management
    - Event coordination

    Usage:
        # From configuration file
        engine = TradingEngine.from_config("config.yaml")

        # Or programmatic setup
        engine = TradingEngine()

        # Add simulator for paper trading
        engine.add_simulated_exchange(
            initial_balance={"USD": 100000, "BTC": 1.0}
        )
        engine.add_instrument("BTCUSDT", 50000.0)

        # Or connect to real exchange
        engine.add_binance_exchange(api_key="...", api_secret="...")

        # Start engine
        await engine.start()

        # Execute trades
        order = await engine.submit_order("BTCUSDT", OrderSide.BUY, 0.1)

        # Or run a strategy
        strategy = MyStrategy(engine.get_exchange(), strategy_config)
        engine.add_strategy(strategy)
        await engine.run()
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config
        self.events = EventEmitter()

        # Exchange connections
        self._exchanges: dict[str, Exchange] = {}
        self._default_exchange: Optional[str] = None

        # Strategies
        self._strategies: dict[str, Strategy] = {}

        # Risk management
        self._risk_manager: Optional[RiskManager] = None

        # Calibration data
        self._calibrators: dict[str, MarketCalibrator] = {}
        self._calibration_results: dict[str, CalibrationResult] = {}

        # State
        self._running = False
        self._start_time: Optional[datetime] = None

        # Initialize from config if provided
        if config:
            self._init_from_config(config)

    @classmethod
    def from_config(cls, config_path: str) -> "TradingEngine":
        """Create engine from configuration file."""
        config = load_config(config_path)
        return cls(config)

    def _init_from_config(self, config: TradingConfig) -> None:
        """Initialize engine from configuration."""
        # Setup risk manager
        self._risk_manager = RiskManager(
            RiskConfig(
                max_position_pct=config.risk.max_position_pct,
                max_total_exposure=config.risk.max_total_exposure,
                max_loss_per_trade_pct=config.risk.max_loss_per_trade_pct,
                daily_loss_limit_pct=config.risk.daily_loss_limit_pct,
                max_drawdown_pct=config.risk.max_drawdown_pct,
            ),
            initial_equity=Decimal(str(sum(config.simulator.initial_balance.values()))),
        )

        # Setup exchanges
        for exchange_setup in config.exchanges:
            self._setup_exchange(exchange_setup)

        # Setup simulator if enabled
        if config.simulator.enabled and not any(
            ex.type == ExchangeType.SIMULATED for ex in config.exchanges
        ):
            self.add_simulated_exchange(
                initial_balance=config.simulator.initial_balance,
                volatility=config.simulator.default_volatility,
                spread_pct=config.simulator.default_spread_pct,
                commission_rate=config.simulator.commission_rate,
            )

    def _setup_exchange(self, setup: ExchangeSetup) -> None:
        """Setup an exchange from configuration."""
        if setup.type == ExchangeType.BINANCE:
            self.add_binance_exchange(
                name=setup.name,
                api_key=setup.api_key,
                api_secret=setup.api_secret,
                testnet=setup.testnet or setup.paper_trading,
            )
        elif setup.type == ExchangeType.BINANCE_FUTURES:
            self.add_binance_exchange(
                name=setup.name,
                api_key=setup.api_key,
                api_secret=setup.api_secret,
                testnet=setup.testnet or setup.paper_trading,
                futures=True,
            )
        elif setup.type == ExchangeType.INTERACTIVE_BROKERS:
            self.add_ib_exchange(
                name=setup.name,
                host=setup.host or "127.0.0.1",
                port=setup.port or 7497,
                client_id=setup.client_id,
                account_id=setup.account_id,
            )
        elif setup.type == ExchangeType.OANDA:
            self.add_oanda_exchange(
                name=setup.name,
                api_key=setup.api_key,
                account_id=setup.account_id,
                practice=setup.paper_trading,
            )
        elif setup.type == ExchangeType.SIMULATED:
            self.add_simulated_exchange(name=setup.name)

    # ==================== Exchange Management ====================

    def add_simulated_exchange(
        self,
        name: str = "simulator",
        initial_balance: Optional[dict[str, float]] = None,
        volatility: float = 0.20,
        spread_pct: float = 0.001,
        commission_rate: float = 0.001,
        random_seed: Optional[int] = None,
    ) -> SimulatedExchange:
        """
        Add a simulated exchange for paper trading.

        Args:
            name: Exchange identifier
            initial_balance: Starting balances by currency
            volatility: Default annualized volatility
            spread_pct: Default bid-ask spread
            commission_rate: Trading commission rate
            random_seed: Random seed for reproducibility

        Returns:
            SimulatedExchange instance
        """
        config = SimulatorConfig(
            initial_balance=initial_balance or {"USD": 100000.0},
            default_model_config=MarketModelConfig(
                volatility=volatility,
                spread_pct=spread_pct,
            ),
            commission_rate=commission_rate,
            random_seed=random_seed,
        )

        exchange = SimulatedExchange(config)
        self._exchanges[name] = exchange

        if not self._default_exchange:
            self._default_exchange = name

        return exchange

    def add_binance_exchange(
        self,
        name: str = "binance",
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        futures: bool = False,
    ) -> BinanceExchange:
        """
        Add Binance exchange connection.

        Args:
            name: Exchange identifier
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet (paper trading)
            futures: Use futures market

        Returns:
            BinanceExchange instance
        """
        config = BinanceConfig(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            use_futures=futures,
        )

        exchange = BinanceExchange(config)
        self._exchanges[name] = exchange

        if not self._default_exchange:
            self._default_exchange = name

        return exchange

    def add_ib_exchange(
        self,
        name: str = "ib",
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        account_id: str = "",
    ) -> InteractiveBrokersExchange:
        """
        Add Interactive Brokers connection.

        Args:
            name: Exchange identifier
            host: TWS/Gateway host
            port: TWS/Gateway port
            client_id: Client ID
            account_id: Account ID

        Returns:
            InteractiveBrokersExchange instance
        """
        config = IBConfig(
            host=host,
            port=port,
            client_id=client_id,
            account_id=account_id,
        )

        exchange = InteractiveBrokersExchange(config)
        self._exchanges[name] = exchange

        if not self._default_exchange:
            self._default_exchange = name

        return exchange

    def add_oanda_exchange(
        self,
        name: str = "oanda",
        api_key: str = "",
        account_id: str = "",
        practice: bool = True,
    ) -> OandaExchange:
        """
        Add OANDA exchange connection.

        Args:
            name: Exchange identifier
            api_key: OANDA API token
            account_id: OANDA account ID
            practice: Use practice account

        Returns:
            OandaExchange instance
        """
        config = OandaConfig(
            api_key=api_key,
            account_id=account_id,
            practice=practice,
        )

        exchange = OandaExchange(config)
        self._exchanges[name] = exchange

        if not self._default_exchange:
            self._default_exchange = name

        return exchange

    def get_exchange(self, name: Optional[str] = None) -> Exchange:
        """
        Get an exchange by name.

        Args:
            name: Exchange name (uses default if None)

        Returns:
            Exchange instance
        """
        name = name or self._default_exchange
        if not name or name not in self._exchanges:
            raise ValueError(f"Exchange not found: {name}")
        return self._exchanges[name]

    # ==================== Instrument Management ====================

    def add_instrument(
        self,
        symbol: str,
        initial_price: float,
        exchange_name: Optional[str] = None,
        volatility: Optional[float] = None,
        spread_pct: Optional[float] = None,
        asset_class: AssetClass = AssetClass.CRYPTO,
        calibration: Optional[CalibrationResult] = None,
    ) -> None:
        """
        Add a trading instrument to the simulated exchange.

        Args:
            symbol: Trading symbol
            initial_price: Starting price
            exchange_name: Target exchange (must be simulated)
            volatility: Annual volatility
            spread_pct: Bid-ask spread percentage
            asset_class: Asset classification
            calibration: Optional calibration results
        """
        exchange = self.get_exchange(exchange_name)

        if not isinstance(exchange, SimulatedExchange):
            raise ValueError("Can only add instruments to simulated exchange")

        if calibration:
            exchange.add_instrument_from_calibration(
                symbol=symbol,
                initial_price=initial_price,
                calibration=calibration,
                asset_class=asset_class,
            )
        else:
            config = MarketModelConfig(
                volatility=volatility or 0.20,
                spread_pct=spread_pct or 0.001,
            )
            exchange.add_instrument(
                symbol=symbol,
                initial_price=initial_price,
                model_config=config,
                asset_class=asset_class,
            )

    # ==================== Calibration ====================

    def get_calibrator(self, symbol: str) -> MarketCalibrator:
        """Get or create a calibrator for a symbol."""
        if symbol not in self._calibrators:
            self._calibrators[symbol] = MarketCalibrator(symbol)
        return self._calibrators[symbol]

    async def calibrate_from_exchange(
        self,
        symbol: str,
        exchange_name: Optional[str] = None,
        resolution: DataResolution = DataResolution.HOUR_1,
        lookback_days: int = 30,
    ) -> CalibrationResult:
        """
        Calibrate market model from real exchange data.

        Args:
            symbol: Symbol to calibrate
            exchange_name: Source exchange
            resolution: Data resolution
            lookback_days: Historical data lookback

        Returns:
            Calibration results
        """
        exchange = self.get_exchange(exchange_name)
        calibrator = self.get_calibrator(symbol)

        # Fetch historical data
        from datetime import timedelta
        end = datetime.utcnow()
        start = end - timedelta(days=lookback_days)

        bars = await exchange.get_historical_bars(
            symbol=symbol,
            resolution=resolution,
            start=start,
            end=end,
        )

        calibrator.add_bars(bars)
        result = calibrator.calibrate()

        self._calibration_results[symbol] = result
        return result

    # ==================== Strategy Management ====================

    def add_strategy(self, strategy: Strategy, name: Optional[str] = None) -> None:
        """
        Add a strategy to the engine.

        Args:
            strategy: Strategy instance
            name: Strategy identifier
        """
        name = name or strategy.config.name
        self._strategies[name] = strategy

        # Wire up events
        strategy.events.on_all(lambda e: self.events.emit(e))

    def get_strategy(self, name: str) -> Optional[Strategy]:
        """Get a strategy by name."""
        return self._strategies.get(name)

    # ==================== Lifecycle ====================

    async def start(self) -> None:
        """Start the trading engine and all exchanges."""
        if self._running:
            return

        self._running = True
        self._start_time = datetime.utcnow()

        # Connect all exchanges
        for name, exchange in self._exchanges.items():
            try:
                await exchange.connect()
                print(f"Connected to {exchange.name}")
            except Exception as e:
                print(f"Failed to connect to {name}: {e}")

        self.events.emit(Event(
            event_type=EventType.CONNECTED,
            source="TradingEngine",
        ))

    async def stop(self) -> None:
        """Stop the trading engine."""
        self._running = False

        # Stop all strategies
        for strategy in self._strategies.values():
            await strategy.stop()

        # Disconnect all exchanges
        for exchange in self._exchanges.values():
            await exchange.disconnect()

        self.events.emit(Event(
            event_type=EventType.DISCONNECTED,
            source="TradingEngine",
        ))

    async def run(self, until: Optional[datetime] = None) -> None:
        """
        Run the engine with all strategies.

        Args:
            until: Optional stop time
        """
        await self.start()

        # Start all strategies
        for strategy in self._strategies.values():
            await strategy.start()

        try:
            # Run until stopped or until time reached
            while self._running:
                if until and datetime.utcnow() >= until:
                    break
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\nStopping engine...")

        finally:
            await self.stop()

    # ==================== Trading Operations ====================

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        exchange_name: Optional[str] = None,
    ) -> Order:
        """
        Submit an order.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            order_type: Order type
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            exchange_name: Target exchange

        Returns:
            Submitted order
        """
        exchange = self.get_exchange(exchange_name)

        order = Order(
            symbol=symbol,
            side=side,
            quantity=Decimal(str(quantity)),
            order_type=order_type,
            limit_price=Decimal(str(limit_price)) if limit_price else None,
            stop_price=Decimal(str(stop_price)) if stop_price else None,
        )

        # Risk check
        if self._risk_manager:
            ticker = await exchange.get_ticker(symbol)
            price = ticker.mid_price if ticker else Decimal(str(limit_price or 0))

            allowed, reason = self._risk_manager.can_open_position(
                symbol, order.quantity, price, side
            )
            if not allowed:
                raise ValueError(f"Risk check failed: {reason}")

        return await exchange.submit_order(order)

    async def close_position(
        self,
        symbol: str,
        exchange_name: Optional[str] = None,
    ) -> Optional[Order]:
        """
        Close a position.

        Args:
            symbol: Position symbol
            exchange_name: Target exchange

        Returns:
            Closing order or None
        """
        exchange = self.get_exchange(exchange_name)
        return await exchange.close_position(symbol)

    async def close_all_positions(
        self,
        exchange_name: Optional[str] = None,
    ) -> list[Order]:
        """Close all positions on an exchange."""
        exchange = self.get_exchange(exchange_name)
        positions = await exchange.get_positions()

        orders = []
        for position in positions:
            order = await exchange.close_position(position.symbol)
            if order:
                orders.append(order)

        return orders

    # ==================== Market Data ====================

    async def get_ticker(
        self,
        symbol: str,
        exchange_name: Optional[str] = None,
    ) -> Optional[Tick]:
        """Get current ticker."""
        exchange = self.get_exchange(exchange_name)
        return await exchange.get_ticker(symbol)

    async def get_positions(
        self,
        exchange_name: Optional[str] = None,
    ) -> list[Position]:
        """Get all positions."""
        exchange = self.get_exchange(exchange_name)
        return await exchange.get_positions()

    async def subscribe_ticks(
        self,
        symbols: list[str],
        callback: Callable[[Tick], None],
        exchange_name: Optional[str] = None,
    ) -> None:
        """Subscribe to tick data."""
        exchange = self.get_exchange(exchange_name)
        await exchange.subscribe_ticks(symbols, callback)

    # ==================== Account ====================

    async def get_equity(self, exchange_name: Optional[str] = None) -> Decimal:
        """Get total account equity."""
        exchange = self.get_exchange(exchange_name)

        if isinstance(exchange, SimulatedExchange):
            return exchange.get_equity()

        balances = await exchange.get_balances()
        return sum(b.total for b in balances)

    @property
    def risk_manager(self) -> Optional[RiskManager]:
        """Get the risk manager."""
        return self._risk_manager

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running
