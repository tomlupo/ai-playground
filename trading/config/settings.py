"""
Configuration system for the trading framework.

Supports:
- YAML configuration files
- Environment variable overrides
- Multiple exchange setups
- Strategy configurations
"""

import os
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Optional, Any
import json

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ExchangeType(str, Enum):
    """Supported exchange types."""
    BINANCE = "binance"
    BINANCE_FUTURES = "binance_futures"
    INTERACTIVE_BROKERS = "interactive_brokers"
    OANDA = "oanda"
    SIMULATED = "simulated"


@dataclass
class ExchangeSetup:
    """Configuration for a single exchange connection."""
    type: ExchangeType
    name: str = ""

    # Credentials (can be overridden by env vars)
    api_key: str = ""
    api_secret: str = ""

    # Connection settings
    testnet: bool = False
    sandbox: bool = False

    # Exchange-specific settings
    account_id: str = ""
    host: str = ""
    port: int = 0
    client_id: int = 1

    # Paper trading
    paper_trading: bool = True

    # Extra settings
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.type.value

        # Load credentials from environment if not provided
        env_prefix = self.type.value.upper()
        if not self.api_key:
            self.api_key = os.getenv(f"{env_prefix}_API_KEY", "")
        if not self.api_secret:
            self.api_secret = os.getenv(f"{env_prefix}_API_SECRET", "")
        if not self.account_id:
            self.account_id = os.getenv(f"{env_prefix}_ACCOUNT_ID", "")


@dataclass
class SimulatorSetup:
    """Configuration for the market simulator."""
    enabled: bool = True

    # Initial balances
    initial_balance: dict[str, float] = field(default_factory=lambda: {"USD": 100000.0})

    # Market model defaults
    default_volatility: float = 0.20
    default_spread_pct: float = 0.001

    # Execution simulation
    commission_rate: float = 0.001
    slippage_bps: float = 1.0
    fill_latency_ms: float = 50.0

    # Random seed for reproducibility (0 = random)
    random_seed: int = 0


@dataclass
class RiskSettings:
    """Risk management settings."""
    # Position limits
    max_position_pct: float = 0.10
    max_total_exposure: float = 1.0

    # Loss limits
    max_loss_per_trade_pct: float = 0.02
    daily_loss_limit_pct: float = 0.05
    max_drawdown_pct: float = 0.20

    # Position sizing
    default_sizer: str = "percent_equity"  # fixed, percent_equity, volatility, kelly
    default_size_pct: float = 0.02


@dataclass
class LoggingSettings:
    """Logging configuration."""
    level: str = "INFO"
    log_file: str = ""
    log_trades: bool = True
    log_orders: bool = True
    log_signals: bool = True


@dataclass
class TradingConfig:
    """
    Main configuration for the trading framework.

    Can be loaded from YAML file or constructed programmatically.

    Usage:
        # From YAML
        config = load_config("config.yaml")

        # Programmatic
        config = TradingConfig(
            exchanges=[
                ExchangeSetup(
                    type=ExchangeType.BINANCE,
                    testnet=True,
                ),
            ],
            symbols=["BTCUSDT", "ETHUSDT"],
        )
    """
    # Exchange configurations
    exchanges: list[ExchangeSetup] = field(default_factory=list)

    # Simulator configuration
    simulator: SimulatorSetup = field(default_factory=SimulatorSetup)

    # Trading symbols
    symbols: list[str] = field(default_factory=list)

    # Risk settings
    risk: RiskSettings = field(default_factory=RiskSettings)

    # Logging
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    # Strategy settings (strategy-specific, passed through)
    strategy: dict = field(default_factory=dict)

    # Data directory
    data_dir: str = "./data"

    # Mode
    mode: str = "paper"  # live, paper, backtest

    def get_exchange_setup(self, exchange_type: ExchangeType) -> Optional[ExchangeSetup]:
        """Get exchange setup by type."""
        for setup in self.exchanges:
            if setup.type == exchange_type:
                return setup
        return None

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if not self.exchanges and not self.simulator.enabled:
            issues.append("No exchanges configured and simulator disabled")

        if not self.symbols:
            issues.append("No trading symbols configured")

        for exchange in self.exchanges:
            if exchange.type != ExchangeType.SIMULATED:
                if not exchange.api_key and not exchange.paper_trading:
                    issues.append(f"No API key for {exchange.name}")

        if self.risk.max_position_pct <= 0 or self.risk.max_position_pct > 1:
            issues.append("Invalid max_position_pct (should be 0-1)")

        return issues


def load_config(path: str) -> TradingConfig:
    """
    Load configuration from file.

    Supports YAML and JSON formats.

    Args:
        path: Path to configuration file

    Returns:
        TradingConfig instance
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for YAML config files: pip install pyyaml")
        data = yaml.safe_load(content)
    elif path.suffix == ".json":
        data = json.loads(content)
    else:
        # Try YAML first, then JSON
        try:
            if YAML_AVAILABLE:
                data = yaml.safe_load(content)
            else:
                data = json.loads(content)
        except Exception:
            data = json.loads(content)

    return _dict_to_config(data)


def save_config(config: TradingConfig, path: str) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        path: Output file path
    """
    path = Path(path)

    data = _config_to_dict(config)

    if path.suffix in (".yaml", ".yml"):
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for YAML config files")
        content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    else:
        content = json.dumps(data, indent=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _dict_to_config(data: dict) -> TradingConfig:
    """Convert dictionary to TradingConfig."""
    # Parse exchanges
    exchanges = []
    for ex_data in data.get("exchanges", []):
        ex_type = ex_data.get("type", "simulated")
        if isinstance(ex_type, str):
            ex_type = ExchangeType(ex_type)

        exchanges.append(ExchangeSetup(
            type=ex_type,
            name=ex_data.get("name", ""),
            api_key=ex_data.get("api_key", ""),
            api_secret=ex_data.get("api_secret", ""),
            testnet=ex_data.get("testnet", False),
            sandbox=ex_data.get("sandbox", False),
            account_id=ex_data.get("account_id", ""),
            host=ex_data.get("host", ""),
            port=ex_data.get("port", 0),
            client_id=ex_data.get("client_id", 1),
            paper_trading=ex_data.get("paper_trading", True),
            extra=ex_data.get("extra", {}),
        ))

    # Parse simulator
    sim_data = data.get("simulator", {})
    simulator = SimulatorSetup(
        enabled=sim_data.get("enabled", True),
        initial_balance=sim_data.get("initial_balance", {"USD": 100000.0}),
        default_volatility=sim_data.get("default_volatility", 0.20),
        default_spread_pct=sim_data.get("default_spread_pct", 0.001),
        commission_rate=sim_data.get("commission_rate", 0.001),
        slippage_bps=sim_data.get("slippage_bps", 1.0),
        fill_latency_ms=sim_data.get("fill_latency_ms", 50.0),
        random_seed=sim_data.get("random_seed", 0),
    )

    # Parse risk
    risk_data = data.get("risk", {})
    risk = RiskSettings(
        max_position_pct=risk_data.get("max_position_pct", 0.10),
        max_total_exposure=risk_data.get("max_total_exposure", 1.0),
        max_loss_per_trade_pct=risk_data.get("max_loss_per_trade_pct", 0.02),
        daily_loss_limit_pct=risk_data.get("daily_loss_limit_pct", 0.05),
        max_drawdown_pct=risk_data.get("max_drawdown_pct", 0.20),
        default_sizer=risk_data.get("default_sizer", "percent_equity"),
        default_size_pct=risk_data.get("default_size_pct", 0.02),
    )

    # Parse logging
    log_data = data.get("logging", {})
    logging = LoggingSettings(
        level=log_data.get("level", "INFO"),
        log_file=log_data.get("log_file", ""),
        log_trades=log_data.get("log_trades", True),
        log_orders=log_data.get("log_orders", True),
        log_signals=log_data.get("log_signals", True),
    )

    return TradingConfig(
        exchanges=exchanges,
        simulator=simulator,
        symbols=data.get("symbols", []),
        risk=risk,
        logging=logging,
        strategy=data.get("strategy", {}),
        data_dir=data.get("data_dir", "./data"),
        mode=data.get("mode", "paper"),
    )


def _config_to_dict(config: TradingConfig) -> dict:
    """Convert TradingConfig to dictionary."""
    return {
        "mode": config.mode,
        "data_dir": config.data_dir,
        "symbols": config.symbols,
        "exchanges": [
            {
                "type": ex.type.value,
                "name": ex.name,
                "testnet": ex.testnet,
                "sandbox": ex.sandbox,
                "paper_trading": ex.paper_trading,
                "account_id": ex.account_id,
                "host": ex.host,
                "port": ex.port,
                "client_id": ex.client_id,
                "extra": ex.extra,
                # Note: Don't save credentials
            }
            for ex in config.exchanges
        ],
        "simulator": {
            "enabled": config.simulator.enabled,
            "initial_balance": config.simulator.initial_balance,
            "default_volatility": config.simulator.default_volatility,
            "default_spread_pct": config.simulator.default_spread_pct,
            "commission_rate": config.simulator.commission_rate,
            "slippage_bps": config.simulator.slippage_bps,
            "fill_latency_ms": config.simulator.fill_latency_ms,
            "random_seed": config.simulator.random_seed,
        },
        "risk": {
            "max_position_pct": config.risk.max_position_pct,
            "max_total_exposure": config.risk.max_total_exposure,
            "max_loss_per_trade_pct": config.risk.max_loss_per_trade_pct,
            "daily_loss_limit_pct": config.risk.daily_loss_limit_pct,
            "max_drawdown_pct": config.risk.max_drawdown_pct,
            "default_sizer": config.risk.default_sizer,
            "default_size_pct": config.risk.default_size_pct,
        },
        "logging": {
            "level": config.logging.level,
            "log_file": config.logging.log_file,
            "log_trades": config.logging.log_trades,
            "log_orders": config.logging.log_orders,
            "log_signals": config.logging.log_signals,
        },
        "strategy": config.strategy,
    }


def create_default_config() -> TradingConfig:
    """Create a default configuration for getting started."""
    return TradingConfig(
        mode="paper",
        symbols=["BTCUSDT"],
        exchanges=[
            ExchangeSetup(
                type=ExchangeType.SIMULATED,
                name="simulator",
            ),
        ],
        simulator=SimulatorSetup(
            enabled=True,
            initial_balance={"USD": 100000.0, "BTC": 1.0},
        ),
        risk=RiskSettings(
            max_position_pct=0.10,
            daily_loss_limit_pct=0.05,
        ),
    )
