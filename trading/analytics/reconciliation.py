"""
Account reconciliation between local records and broker.

Provides:
- Position reconciliation
- Balance reconciliation
- Order status verification
- Trade matching
- Discrepancy detection and alerting
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Callable, Any
import asyncio

from trading.core.models import Order, Position, Balance, Trade, Fill
from trading.core.enums import OrderStatus
from trading.exchanges.base import Exchange


class DiscrepancyType(Enum):
    """Types of reconciliation discrepancies."""
    # Position discrepancies
    POSITION_QUANTITY_MISMATCH = auto()
    POSITION_MISSING_LOCAL = auto()
    POSITION_MISSING_BROKER = auto()
    POSITION_PRICE_MISMATCH = auto()

    # Balance discrepancies
    BALANCE_MISMATCH = auto()
    BALANCE_MISSING_LOCAL = auto()
    BALANCE_MISSING_BROKER = auto()

    # Order discrepancies
    ORDER_STATUS_MISMATCH = auto()
    ORDER_MISSING_LOCAL = auto()
    ORDER_MISSING_BROKER = auto()
    ORDER_FILL_MISMATCH = auto()

    # Trade discrepancies
    TRADE_MISSING = auto()
    TRADE_PRICE_MISMATCH = auto()
    TRADE_QUANTITY_MISMATCH = auto()


class DiscrepancySeverity(Enum):
    """Severity level of discrepancies."""
    INFO = auto()      # Minor, informational
    WARNING = auto()   # Should be investigated
    CRITICAL = auto()  # Requires immediate attention


@dataclass
class Discrepancy:
    """A detected discrepancy between local and broker records."""
    type: DiscrepancyType
    severity: DiscrepancySeverity
    symbol: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Local vs broker values
    local_value: Any = None
    broker_value: Any = None
    difference: Optional[Decimal] = None

    # Context
    description: str = ""
    order_id: Optional[str] = None
    trade_id: Optional[str] = None

    # Resolution
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.severity.name}] {self.type.name}: {self.symbol} - "
            f"Local: {self.local_value}, Broker: {self.broker_value}"
        )


@dataclass
class ReconciliationResult:
    """Result of a reconciliation run."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0

    # Status
    success: bool = True
    error_message: str = ""

    # Discrepancies found
    discrepancies: list[Discrepancy] = field(default_factory=list)

    # Counts
    positions_checked: int = 0
    positions_matched: int = 0
    balances_checked: int = 0
    balances_matched: int = 0
    orders_checked: int = 0
    orders_matched: int = 0

    @property
    def has_discrepancies(self) -> bool:
        return len(self.discrepancies) > 0

    @property
    def critical_count(self) -> int:
        return sum(1 for d in self.discrepancies if d.severity == DiscrepancySeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for d in self.discrepancies if d.severity == DiscrepancySeverity.WARNING)

    def get_discrepancies_by_type(self, disc_type: DiscrepancyType) -> list[Discrepancy]:
        return [d for d in self.discrepancies if d.type == disc_type]

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Reconciliation at {self.timestamp.isoformat()}",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Status: {'OK' if self.success else 'FAILED'}",
            f"Positions: {self.positions_matched}/{self.positions_checked} matched",
            f"Balances: {self.balances_matched}/{self.balances_checked} matched",
            f"Orders: {self.orders_matched}/{self.orders_checked} matched",
            f"Discrepancies: {len(self.discrepancies)} "
            f"({self.critical_count} critical, {self.warning_count} warnings)",
        ]
        return "\n".join(lines)


class AccountReconciler:
    """
    Reconciles local trading records with broker/exchange records.

    Detects and reports discrepancies in:
    - Positions (quantities, prices)
    - Balances (cash, margins)
    - Orders (status, fills)
    - Trades (executions)

    Usage:
        reconciler = AccountReconciler(exchange)

        # Set local state
        reconciler.set_local_positions(my_positions)
        reconciler.set_local_balances(my_balances)
        reconciler.set_local_orders(my_orders)

        # Run reconciliation
        result = await reconciler.reconcile()

        if result.has_discrepancies:
            for disc in result.discrepancies:
                print(f"Discrepancy: {disc}")

        # Monitor continuously
        reconciler.on_discrepancy(alert_handler)
        await reconciler.start_monitoring(interval_seconds=60)
    """

    def __init__(
        self,
        exchange: Exchange,
        tolerance_pct: float = 0.001,  # 0.1% tolerance
        tolerance_absolute: Decimal = Decimal("0.01"),
    ):
        self.exchange = exchange
        self.tolerance_pct = tolerance_pct
        self.tolerance_absolute = tolerance_absolute

        # Local records
        self._local_positions: dict[str, Position] = {}
        self._local_balances: dict[str, Balance] = {}
        self._local_orders: dict[str, Order] = {}
        self._local_trades: list[Trade] = []

        # History
        self._reconciliation_history: list[ReconciliationResult] = []
        self._discrepancy_history: list[Discrepancy] = []

        # Callbacks
        self._discrepancy_callbacks: list[Callable[[Discrepancy], None]] = []

        # Monitoring
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    # ==================== Local Record Management ====================

    def set_local_positions(self, positions: dict[str, Position]) -> None:
        """Set local position records."""
        self._local_positions = positions.copy()

    def set_local_balances(self, balances: dict[str, Balance]) -> None:
        """Set local balance records."""
        self._local_balances = balances.copy()

    def set_local_orders(self, orders: dict[str, Order]) -> None:
        """Set local order records."""
        self._local_orders = orders.copy()

    def update_local_position(self, symbol: str, position: Position) -> None:
        """Update a single local position."""
        self._local_positions[symbol] = position

    def update_local_balance(self, currency: str, balance: Balance) -> None:
        """Update a single local balance."""
        self._local_balances[currency] = balance

    def update_local_order(self, order_id: str, order: Order) -> None:
        """Update a single local order."""
        self._local_orders[order_id] = order

    # ==================== Reconciliation ====================

    async def reconcile(self) -> ReconciliationResult:
        """
        Run full reconciliation against broker.

        Returns:
            ReconciliationResult with all findings.
        """
        start_time = datetime.utcnow()
        result = ReconciliationResult()

        try:
            # Reconcile positions
            position_result = await self._reconcile_positions()
            result.positions_checked = position_result["checked"]
            result.positions_matched = position_result["matched"]
            result.discrepancies.extend(position_result["discrepancies"])

            # Reconcile balances
            balance_result = await self._reconcile_balances()
            result.balances_checked = balance_result["checked"]
            result.balances_matched = balance_result["matched"]
            result.discrepancies.extend(balance_result["discrepancies"])

            # Reconcile orders
            order_result = await self._reconcile_orders()
            result.orders_checked = order_result["checked"]
            result.orders_matched = order_result["matched"]
            result.discrepancies.extend(order_result["discrepancies"])

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()
        result.timestamp = start_time

        # Store in history
        self._reconciliation_history.append(result)
        self._discrepancy_history.extend(result.discrepancies)

        # Notify callbacks
        for disc in result.discrepancies:
            self._notify_discrepancy(disc)

        return result

    async def _reconcile_positions(self) -> dict:
        """Reconcile positions."""
        result = {"checked": 0, "matched": 0, "discrepancies": []}

        # Get broker positions
        broker_positions = await self.exchange.get_positions()
        broker_pos_dict = {p.symbol: p for p in broker_positions}

        # Check all local positions
        all_symbols = set(self._local_positions.keys()) | set(broker_pos_dict.keys())

        for symbol in all_symbols:
            result["checked"] += 1

            local_pos = self._local_positions.get(symbol)
            broker_pos = broker_pos_dict.get(symbol)

            if local_pos and not broker_pos:
                # Position exists locally but not on broker
                result["discrepancies"].append(Discrepancy(
                    type=DiscrepancyType.POSITION_MISSING_BROKER,
                    severity=DiscrepancySeverity.CRITICAL,
                    symbol=symbol,
                    local_value=float(local_pos.quantity),
                    broker_value=0,
                    description=f"Position {symbol} exists locally ({local_pos.quantity}) but not on broker",
                ))

            elif broker_pos and not local_pos:
                # Position exists on broker but not locally
                result["discrepancies"].append(Discrepancy(
                    type=DiscrepancyType.POSITION_MISSING_LOCAL,
                    severity=DiscrepancySeverity.CRITICAL,
                    symbol=symbol,
                    local_value=0,
                    broker_value=float(broker_pos.quantity),
                    description=f"Position {symbol} exists on broker ({broker_pos.quantity}) but not locally",
                ))

            elif local_pos and broker_pos:
                # Both exist - check quantities
                if not self._values_match(local_pos.quantity, broker_pos.quantity):
                    result["discrepancies"].append(Discrepancy(
                        type=DiscrepancyType.POSITION_QUANTITY_MISMATCH,
                        severity=DiscrepancySeverity.CRITICAL,
                        symbol=symbol,
                        local_value=float(local_pos.quantity),
                        broker_value=float(broker_pos.quantity),
                        difference=local_pos.quantity - broker_pos.quantity,
                        description=f"Position quantity mismatch for {symbol}",
                    ))
                else:
                    result["matched"] += 1

        return result

    async def _reconcile_balances(self) -> dict:
        """Reconcile account balances."""
        result = {"checked": 0, "matched": 0, "discrepancies": []}

        # Get broker balances
        broker_balances = await self.exchange.get_balances()
        broker_bal_dict = {b.currency: b for b in broker_balances}

        # Check all currencies
        all_currencies = set(self._local_balances.keys()) | set(broker_bal_dict.keys())

        for currency in all_currencies:
            result["checked"] += 1

            local_bal = self._local_balances.get(currency)
            broker_bal = broker_bal_dict.get(currency)

            if local_bal and not broker_bal:
                result["discrepancies"].append(Discrepancy(
                    type=DiscrepancyType.BALANCE_MISSING_BROKER,
                    severity=DiscrepancySeverity.WARNING,
                    symbol=currency,
                    local_value=float(local_bal.total),
                    broker_value=0,
                ))

            elif broker_bal and not local_bal:
                # Only report if significant balance
                if broker_bal.total > self.tolerance_absolute:
                    result["discrepancies"].append(Discrepancy(
                        type=DiscrepancyType.BALANCE_MISSING_LOCAL,
                        severity=DiscrepancySeverity.WARNING,
                        symbol=currency,
                        local_value=0,
                        broker_value=float(broker_bal.total),
                    ))

            elif local_bal and broker_bal:
                if not self._values_match(local_bal.total, broker_bal.total):
                    result["discrepancies"].append(Discrepancy(
                        type=DiscrepancyType.BALANCE_MISMATCH,
                        severity=DiscrepancySeverity.WARNING,
                        symbol=currency,
                        local_value=float(local_bal.total),
                        broker_value=float(broker_bal.total),
                        difference=local_bal.total - broker_bal.total,
                    ))
                else:
                    result["matched"] += 1

        return result

    async def _reconcile_orders(self) -> dict:
        """Reconcile open orders."""
        result = {"checked": 0, "matched": 0, "discrepancies": []}

        # Get broker open orders
        broker_orders = await self.exchange.get_open_orders()
        broker_order_dict = {}
        for o in broker_orders:
            key = o.exchange_order_id or o.order_id
            broker_order_dict[key] = o

        # Check local active orders
        for order_id, local_order in self._local_orders.items():
            if not local_order.is_active:
                continue

            result["checked"] += 1

            # Find matching broker order
            broker_order = broker_order_dict.get(local_order.exchange_order_id)
            if not broker_order:
                broker_order = broker_order_dict.get(order_id)

            if not broker_order:
                # Order missing on broker (might be filled/cancelled)
                result["discrepancies"].append(Discrepancy(
                    type=DiscrepancyType.ORDER_MISSING_BROKER,
                    severity=DiscrepancySeverity.WARNING,
                    symbol=local_order.symbol,
                    local_value=local_order.status.name,
                    broker_value="NOT FOUND",
                    order_id=order_id,
                    description=f"Order {order_id} not found on broker",
                ))
            else:
                # Check status match
                if local_order.status != broker_order.status:
                    result["discrepancies"].append(Discrepancy(
                        type=DiscrepancyType.ORDER_STATUS_MISMATCH,
                        severity=DiscrepancySeverity.WARNING,
                        symbol=local_order.symbol,
                        local_value=local_order.status.name,
                        broker_value=broker_order.status.name,
                        order_id=order_id,
                    ))
                # Check filled quantity
                elif not self._values_match(local_order.filled_quantity, broker_order.filled_quantity):
                    result["discrepancies"].append(Discrepancy(
                        type=DiscrepancyType.ORDER_FILL_MISMATCH,
                        severity=DiscrepancySeverity.WARNING,
                        symbol=local_order.symbol,
                        local_value=float(local_order.filled_quantity),
                        broker_value=float(broker_order.filled_quantity),
                        order_id=order_id,
                    ))
                else:
                    result["matched"] += 1

        return result

    def _values_match(self, local: Decimal, broker: Decimal) -> bool:
        """Check if two values match within tolerance."""
        if local == broker:
            return True

        # Check absolute tolerance
        diff = abs(local - broker)
        if diff <= self.tolerance_absolute:
            return True

        # Check percentage tolerance
        if local != 0:
            pct_diff = diff / abs(local)
            if pct_diff <= Decimal(str(self.tolerance_pct)):
                return True

        return False

    # ==================== Monitoring ====================

    def on_discrepancy(self, callback: Callable[[Discrepancy], None]) -> None:
        """Register callback for discrepancy notifications."""
        self._discrepancy_callbacks.append(callback)

    def _notify_discrepancy(self, discrepancy: Discrepancy) -> None:
        """Notify all registered callbacks."""
        for callback in self._discrepancy_callbacks:
            try:
                callback(discrepancy)
            except Exception as e:
                print(f"Discrepancy callback error: {e}")

    async def start_monitoring(
        self,
        interval_seconds: float = 60.0,
        on_result: Optional[Callable[[ReconciliationResult], None]] = None,
    ) -> None:
        """
        Start continuous reconciliation monitoring.

        Args:
            interval_seconds: Time between reconciliation runs
            on_result: Callback for each reconciliation result
        """
        self._monitoring = True

        async def monitor_loop():
            while self._monitoring:
                try:
                    result = await self.reconcile()
                    if on_result:
                        on_result(result)
                except Exception as e:
                    print(f"Monitoring error: {e}")

                await asyncio.sleep(interval_seconds)

        self._monitor_task = asyncio.create_task(monitor_loop())

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()

    # ==================== Reporting ====================

    def get_discrepancy_history(
        self,
        since: Optional[datetime] = None,
        disc_type: Optional[DiscrepancyType] = None,
        severity: Optional[DiscrepancySeverity] = None,
        unresolved_only: bool = False,
    ) -> list[Discrepancy]:
        """Get historical discrepancies with filters."""
        result = self._discrepancy_history

        if since:
            result = [d for d in result if d.timestamp >= since]

        if disc_type:
            result = [d for d in result if d.type == disc_type]

        if severity:
            result = [d for d in result if d.severity == severity]

        if unresolved_only:
            result = [d for d in result if not d.resolved]

        return result

    def resolve_discrepancy(
        self,
        discrepancy: Discrepancy,
        notes: str = "",
    ) -> None:
        """Mark a discrepancy as resolved."""
        discrepancy.resolved = True
        discrepancy.resolved_at = datetime.utcnow()
        discrepancy.resolution_notes = notes

    def get_reconciliation_history(
        self,
        count: int = 10
    ) -> list[ReconciliationResult]:
        """Get recent reconciliation results."""
        return self._reconciliation_history[-count:]
