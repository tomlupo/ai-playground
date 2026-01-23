"""Tests for schema definitions."""

from datetime import date, datetime

import pyarrow as pa
import pytest

from src.schemas import TradeSchema, PriceSchema, PositionSchema
from src.schemas.base import BaseSchema, SchemaField


class TestSchemaField:
    """Tests for SchemaField."""

    def test_to_arrow_field(self):
        """Test converting to PyArrow field."""
        field = SchemaField(
            name="test_col",
            dtype=pa.string(),
            nullable=False,
            description="Test column"
        )

        arrow_field = field.to_arrow_field()

        assert arrow_field.name == "test_col"
        assert arrow_field.type == pa.string()
        assert not arrow_field.nullable


class TestBaseSchema:
    """Tests for BaseSchema."""

    def test_fields(self):
        """Test getting schema fields."""
        fields = TradeSchema.fields()
        field_names = [f.name for f in fields]

        assert "trade_id" in field_names
        assert "symbol" in field_names
        assert "price" in field_names

    def test_to_arrow_schema(self):
        """Test converting to PyArrow schema."""
        schema = TradeSchema.to_arrow_schema()

        assert isinstance(schema, pa.Schema)
        assert "trade_id" in schema.names
        assert "symbol" in schema.names

    def test_field_names(self):
        """Test getting field names."""
        names = TradeSchema.field_names()

        assert "trade_id" in names
        assert "symbol" in names
        assert "price" in names

    def test_empty_table(self):
        """Test creating empty table with schema."""
        table = TradeSchema.empty_table()

        assert len(table) == 0
        assert "trade_id" in table.column_names

    def test_from_dicts(self):
        """Test creating table from dictionaries."""
        records = [
            {
                "trade_id": "T001",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100.0,
                "price": 150.0,
                "trade_time": datetime.now(),
                "date": date.today(),
            },
            {
                "trade_id": "T002",
                "symbol": "GOOG",
                "side": "sell",
                "quantity": 50.0,
                "price": 2800.0,
                "trade_time": datetime.now(),
                "date": date.today(),
            },
        ]

        table = TradeSchema.from_dicts(records)

        assert len(table) == 2
        assert table["trade_id"][0].as_py() == "T001"


class TestTradeSchema:
    """Tests for TradeSchema."""

    def test_validate_valid_data(self):
        """Test validation with valid data."""
        data = pa.table({
            "trade_id": ["T001"],
            "symbol": ["AAPL"],
            "side": ["buy"],
            "quantity": [100.0],
            "price": [150.0],
            "trade_time": [datetime.now()],
            "date": [date.today()],
        })

        errors = TradeSchema.validate(data)
        assert len(errors) == 0

    def test_validate_missing_required(self):
        """Test validation with missing required columns."""
        data = pa.table({
            "symbol": ["AAPL"],
            "price": [150.0],
        })

        errors = TradeSchema.validate(data)
        assert len(errors) > 0
        assert any("trade_id" in e for e in errors)


class TestPriceSchema:
    """Tests for PriceSchema."""

    def test_schema_fields(self):
        """Test price schema has expected fields."""
        names = PriceSchema.field_names()

        assert "symbol" in names
        assert "open" in names
        assert "high" in names
        assert "low" in names
        assert "close" in names
        assert "volume" in names


class TestPositionSchema:
    """Tests for PositionSchema."""

    def test_schema_fields(self):
        """Test position schema has expected fields."""
        names = PositionSchema.field_names()

        assert "symbol" in names
        assert "quantity" in names
        assert "avg_cost" in names
        assert "market_value" in names
        assert "unrealized_pnl" in names
