"""Base schema definitions and utilities."""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional, Type, get_type_hints
import pyarrow as pa


# Type mapping from Python types to PyArrow types
PYTHON_TO_ARROW = {
    int: pa.int64(),
    float: pa.float64(),
    str: pa.string(),
    bool: pa.bool_(),
    date: pa.date32(),
    datetime: pa.timestamp("us"),
    Decimal: pa.decimal128(18, 8),
    bytes: pa.binary(),
}


def get_arrow_type(python_type: Type) -> pa.DataType:
    """Convert Python type to PyArrow type."""
    # Handle Optional types
    origin = getattr(python_type, "__origin__", None)
    if origin is type(None):
        return pa.null()

    # Handle Optional[X] (Union[X, None])
    if origin is type(Optional):
        args = getattr(python_type, "__args__", ())
        if args:
            return get_arrow_type(args[0])

    # Handle basic types
    if python_type in PYTHON_TO_ARROW:
        return PYTHON_TO_ARROW[python_type]

    # Handle list types
    if origin is list:
        args = getattr(python_type, "__args__", ())
        if args:
            return pa.list_(get_arrow_type(args[0]))
        return pa.list_(pa.string())

    # Default to string
    return pa.string()


@dataclass
class SchemaField:
    """Definition of a schema field."""

    name: str
    dtype: pa.DataType
    nullable: bool = True
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def to_arrow_field(self) -> pa.Field:
        """Convert to PyArrow field."""
        return pa.field(
            self.name,
            self.dtype,
            nullable=self.nullable,
            metadata={b"description": self.description.encode()} if self.description else None,
        )


class BaseSchema:
    """
    Base class for schema definitions.

    Subclasses define fields as class attributes using SchemaField.
    """

    @classmethod
    def fields(cls) -> list[SchemaField]:
        """Get all schema fields."""
        return [
            getattr(cls, name)
            for name in dir(cls)
            if isinstance(getattr(cls, name), SchemaField)
        ]

    @classmethod
    def to_arrow_schema(cls) -> pa.Schema:
        """Convert to PyArrow schema."""
        return pa.schema([f.to_arrow_field() for f in cls.fields()])

    @classmethod
    def field_names(cls) -> list[str]:
        """Get list of field names."""
        return [f.name for f in cls.fields()]

    @classmethod
    def validate(cls, table: pa.Table) -> list[str]:
        """
        Validate a PyArrow table against the schema.

        Returns list of validation errors (empty if valid).
        """
        errors = []
        schema = cls.to_arrow_schema()

        # Check for missing required columns
        for field in cls.fields():
            if field.name not in table.column_names:
                if not field.nullable:
                    errors.append(f"Missing required column: {field.name}")
            else:
                # Check type compatibility
                actual_type = table.schema.field(field.name).type
                if not pa.types.is_nested(field.dtype):
                    if not actual_type.equals(field.dtype):
                        # Allow some type coercion
                        if not cls._types_compatible(actual_type, field.dtype):
                            errors.append(
                                f"Type mismatch for {field.name}: "
                                f"expected {field.dtype}, got {actual_type}"
                            )

        return errors

    @classmethod
    def _types_compatible(cls, actual: pa.DataType, expected: pa.DataType) -> bool:
        """Check if types are compatible for coercion."""
        # Allow int to float
        if pa.types.is_integer(actual) and pa.types.is_floating(expected):
            return True
        # Allow smaller ints to larger ints
        if pa.types.is_integer(actual) and pa.types.is_integer(expected):
            return True
        # Allow date/timestamp interop
        if (pa.types.is_date(actual) or pa.types.is_timestamp(actual)) and \
           (pa.types.is_date(expected) or pa.types.is_timestamp(expected)):
            return True
        return False

    @classmethod
    def cast(cls, table: pa.Table) -> pa.Table:
        """Cast table columns to match schema types."""
        schema = cls.to_arrow_schema()
        arrays = []

        for field in schema:
            if field.name in table.column_names:
                col = table.column(field.name)
                if not col.type.equals(field.type):
                    col = col.cast(field.type)
                arrays.append(col)
            elif field.nullable:
                # Add null column for missing nullable fields
                arrays.append(pa.nulls(len(table), type=field.type))
            else:
                raise ValueError(f"Missing required column: {field.name}")

        return pa.table(dict(zip(schema.names, arrays)))

    @classmethod
    def empty_table(cls) -> pa.Table:
        """Create an empty table with the schema."""
        return pa.table({f.name: pa.array([], type=f.dtype) for f in cls.fields()})

    @classmethod
    def from_dicts(cls, records: list[dict]) -> pa.Table:
        """Create a table from a list of dictionaries."""
        if not records:
            return cls.empty_table()

        columns = {f.name: [] for f in cls.fields()}

        for record in records:
            for field in cls.fields():
                columns[field.name].append(record.get(field.name))

        arrays = {
            name: pa.array(values, type=cls._get_field(name).dtype)
            for name, values in columns.items()
        }

        return pa.table(arrays)

    @classmethod
    def _get_field(cls, name: str) -> SchemaField:
        """Get a field by name."""
        for field in cls.fields():
            if field.name == name:
                return field
        raise ValueError(f"Unknown field: {name}")
