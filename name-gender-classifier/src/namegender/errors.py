class DataError(Exception):
    """Base error for data loading and acquisition failures."""


class IntegrityError(DataError):
    """Raised when a cached file does not match its expected integrity metadata."""


class SchemaError(DataError):
    """Raised when the raw dataset violates the expected schema or value constraints."""
