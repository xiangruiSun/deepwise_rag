"""Application error hierarchy mapped to structured HTTP error responses."""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthenticated"


class AuthorizationError(AppError):
    status_code = 403
    code = "forbidden"


class ValidationFailure(AppError):
    status_code = 422
    code = "validation_error"


class ConfigurationError(AppError):
    status_code = 500
    code = "configuration_error"


class DependencyUnavailable(AppError):
    status_code = 503
    code = "dependency_unavailable"


class RequestTimeout(AppError):
    status_code = 504
    code = "timeout"


class DatasetSchemaError(AppError):
    """Raised when a dataset layout or a required field cannot be resolved.

    Ingestion must fail loudly rather than guess the meaning of a column."""

    status_code = 422
    code = "dataset_schema_error"


class IndexCompatibilityError(AppError):
    status_code = 409
    code = "index_incompatible"
