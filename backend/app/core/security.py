"""Server-side authentication, role checks and case access policy.

The caller's identity and permission scope are derived exclusively from the
credential presented to the server.  Nothing in a request body can widen it:
``AccessPolicy.scope_filter()`` is merged into every lexical, vector, image and
structured query *before* retrieval, and ``assert_case_allowed`` guards every
direct record lookup (cases, reports, images, citations, experiment artifacts).
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from enum import StrEnum

from app.core.config import Settings
from app.core.errors import AuthenticationError, AuthorizationError


class Role(StrEnum):
    RESEARCHER = "researcher"
    ANNOTATOR = "annotator"
    ADMIN = "admin"


_ROLE_RANK = {Role.ANNOTATOR: 1, Role.RESEARCHER: 2, Role.ADMIN: 3}


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: Role
    scopes: frozenset[str] = field(default_factory=frozenset)

    def has_role(self, minimum: Role) -> bool:
        return _ROLE_RANK[self.role] >= _ROLE_RANK[minimum]


@dataclass(frozen=True)
class AccessPolicy:
    """The set of ``access_scope`` values a principal may read."""

    principal: Principal

    @property
    def allowed_scopes(self) -> list[str]:
        return sorted(self.principal.scopes)

    def allows(self, access_scope: str | None) -> bool:
        return access_scope is not None and access_scope in self.principal.scopes

    def assert_case_allowed(self, access_scope: str | None, case_id: str) -> None:
        if not self.allows(access_scope):
            # Same error as "missing" would leak less, but a 403 on an opaque id is the
            # documented behaviour; the message never contains clinical content.
            raise AuthorizationError("access to this case is not permitted", {"case_id": case_id})

    def scope_filter(self) -> dict[str, list[str]]:
        return {"access_scope": self.allowed_scopes}


def authenticate(api_key: str | None, settings: Settings) -> Principal:
    if not api_key:
        raise AuthenticationError("missing API key (send it in the X-API-Key header)")
    for known, spec in settings.api_keys().items():
        if hmac.compare_digest(known, api_key):
            return Principal(
                user_id=str(spec.get("user", "unknown")),
                role=Role(spec.get("role", "researcher")),
                scopes=frozenset(spec.get("scopes", [])),
            )
    if (
        settings.environment != "production"
        and settings.dev_api_key
        and hmac.compare_digest(settings.dev_api_key, api_key)
    ):
        return Principal(user_id="dev", role=Role.ADMIN, scopes=frozenset({"research"}))
    raise AuthenticationError("invalid API key")


def require_role(principal: Principal, minimum: Role) -> None:
    if not principal.has_role(minimum):
        raise AuthorizationError(f"role '{minimum.value}' or higher is required")


def system_principal(scopes: list[str]) -> Principal:
    """Principal used by offline CLI jobs (ingestion, evaluation)."""
    return Principal(user_id="system", role=Role.ADMIN, scopes=frozenset(scopes))
