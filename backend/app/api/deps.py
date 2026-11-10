"""FastAPI dependencies: application context, authentication, timeouts, pagination."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from fastapi import Depends, Header, Query, Request
from starlette.concurrency import run_in_threadpool

from app.core.container import AppContext
from app.core.errors import RequestTimeout
from app.core.security import AccessPolicy, Principal, Role, authenticate, require_role

T = TypeVar("T")


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx  # type: ignore[no-any-return]


Ctx = Annotated[AppContext, Depends(get_ctx)]


def get_principal(ctx: Ctx, x_api_key: Annotated[str | None, Header()] = None) -> Principal:
    # identity and scope come exclusively from the presented credential
    return authenticate(x_api_key, ctx.settings)


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


def get_policy(principal: CurrentPrincipal) -> AccessPolicy:
    return AccessPolicy(principal)


Policy = Annotated[AccessPolicy, Depends(get_policy)]


def researcher(principal: CurrentPrincipal) -> Principal:
    require_role(principal, Role.RESEARCHER)
    return principal


def admin(principal: CurrentPrincipal) -> Principal:
    require_role(principal, Role.ADMIN)
    return principal


Researcher = Annotated[Principal, Depends(researcher)]
Admin = Annotated[Principal, Depends(admin)]


class Page:
    def __init__(
        self,
        limit: Annotated[int, Query(ge=1, le=200)] = 25,
        offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    ) -> None:
        self.limit, self.offset = limit, offset


Pagination = Annotated[Page, Depends()]


async def run_bounded(ctx: AppContext, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run blocking work in the thread pool with the configured request timeout."""
    try:
        return await asyncio.wait_for(
            run_in_threadpool(func, *args, **kwargs), timeout=ctx.settings.request_timeout_seconds
        )
    except TimeoutError as exc:
        raise RequestTimeout("the request exceeded the configured time limit") from exc
