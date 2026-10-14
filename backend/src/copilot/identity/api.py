"""Login and the caller's identity (spec §10). Identity comes only from a verified bearer token."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import Field

from ..contracts import ContractModel, Principal
from ..search.api import ApiError, request_id
from .tokens import TokenError, TokenIssuer
from .users import UserStore

_BEARER = HTTPBearer(auto_error=False, description="An access token from POST /v1/auth/token")
_SIGN_IN_AGAIN = "your session is not valid; sign in again"


@dataclass(frozen=True)
class Identity:
    users: UserStore
    tokens: TokenIssuer


class TokenRequest(ContractModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)


class TokenResponse(ContractModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    request_id: UUID


class MeResponse(ContractModel):
    user_id: UUID
    email: str
    request_id: UUID


class DeleteAccountRequest(ContractModel):
    password: str = Field(min_length=1, max_length=1024)


def _identity(request: Request) -> Identity:
    identity = getattr(request.app.state, "identity", None)
    if not isinstance(identity, Identity):
        raise ApiError(503, "identity_unavailable", "accounts are not configured", retryable=True)
    return identity


def current_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)],
) -> Principal:
    if credentials is None or not credentials.credentials:
        raise ApiError(401, "authentication_required", "sign in to continue")
    identity = _identity(request)
    try:
        user_id = identity.tokens.subject(credentials.credentials)
    except TokenError as error:
        raise ApiError(401, error.code, _SIGN_IN_AGAIN) from error
    if not identity.users.active(user_id):
        raise ApiError(401, "token_invalid", _SIGN_IN_AGAIN)
    return Principal(user_id=user_id)


CurrentPrincipal = Annotated[Principal, Depends(current_principal)]


def router() -> APIRouter:
    api = APIRouter()

    @api.post("/v1/auth/token", response_model=TokenResponse)
    def token(body: TokenRequest, request: Request) -> TokenResponse:
        identity = _identity(request)
        user_id = identity.users.authenticate(body.email, body.password)
        if user_id is None:
            raise ApiError(401, "invalid_credentials", "email or password is incorrect")
        issued = identity.tokens.issue(user_id)
        return TokenResponse(
            access_token=issued.token, expires_in=issued.expires_in, request_id=request_id(request)
        )

    @api.get("/v1/me", response_model=MeResponse)
    def me(principal: CurrentPrincipal, request: Request) -> MeResponse:
        email = _identity(request).users.email(principal.user_id)
        if email is None:
            raise ApiError(401, "token_invalid", _SIGN_IN_AGAIN)
        return MeResponse(user_id=principal.user_id, email=email, request_id=request_id(request))

    @api.delete("/v1/me", status_code=204, response_class=Response)
    def delete_me(
        body: DeleteAccountRequest, principal: CurrentPrincipal, request: Request
    ) -> Response:
        users = _identity(request).users
        if not users.check_password(principal.user_id, body.password):
            raise ApiError(401, "invalid_credentials", "the password is incorrect")
        users.delete(principal.user_id)
        return Response(status_code=204)

    return api
