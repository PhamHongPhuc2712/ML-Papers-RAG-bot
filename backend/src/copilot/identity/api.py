"""Login and the caller's identity (spec §10). Identity comes only from a verified bearer token.

Since the sign-up plan (2026-10-21) a token is issued only for a code sent to the email by
``POST /v1/auth/code``; an email without an account gets one on its first correct code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import Field

from ..contracts import ContractModel, Principal
from ..search.api import ApiError, request_id
from .codes import RESEND_AFTER, CodeRateLimited, CodeStore, Verdict
from .mail import MailError, MailSender
from .tokens import TokenError, TokenIssuer
from .users import AccountError, UserStore, normalize_email

_BEARER = HTTPBearer(auto_error=False, description="An access token from POST /v1/auth/token")
_SIGN_IN_AGAIN = "your session is not valid; sign in again"


_CODE_SUBJECT = "Your ML Research Copilot sign-in code"
_CODE_BODY = (
    "Your sign-in code is {code}.\n\n"
    "It works once, for 10 minutes. If you did not ask for it, you can ignore this email.\n"
)
_CODE_MESSAGES = {
    Verdict.INVALID: "that code is not right; check it and try again",
    Verdict.EXHAUSTED: "too many wrong tries; ask for a new code",
    Verdict.EXPIRED: "that code has expired or was already used; ask for a new code",
}


@dataclass(frozen=True)
class Identity:
    users: UserStore
    tokens: TokenIssuer
    codes: CodeStore | None = None
    # None when no SMTP host is configured: asking for a code is then a typed 503.
    mail: MailSender | None = None


class CodeRequest(ContractModel):
    email: str = Field(min_length=3, max_length=320)


class CodeResponse(ContractModel):
    resend_after: int
    request_id: UUID


class TokenRequest(ContractModel):
    """An email with the code sent to it: the only way in since the sign-up plan (A2)."""

    email: str = Field(min_length=3, max_length=320)
    code: str = Field(pattern=r"^\d{6}$")


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

    @api.post("/v1/auth/code", status_code=202, response_model=CodeResponse)
    def code(body: CodeRequest, request: Request) -> CodeResponse:
        """Email a sign-in code. The answer is the same whether or not the email has an
        account, and whether or not the hourly limit let a code go out."""

        identity = _identity(request)
        email = _email_or_422(body.email)
        if identity.codes is None or identity.mail is None:
            raise ApiError(503, "mail_unavailable", "email is not configured", retryable=True)
        answer = CodeResponse(
            resend_after=int(RESEND_AFTER.total_seconds()), request_id=request_id(request)
        )
        if identity.users.disabled(email):
            return answer
        try:
            issued = identity.codes.issue(email)
        except CodeRateLimited as limited:
            raise ApiError(
                429,
                "code_rate_limited",
                f"wait {limited.retry_after} s before asking for another code",
                retryable=True,
                retry_after=limited.retry_after,
            ) from limited
        if issued.code is None or issued.code_id is None:
            return answer
        try:
            identity.mail.send(email, _CODE_SUBJECT, _CODE_BODY.format(code=issued.code))
        except MailError as error:
            identity.codes.discard(issued.code_id)
            raise ApiError(
                503, "mail_unavailable", "the code could not be sent; try again", retryable=True
            ) from error
        return answer

    @api.post("/v1/auth/token", response_model=TokenResponse)
    def token(body: TokenRequest, request: Request) -> TokenResponse:
        identity = _identity(request)
        user_id = _by_code(identity, _email_or_422(body.email), body.code)
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


def _email_or_422(email: str) -> str:
    try:
        return normalize_email(email)
    except AccountError as error:
        raise ApiError(422, "validation_error", "email: not a valid email address") from error


def _by_code(identity: Identity, email: str, code: str) -> UUID:
    if identity.codes is None:
        raise ApiError(503, "mail_unavailable", "email is not configured", retryable=True)
    verdict = identity.codes.verify(email, code)
    if verdict is not Verdict.ACCEPTED:
        raise ApiError(401, verdict.value, _CODE_MESSAGES[verdict])
    user_id = identity.users.for_code(email)
    if user_id is None:
        # A disabled account: the code was right, but there is nothing to sign in to.
        raise ApiError(401, Verdict.EXPIRED.value, _CODE_MESSAGES[Verdict.EXPIRED])
    return user_id
