"""Application-issued access tokens (spec §10): HS256 JWTs under a key derived from SECRET_KEY.

The token names its subject and nothing else; whether that account may still act is checked
against the database on every request, so disabling an account ends its sessions at once.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID, uuid4

import jwt

ISSUER = "ml-research-copilot"
AUDIENCE = "copilot-api"
ALGORITHM = "HS256"


class TokenError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def token_key(secret_key: str) -> bytes:
    """A key used only for access tokens, so the application secret is never used raw."""

    return hmac.new(secret_key.encode(), b"copilot:access-token:v1", hashlib.sha256).digest()


@dataclass(frozen=True)
class IssuedToken:
    token: str
    expires_in: int


class TokenIssuer:
    def __init__(
        self, secret_key: str, ttl_seconds: int, clock: Callable[[], float] = time.time
    ) -> None:
        self._key = token_key(secret_key)
        self._ttl = ttl_seconds
        self._clock = clock

    def issue(self, user_id: UUID) -> IssuedToken:
        now = int(self._clock())
        claims = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": str(user_id),
            "iat": now,
            "exp": now + self._ttl,
            "jti": uuid4().hex,
        }
        return IssuedToken(jwt.encode(claims, self._key, algorithm=ALGORITHM), self._ttl)

    def subject(self, token: str) -> UUID:
        try:
            claims = jwt.decode(
                token,
                self._key,
                algorithms=[ALGORITHM],
                audience=AUDIENCE,
                issuer=ISSUER,
                options={"require": ["exp", "iat", "sub", "iss", "aud"]},
            )
        except jwt.ExpiredSignatureError as error:
            raise TokenError("token_expired") from error
        except jwt.InvalidTokenError as error:
            raise TokenError("token_invalid") from error
        try:
            return UUID(str(claims["sub"]))
        except ValueError as error:
            raise TokenError("token_invalid") from error
