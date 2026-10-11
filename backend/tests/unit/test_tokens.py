"""Access tokens and passwords: what is issued verifies, and nothing else does."""

from __future__ import annotations

import time
from uuid import uuid4

import jwt
import pytest

from copilot.identity.passwords import hash_password, verify_password
from copilot.identity.tokens import AUDIENCE, ISSUER, TokenError, TokenIssuer, token_key

SECRET = "test-only-secret"


def test_an_issued_token_verifies_to_its_subject():
    issuer = TokenIssuer(SECRET, ttl_seconds=600)
    user = uuid4()
    issued = issuer.issue(user)
    assert issued.expires_in == 600
    assert issuer.subject(issued.token) == user


def test_an_expired_token_is_refused_as_expired():
    token = TokenIssuer(SECRET, ttl_seconds=60, clock=lambda: time.time() - 3600).issue(uuid4())
    with pytest.raises(TokenError) as caught:
        TokenIssuer(SECRET, ttl_seconds=60).subject(token.token)
    assert caught.value.code == "token_expired"


@pytest.mark.parametrize(
    "change", [{"aud": "someone-else"}, {"iss": "elsewhere"}, {"sub": "not-a-uuid"}]
)
def test_a_wrong_audience_issuer_or_subject_is_invalid(change):
    now = int(time.time())
    claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": str(uuid4()), "iat": now, "exp": now + 60}
    token = jwt.encode({**claims, **change}, token_key(SECRET), algorithm="HS256")
    with pytest.raises(TokenError) as caught:
        TokenIssuer(SECRET, ttl_seconds=60).subject(token)
    assert caught.value.code == "token_invalid"


def test_another_key_no_algorithm_or_garbage_is_invalid():
    user = uuid4()
    now = int(time.time())
    forged = TokenIssuer("another-secret", ttl_seconds=60).issue(user).token
    unsigned = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": str(user), "iat": now, "exp": now + 60},
        None,
        algorithm="none",
    )
    for token in (forged, unsigned, "not.a.token", ""):
        with pytest.raises(TokenError) as caught:
            TokenIssuer(SECRET, ttl_seconds=60).subject(token)
        assert caught.value.code == "token_invalid"


def test_the_signing_key_is_derived_never_the_raw_secret():
    key = token_key(SECRET)
    assert key != SECRET.encode() and len(key) == 32


def test_passwords_hash_with_argon2id_and_verify():
    stored = hash_password("correct horse battery staple")
    assert stored.startswith("$argon2id$")
    assert verify_password(stored, "correct horse battery staple")
    assert not verify_password(stored, "wrong horse battery staple")
    # An unknown account verifies against a dummy hash and still fails.
    assert not verify_password(None, "correct horse battery staple")


def test_short_passwords_are_refused():
    with pytest.raises(ValueError, match="password_length"):
        hash_password("short")
