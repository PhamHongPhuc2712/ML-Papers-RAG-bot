"""Argon2id password hashes (spec §10; RFC 9106's low-memory profile, argon2-cffi's default)."""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 1024

_HASHER = PasswordHasher()
# Verifying against a real hash when an email is unknown makes a failed login take as long
# as a wrong password, so response time does not reveal which emails have accounts.
_DUMMY = _HASHER.hash("unknown-account-placeholder")


def hash_password(password: str) -> str:
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        raise ValueError("password_length")
    return _HASHER.hash(password)


def verify_password(stored: str | None, password: str) -> bool:
    try:
        matched = _HASHER.verify(stored or _DUMMY, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    return matched and stored is not None


def needs_rehash(stored: str) -> bool:
    return _HASHER.check_needs_rehash(stored)
