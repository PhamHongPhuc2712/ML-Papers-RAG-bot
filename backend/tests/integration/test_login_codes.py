"""Sign-in and sign-up by emailed code (sign-up plan, A1): one flow, rules fixed in the plan."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from copilot.app import create_app
from copilot.db.session import assert_safe_test_database
from copilot.identity.mail import MailError, RecordingSender, SmtpSender

pytestmark = pytest.mark.integration
DOMAIN = "codes.example.org"
_CODE = re.compile(r"\b(\d{6})\b")


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


class FailingSender:
    def send(self, to: str, subject: str, body: str) -> None:
        raise MailError("smtp down")


@pytest.fixture
def codes(test_settings, migrated_database, test_qdrant):
    """An app with a recording mail sender and a settable clock; its accounts and codes are
    deleted afterwards."""

    clock = Clock()
    sender = RecordingSender()

    def make(mail=None) -> TestClient:
        app = create_app(
            test_settings,
            {
                "db_engine": migrated_database,
                "qdrant_client": test_qdrant,
                "mail_sender": sender if mail is None else mail,
                "identity_clock": clock,
            },
        )
        return TestClient(app)

    yield make, sender, clock
    assert_safe_test_database(test_settings)
    with migrated_database.begin() as connection:
        connection.execute(
            text("delete from login_codes where email like :pattern"), {"pattern": f"%@{DOMAIN}"}
        )
        connection.execute(
            text("delete from users where email like :pattern"), {"pattern": f"%@{DOMAIN}"}
        )


def _email() -> str:
    return f"code-{uuid4().hex[:12]}@{DOMAIN}"


def _code_in(body: str) -> str:
    match = _CODE.search(body)
    assert match, body
    return match.group(1)


def _ask(client: TestClient, email: str):
    return client.post("/v1/auth/code", json={"email": email})


def _sign_in(client: TestClient, email: str, code: str):
    return client.post("/v1/auth/token", json={"email": email, "code": code})


def test_a_new_email_signs_up_with_its_first_correct_code(codes, migrated_database):
    make, sender, _ = codes
    email = _email()
    with make() as client:
        asked = _ask(client, email.upper())
        assert asked.status_code == 202 and asked.json()["resend_after"] == 60
        (mail,) = sender.sent
        assert mail.to == email and "code" in mail.subject.lower()
        signed = _sign_in(client, email, _code_in(mail.body))
        assert signed.status_code == 200
        token = signed.json()["access_token"]
        me = client.get("/v1/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200 and me.json()["email"] == email
    with migrated_database.connect() as connection:
        row = connection.execute(
            text("select password_hash from users where email = :email"), {"email": email}
        ).one()
    assert row.password_hash is None


def test_an_existing_account_signs_in_as_itself(codes, make_user):
    make, sender, _ = codes
    email = _email()
    user_id, _ = make_user(email=email)
    with make() as client:
        _ask(client, email)
        signed = _sign_in(client, email, _code_in(sender.sent[-1].body))
        me = client.get(
            "/v1/me", headers={"Authorization": f"Bearer {signed.json()['access_token']}"}
        )
    assert me.json()["user_id"] == str(user_id)


def test_the_answer_does_not_say_whether_an_email_has_an_account(codes, make_user):
    make, sender, _ = codes
    known = _email()
    make_user(email=known)
    with make() as client:
        answers = [_ask(client, address) for address in (known, _email())]
    assert [a.status_code for a in answers] == [202, 202]
    assert answers[0].json().keys() == answers[1].json().keys()
    assert answers[0].json()["resend_after"] == answers[1].json()["resend_after"]
    assert len(sender.sent) == 2


def test_a_code_works_once(codes):
    make, sender, _ = codes
    email = _email()
    with make() as client:
        _ask(client, email)
        code = _code_in(sender.sent[-1].body)
        assert _sign_in(client, email, code).status_code == 200
        again = _sign_in(client, email, code)
    assert again.status_code == 401 and again.json()["error"]["code"] == "code_expired"


def test_a_code_expires_after_ten_minutes(codes):
    make, sender, clock = codes
    email = _email()
    with make() as client:
        _ask(client, email)
        code = _code_in(sender.sent[-1].body)
        clock.advance(minutes=10, seconds=1)
        late = _sign_in(client, email, code)
    assert late.status_code == 401 and late.json()["error"]["code"] == "code_expired"


def test_the_sixth_try_finds_the_code_dead(codes):
    make, sender, _ = codes
    email = _email()
    with make() as client:
        _ask(client, email)
        code = _code_in(sender.sent[-1].body)
        wrong = f"{(int(code) + 1) % 1_000_000:06d}"
        tries = [_sign_in(client, email, wrong) for _ in range(5)]
        assert [t.json()["error"]["code"] for t in tries] == ["invalid_code"] * 4 + [
            "code_exhausted"
        ]
        right = _sign_in(client, email, code)
    assert right.status_code == 401 and right.json()["error"]["code"] == "code_exhausted"


def test_a_new_code_ends_the_old_one(codes):
    make, sender, clock = codes
    email = _email()
    with make() as client:
        _ask(client, email)
        first = _code_in(sender.sent[-1].body)
        clock.advance(seconds=61)
        _ask(client, email)
        second = _code_in(sender.sent[-1].body)
        old = _sign_in(client, email, first) if first != second else None
        new = _sign_in(client, email, second)
    assert old is None or old.status_code == 401
    assert new.status_code == 200


def test_asking_again_within_a_minute_is_refused_with_its_wait(codes):
    make, sender, clock = codes
    email = _email()
    with make() as client:
        _ask(client, email)
        clock.advance(seconds=20)
        soon = _ask(client, email)
    assert soon.status_code == 429
    error = soon.json()["error"]
    assert error["code"] == "code_rate_limited" and error["retry_after"] == 40
    assert len(sender.sent) == 1


def test_a_used_code_does_not_make_the_next_one_wait(codes):
    # Found by A6's walk: Settings asked for a delete code 24 s after sign-in, was refused, and
    # pointed at the sign-in code, already used. The wait is for a code still waiting to be used.
    make, sender, clock = codes
    email = _email()
    with make() as client:
        _ask(client, email)
        assert _sign_in(client, email, _code_in(sender.sent[-1].body)).status_code == 200
        clock.advance(seconds=20)
        again = _ask(client, email)
        clock.advance(seconds=5)
        unused = _ask(client, email)
    assert again.status_code == 202 and len(sender.sent) == 2
    assert unused.status_code == 429 and unused.json()["error"]["retry_after"] == 55


def test_the_sixth_code_in_an_hour_is_silently_not_sent(codes):
    make, sender, clock = codes
    email = _email()
    with make() as client:
        answers = []
        for _ in range(6):
            answers.append(_ask(client, email))
            clock.advance(seconds=61)
    assert [a.status_code for a in answers] == [202] * 6
    assert len(sender.sent) == 5


def test_a_disabled_account_gets_no_mail_and_cannot_sign_in(
    codes, make_user, migrated_database
):
    make, sender, _ = codes
    email = _email()
    make_user(email=email)
    with make() as client:
        _ask(client, email)
        code = _code_in(sender.sent[-1].body)
        with migrated_database.begin() as connection:
            connection.execute(
                text("update users set disabled_at = now() where email = :email"),
                {"email": email},
            )
        refused = _sign_in(client, email, code)
        asked = _ask(client, email)
    assert refused.status_code == 401
    assert asked.status_code == 202 and len(sender.sent) == 1


def test_an_unsendable_code_is_a_retryable_503_and_leaves_no_live_code(codes, migrated_database):
    make, _, _ = codes
    email = _email()
    with make(mail=FailingSender()) as client:
        failed = _ask(client, email)
    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "mail_unavailable"
    assert failed.json()["error"]["retryable"] is True
    with migrated_database.connect() as connection:
        live = connection.execute(
            text(
                "select count(*) from login_codes where email = :email and consumed_at is null"
            ),
            {"email": email},
        ).scalar_one()
    assert live == 0


def test_a_malformed_email_or_code_is_a_422(codes):
    make, _, _ = codes
    with make() as client:
        assert _ask(client, "not-an-email").status_code == 422
        assert _sign_in(client, _email(), "12ab56").status_code == 422


def test_the_code_is_never_stored_or_logged(codes, migrated_database, caplog):
    make, sender, _ = codes
    email = _email()
    with caplog.at_level(logging.DEBUG), make() as client:
        _ask(client, email)
        code = _code_in(sender.sent[-1].body)
        _sign_in(client, email, code)
    assert code not in caplog.text
    with migrated_database.connect() as connection:
        rows = connection.execute(
            text("select * from login_codes where email = :email"), {"email": email}
        ).mappings()
        stored = json.dumps([dict(row) for row in rows], default=str)
    assert code not in stored


def test_a_code_reaches_mailpit(test_settings):
    """The SMTP sender against the test profile's Mailpit (compose service ``mailpit-test``)."""

    smtp_port = int(os.environ.get("TEST_MAILPIT_SMTP_PORT", "51025"))
    api = os.environ.get("TEST_MAILPIT_URL", "http://127.0.0.1:58025")
    email = _email()
    subject = f"Your sign-in code {uuid4().hex[:8]}"
    SmtpSender("127.0.0.1", smtp_port, "ML Research Copilot <no-reply@localhost>").send(
        email, subject, "Your code is 123456."
    )
    with urllib.request.urlopen(f"{api}/api/v1/search?query=to:{email}", timeout=10) as reply:
        found = json.load(reply)
    assert [m["Subject"] for m in found["messages"]] == [subject]
