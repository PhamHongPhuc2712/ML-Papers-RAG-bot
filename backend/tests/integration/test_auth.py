"""Accounts: login issues a token, the token is the only identity, disabling ends it."""

from __future__ import annotations

import io

import pytest

from copilot.cli import main
from copilot.identity.tokens import TokenIssuer
from copilot.identity.users import AccountError, UserStore

pytestmark = pytest.mark.integration
PASSWORD = "correct horse battery staple"


def test_login_issues_a_token_that_identifies_the_user(api_client, make_user):
    user_id, _ = make_user(email="Alice@Example.org")
    response = api_client.post(
        "/v1/auth/token", json={"email": "alice@example.org", "password": PASSWORD}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer" and body["expires_in"] == 43200 and body["request_id"]
    me = api_client.get("/v1/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json()["user_id"] == str(user_id) and me.json()["email"] == "alice@example.org"


def test_a_wrong_password_and_an_unknown_email_fail_identically(api_client, make_user):
    make_user(email="bob@example.org")
    wrong = api_client.post(
        "/v1/auth/token", json={"email": "bob@example.org", "password": "not the password!"}
    )
    unknown = api_client.post(
        "/v1/auth/token", json={"email": "nobody@example.org", "password": PASSWORD}
    )
    for response in (wrong, unknown):
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_credentials"
    assert wrong.json()["error"]["message"] == unknown.json()["error"]["message"]


def test_missing_malformed_and_forged_tokens_are_401(api_client, make_user):
    user_id, _ = make_user()
    forged = TokenIssuer("another-secret", 600).issue(user_id).token
    cases = [
        ({}, "authentication_required"),
        ({"Authorization": "Basic YWxpY2U6cHc="}, "authentication_required"),
        ({"Authorization": f"Bearer {forged}"}, "token_invalid"),
    ]
    for headers, code in cases:
        response = api_client.get("/v1/me", headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == code


def test_a_disabled_accounts_token_stops_working_at_once(
    api_client, make_user, auth_headers, migrated_database
):
    user_id, email = make_user()
    headers = auth_headers(user_id)
    assert api_client.get("/v1/me", headers=headers).status_code == 200
    assert UserStore(migrated_database).disable(email)
    response = api_client.get("/v1/me", headers=headers)
    assert response.status_code == 401 and response.json()["error"]["code"] == "token_invalid"
    login = api_client.post("/v1/auth/token", json={"email": email, "password": PASSWORD})
    assert login.status_code == 401


def test_emails_are_unique_ignoring_case(make_user, migrated_database):
    make_user(email="carol@example.org")
    with pytest.raises(AccountError) as caught:
        UserStore(migrated_database).create("CAROL@example.org", PASSWORD)
    assert caught.value.code == "email_taken"


def test_a_client_supplied_user_id_is_never_identity(api_client, make_user, auth_headers):
    me, _ = make_user()
    other, _ = make_user()
    response = api_client.get(
        "/v1/me", headers={**auth_headers(me), "X-User-Id": str(other)}, params={"user_id": other}
    )
    assert response.json()["user_id"] == str(me)
    login = api_client.post(
        "/v1/auth/token", json={"email": "x@example.org", "password": PASSWORD, "user_id": str(me)}
    )
    assert login.status_code == 422


def test_the_cli_creates_an_account_without_a_password(test_settings, migrated_database, capsys):
    """Accounts sign in by emailed code since the sign-up plan (A1); stdin is not read."""

    from sqlalchemy import text

    email = "erin-cli@example.org"
    argv = ["users", "create", "--email", email, "--database-url", test_settings.database_url]
    assert main(argv) == 0
    assert '"email": "erin-cli@example.org"' in capsys.readouterr().out
    with migrated_database.begin() as connection:
        stored = connection.execute(
            text("delete from users where email = :email returning password_hash"),
            {"email": email},
        ).scalar_one()
    assert stored is None


def test_the_cli_creates_an_account_reading_the_password_from_stdin(
    test_settings, migrated_database, monkeypatch, capsys
):
    email = "dana-cli@example.org"
    monkeypatch.setattr("sys.stdin", io.StringIO(PASSWORD + "\n"))
    argv = [
        "users",
        "create",
        "--email",
        email,
        "--password-stdin",
        "--database-url",
        test_settings.database_url,
    ]
    assert main(argv) == 0
    assert '"email": "dana-cli@example.org"' in capsys.readouterr().out
    store = UserStore(migrated_database)
    user_id = store.authenticate(email, PASSWORD)
    assert user_id is not None
    with migrated_database.begin() as connection:
        from sqlalchemy import text

        connection.execute(text("delete from users where id = :id"), {"id": user_id})
