"""Sending mail (sign-up plan, A1): a Protocol, an SMTP sender and a recording fake.

On this machine SMTP is Mailpit, a mail catcher in Docker (compose service ``mailpit``):
codes land in its local inbox and nothing leaves the laptop (the developer's decision,
2026-10-21). Real delivery is a different SMTP host, set in the deployment plan.
"""

from __future__ import annotations

import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Protocol


class MailError(Exception):
    """The message could not be handed to the mail server."""


class MailSender(Protocol):
    def send(self, to: str, subject: str, body: str) -> None: ...


class SmtpSender:
    def __init__(self, host: str, port: int, sender: str, *, timeout: float = 10.0) -> None:
        self._host = host
        self._port = port
        self._sender = sender
        self._timeout = timeout

    def send(self, to: str, subject: str, body: str) -> None:
        message = EmailMessage()
        message["From"] = self._sender
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)
        try:
            with smtplib.SMTP(self._host, self._port, timeout=self._timeout) as smtp:
                smtp.send_message(message)
        except (OSError, smtplib.SMTPException) as error:
            raise MailError(type(error).__name__) from error


@dataclass(frozen=True)
class SentMail:
    to: str
    subject: str
    body: str


@dataclass
class RecordingSender:
    """Keeps every message instead of sending it; for tests."""

    sent: list[SentMail] = field(default_factory=list)

    def send(self, to: str, subject: str, body: str) -> None:
        self.sent.append(SentMail(to, subject, body))
