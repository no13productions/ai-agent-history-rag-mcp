"""Tests for production Google credential resolution."""

from datetime import UTC, datetime

import pytest
from google.auth import credentials
from google.auth.compute_engine.credentials import Credentials as ComputeCredentials
from google.auth.exceptions import DefaultCredentialsError

from claude_history_rag import gcp_auth


class ImpersonatedCredentialsStub(credentials.Credentials):
    """Minimal credential carrying the public identity metadata policy needs."""

    def __init__(self, service_account_email: str):
        super().__init__()
        self.service_account_email = service_account_email

    def refresh(self, request) -> None:
        del request
        self.token = "short-lived-token"
        self.expiry = datetime(2030, 1, 1, tzinfo=UTC)


class AttachedCredentialsStub(ComputeCredentials):
    """Metadata-backed credential that reveals its exact identity on refresh."""

    def __init__(self, resolved_email: str):
        super().__init__()
        self.resolved_email = resolved_email
        self.refresh_count = 0

    def refresh(self, request) -> None:
        del request
        self.refresh_count += 1
        self._service_account_email = self.resolved_email
        self.token = "short-lived-token"
        self.expiry = datetime(2030, 1, 1, tzinfo=UTC)


def test_production_adc_never_falls_back_to_active_gcloud_user(monkeypatch):
    """A failed production ADC lookup must not inherit the operator's broad CLI identity."""
    monkeypatch.setattr(
        gcp_auth.google.auth,
        "default",
        lambda **_kwargs: (_ for _ in ()).throw(DefaultCredentialsError("missing")),
    )
    gcloud_calls: list[list[str]] = []
    monkeypatch.setattr(
        gcp_auth.subprocess,
        "run",
        lambda command, **_kwargs: gcloud_calls.append(command),
    )

    with pytest.raises(RuntimeError, match="Production Application Default Credentials"):
        gcp_auth.default_project_and_credentials(
            ["scope"],
            production=True,
            credential_profile="impersonated_service_account",
            expected_service_account="history-rag-runtime@example.iam.gserviceaccount.com",
        )

    assert gcloud_calls == []


def test_production_impersonated_adc_requires_exact_target(monkeypatch):
    """Resolved short-lived credentials must name the admitted application identity."""
    resolved = ImpersonatedCredentialsStub("other@example.iam.gserviceaccount.com")
    monkeypatch.setattr(gcp_auth.google.auth, "default", lambda **_kwargs: (resolved, "p"))
    monkeypatch.setattr(
        gcp_auth,
        "_is_impersonated_credentials",
        lambda candidate: candidate is resolved,
    )

    with pytest.raises(RuntimeError, match="other@example.*expected .*history-rag-runtime"):
        gcp_auth.default_project_and_credentials(
            ["scope"],
            production=True,
            credential_profile="impersonated_service_account",
            expected_service_account="history-rag-runtime@example.iam.gserviceaccount.com",
        )


def test_production_impersonated_adc_accepts_exact_target(monkeypatch):
    expected = "history-rag-runtime@example.iam.gserviceaccount.com"
    resolved = ImpersonatedCredentialsStub(expected)
    monkeypatch.setattr(gcp_auth.google.auth, "default", lambda **_kwargs: (resolved, "p"))
    monkeypatch.setattr(
        gcp_auth,
        "_is_impersonated_credentials",
        lambda candidate: candidate is resolved,
    )

    project, credentials = gcp_auth.default_project_and_credentials(
        ["scope"],
        production=True,
        credential_profile="impersonated_service_account",
        expected_service_account=expected,
    )

    assert project == "p"
    assert credentials is resolved


def test_production_impersonated_adc_rejects_unverified_credential_class(monkeypatch):
    expected = "history-rag-runtime@example.iam.gserviceaccount.com"
    resolved = ImpersonatedCredentialsStub(expected)
    monkeypatch.setattr(gcp_auth.google.auth, "default", lambda **_kwargs: (resolved, "p"))

    with pytest.raises(RuntimeError, match="did not resolve impersonated credentials"):
        gcp_auth.default_project_and_credentials(
            ["scope"],
            production=True,
            credential_profile="impersonated_service_account",
            expected_service_account=expected,
        )


def test_production_attached_adc_refreshes_and_binds_exact_identity(monkeypatch):
    expected = "history-rag-runtime@example.iam.gserviceaccount.com"
    resolved = AttachedCredentialsStub(expected)
    monkeypatch.setattr(gcp_auth.google.auth, "default", lambda **_kwargs: (resolved, "p"))

    project, candidate = gcp_auth.default_project_and_credentials(
        ["scope"],
        production=True,
        credential_profile="attached_service_account",
        expected_service_account=expected,
    )

    assert project == "p"
    assert candidate is resolved
    assert resolved.refresh_count == 1
    assert resolved.service_account_email == expected


def test_production_attached_adc_rejects_refreshed_identity_mismatch(monkeypatch):
    resolved = AttachedCredentialsStub("other@example.iam.gserviceaccount.com")
    monkeypatch.setattr(gcp_auth.google.auth, "default", lambda **_kwargs: (resolved, "p"))

    with pytest.raises(RuntimeError, match="other@example.*expected .*history-rag-runtime"):
        gcp_auth.default_project_and_credentials(
            ["scope"],
            production=True,
            credential_profile="attached_service_account",
            expected_service_account="history-rag-runtime@example.iam.gserviceaccount.com",
        )
