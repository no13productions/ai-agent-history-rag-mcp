"""Google Cloud authentication helpers for local and ADC environments."""

import json
import subprocess
from datetime import UTC, datetime

import google.auth
from google.auth import credentials
from google.auth.exceptions import DefaultCredentialsError
from google.auth.impersonated_credentials import Credentials as ImpersonatedCredentials

GCLOUD_TOKEN_CMD = [
    "gcloud",
    "auth",
    "print-access-token",
    "--format=json",
]


def _gcloud_access_token() -> tuple[str, datetime | None]:
    """Fetch an access token from the active gcloud account."""
    result = subprocess.run(
        GCLOUD_TOKEN_CMD,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        payload = {"token": output}
    if isinstance(payload, str):
        payload = {"token": payload}
    token = payload.get("token")
    if not token:
        raise RuntimeError("gcloud did not return an access token")
    expiry = payload.get("token_expiry")
    expires_at = None
    if expiry:
        expires_at = datetime.fromisoformat(expiry.replace("Z", "+00:00")).astimezone(UTC)
    return token, expires_at


class GcloudCliCredentials(credentials.Credentials):
    """Refreshable credentials backed by the active gcloud CLI login."""

    def __init__(self, scopes: list[str] | tuple[str, ...]):
        super().__init__()
        self._scopes = tuple(scopes)
        self.expiry = None

    @property
    def scopes(self):
        """Configured OAuth scopes."""
        return self._scopes

    def refresh(self, request) -> None:
        """Refresh token by invoking gcloud."""
        del request
        self.token, self.expiry = _gcloud_access_token()

    def with_scopes(self, scopes, default_scopes=None):
        """Return credentials with the requested scopes."""
        del default_scopes
        return GcloudCliCredentials(scopes)


def _is_impersonated_credentials(candidate: credentials.Credentials) -> bool:
    """Return whether ADC resolved Google's public impersonated credential type."""
    return isinstance(candidate, ImpersonatedCredentials)


def _validate_production_credentials(
    candidate: credentials.Credentials,
    profile: str,
    expected_service_account: str,
) -> None:
    """Bind resolved ADC credentials to the declared keyless production profile."""
    if not expected_service_account:
        raise RuntimeError("Production credential identity must be explicit")
    if profile == "impersonated_service_account":
        if not _is_impersonated_credentials(candidate):
            raise RuntimeError(
                "Production ADC did not resolve impersonated credentials for the declared profile"
            )
        actual_identity = getattr(candidate, "service_account_email", "")
        if actual_identity != expected_service_account:
            raise RuntimeError(
                "Production ADC resolved service account "
                f"{actual_identity!r}; expected {expected_service_account!r}"
            )
        return
    if profile == "attached_service_account":
        from google.auth.compute_engine.credentials import Credentials as ComputeCredentials

        if not isinstance(candidate, ComputeCredentials):
            raise RuntimeError(
                "Production ADC did not resolve attached workload credentials for the "
                "declared profile"
            )
        actual_identity = getattr(candidate, "service_account_email", "")
        if actual_identity in {"", "default"}:
            from google.auth.transport.requests import Request

            candidate.refresh(Request())
            actual_identity = getattr(candidate, "service_account_email", "")
        if actual_identity != expected_service_account:
            raise RuntimeError(
                "Production ADC resolved service account "
                f"{actual_identity!r}; expected {expected_service_account!r}"
            )
        return
    raise RuntimeError(f"Unsupported production credential profile: {profile!r}")


def default_project_and_credentials(
    scopes: list[str] | tuple[str, ...],
    *,
    production: bool = False,
    credential_profile: str = "",
    expected_service_account: str = "",
):
    """Resolve project and credentials, with a development-only gcloud fallback.

    Production accepts only the declared ADC profile and exact service-account
    identity. The fallback lets non-production local developer runs work when
    `gcloud auth login` exists but ADC has not been configured.
    """
    try:
        creds, project = google.auth.default(scopes=scopes)
        if production:
            _validate_production_credentials(
                creds,
                credential_profile,
                expected_service_account,
            )
        return project, creds
    except DefaultCredentialsError as err:
        if production:
            raise RuntimeError(
                "Production Application Default Credentials could not be resolved"
            ) from err
        project = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if project == "(unset)":
            project = ""
        if not project:
            raise RuntimeError("No gcloud project configured") from err
        return project, GcloudCliCredentials(scopes)
