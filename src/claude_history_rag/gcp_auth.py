"""Google Cloud authentication helpers for local and ADC environments."""

import json
import subprocess
from datetime import UTC, datetime

import google.auth
from google.auth import credentials
from google.auth.exceptions import DefaultCredentialsError

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


def default_project_and_credentials(scopes: list[str] | tuple[str, ...]):
    """Resolve project and credentials, falling back to active gcloud auth.

    ADC remains the preferred path. The fallback lets local developer runs work
    when `gcloud auth login` exists but `gcloud auth application-default login`
    has not been configured.
    """
    try:
        creds, project = google.auth.default(scopes=scopes)
        return project, creds
    except DefaultCredentialsError as err:
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
