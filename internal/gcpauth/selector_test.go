package gcpauth

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const testIdentity = "history-rag-test@fixture-project.iam.gserviceaccount.com"
const cloudPlatformScopeFixture = "https://www.googleapis.com/auth/cloud-platform"

func validImpersonatedCarrier(identity string) string {
	return fmt.Sprintf(`{
  "type": "impersonated_service_account",
  "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/%s:generateAccessToken",
  "delegates": [],
  "source_credentials": {
    "type": "authorized_user",
    "client_id": "fixture-client",
    "client_secret": "fixture-secret",
    "refresh_token": "fixture-refresh",
    "token_uri": "https://oauth2.googleapis.com/token",
    "universe_domain": "googleapis.com"
  },
  "scopes": ["%s"],
  "quota_project_id": "fixture-project"
}`, identity, cloudPlatformScopeFixture)
}

func TestSelectorValidateClosesProfileUnion(t *testing.T) {
	for _, profile := range []string{CredentialsProfileImpersonatedServiceAccount, CredentialsProfileAttachedServiceAccount} {
		selector := Selector{
			CredentialsSource:   CredentialsSourceApplicationDefault,
			CredentialsProfile:  profile,
			CredentialsIdentity: testIdentity,
		}
		if err := selector.Validate("fixture"); err != nil {
			t.Fatalf("profile %s: %v", profile, err)
		}
	}

	for _, tt := range []struct {
		name     string
		selector Selector
		want     string
	}{
		{name: "missing source", selector: Selector{CredentialsProfile: CredentialsProfileImpersonatedServiceAccount, CredentialsIdentity: testIdentity}, want: "credentials_source"},
		{name: "unknown source", selector: Selector{CredentialsSource: "ambient", CredentialsProfile: CredentialsProfileImpersonatedServiceAccount, CredentialsIdentity: testIdentity}, want: "unsupported credentials_source"},
		{name: "missing profile", selector: Selector{CredentialsSource: CredentialsSourceApplicationDefault, CredentialsIdentity: testIdentity}, want: "credentials_profile"},
		{name: "unknown profile", selector: Selector{CredentialsSource: CredentialsSourceApplicationDefault, CredentialsProfile: "human_adc", CredentialsIdentity: testIdentity}, want: "unsupported credentials_profile"},
		{name: "human identity", selector: Selector{CredentialsSource: CredentialsSourceApplicationDefault, CredentialsProfile: CredentialsProfileImpersonatedServiceAccount, CredentialsIdentity: "person@example.com"}, want: "service-account email"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.selector.Validate("fixture"); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestImpersonatedCarrierRejectsNestedExportableKeyAndAuthorityDrift(t *testing.T) {
	valid := validImpersonatedCarrier(testIdentity)
	if err := validateImpersonatedConfiguration([]byte(valid), testIdentity); err != nil {
		t.Fatalf("valid carrier: %v", err)
	}
	for _, tt := range []struct {
		name   string
		mutate func(string) string
		want   string
	}{
		{name: "service account source", mutate: func(raw string) string {
			return strings.Replace(raw, `"type": "authorized_user"`, `"type": "service_account"`, 1)
		}, want: `source_credentials.type must be "authorized_user"`},
		{name: "nested private key", mutate: func(raw string) string {
			return strings.Replace(raw, `"type": "authorized_user"`, `"type": "authorized_user", "private_key": "forbidden"`, 1)
		}, want: "private_key"},
		{name: "nested private key id", mutate: func(raw string) string {
			return strings.Replace(raw, `"type": "authorized_user"`, `"type": "authorized_user", "private_key_id": "forbidden"`, 1)
		}, want: "private_key_id"},
		{name: "wrong target", mutate: func(raw string) string {
			return strings.Replace(raw, testIdentity, "other-test@fixture-project.iam.gserviceaccount.com", 1)
		}, want: "does not match credentials_identity"},
		{name: "delegate", mutate: func(raw string) string {
			return strings.Replace(raw, `"delegates": []`, `"delegates": ["delegate@fixture-project.iam.gserviceaccount.com"]`, 1)
		}, want: "delegates must be an empty array"},
		{name: "wrong scope", mutate: func(raw string) string {
			return strings.Replace(raw, cloudPlatformScopeFixture, "https://www.googleapis.com/auth/devstorage.read_only", 1)
		}, want: "scopes must contain only"},
		{name: "missing quota project", mutate: func(raw string) string {
			return strings.Replace(raw, ",\n  \"quota_project_id\": \"fixture-project\"", "", 1)
		}, want: "quota_project_id is required"},
		{name: "unknown field", mutate: func(raw string) string {
			return strings.Replace(raw, `"delegates": []`, `"delegates": [], "fallback": true`, 1)
		}, want: "not allowed"},
		{name: "duplicate key", mutate: func(raw string) string {
			return strings.Replace(raw, `"delegates": []`, `"delegates": [], "delegates": []`, 1)
		}, want: "duplicate JSON key"},
		{name: "trailing data", mutate: func(raw string) string { return raw + `{}` }, want: "trailing data"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := validateImpersonatedConfiguration([]byte(tt.mutate(valid)), testIdentity)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

type fakeMetadataClient struct {
	onGCE bool
	email string
	raw   string
	path  string
}

func (f *fakeMetadataClient) OnGCEWithContext(context.Context) bool { return f.onGCE }
func (f *fakeMetadataClient) EmailWithContext(context.Context, string) (string, error) {
	return f.email, nil
}
func (f *fakeMetadataClient) GetWithContext(ctx context.Context, path string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	f.path = path
	return f.raw, nil
}

func TestAttachedTokenSourceBindsExactMetadataIdentity(t *testing.T) {
	clearAmbientGoogleEnvironment(t)
	client := &fakeMetadataClient{
		onGCE: true,
		email: testIdentity,
		raw:   `{"access_token":"metadata-token","token_type":"Bearer","expires_in":3600}`,
	}
	originalMetadata := newMetadataClient
	newMetadataClient = func() metadataClient { return client }
	t.Cleanup(func() { newMetadataClient = originalMetadata })

	selector := Selector{
		CredentialsSource:   CredentialsSourceApplicationDefault,
		CredentialsProfile:  CredentialsProfileAttachedServiceAccount,
		CredentialsIdentity: testIdentity,
	}
	source, err := selector.TokenSourceForRequest(context.Background(), cloudPlatformScopeFixture)
	if err != nil {
		t.Fatal(err)
	}
	token, err := source.Token()
	if err != nil {
		t.Fatal(err)
	}
	if token.AccessToken != "metadata-token" {
		t.Fatalf("access token = %q", token.AccessToken)
	}
	if !strings.Contains(client.path, "service-accounts/"+testIdentity+"/token") {
		t.Fatalf("metadata token path = %q", client.path)
	}

	client.email = "other-test@fixture-project.iam.gserviceaccount.com"
	if _, err := selector.TokenSourceForRequest(context.Background(), cloudPlatformScopeFixture); err == nil || !strings.Contains(err.Error(), "does not match credentials_identity") {
		t.Fatalf("wrong metadata identity error = %v", err)
	}
}

func TestReadOwnerOnlyRegularFileRejectsFinalPathSymlink(t *testing.T) {
	directory := t.TempDir()
	target := filepath.Join(directory, "carrier.json")
	if err := os.WriteFile(target, []byte(`{"type":"fixture"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(directory, "application_default_credentials.json")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	if _, err := readOwnerOnlyRegularFile(link); err == nil || !strings.Contains(err.Error(), "must not be a symlink") {
		t.Fatalf("symlink error = %v", err)
	}
}

func TestImpersonatedTokenSourceUsesOnlyWellKnownOwnerOnlyCarrier(t *testing.T) {
	clearAmbientGoogleEnvironment(t)
	directory := t.TempDir()
	path := filepath.Join(directory, "application_default_credentials.json")
	if err := os.WriteFile(path, []byte(validImpersonatedCarrier(testIdentity)), 0o600); err != nil {
		t.Fatal(err)
	}
	originalPath := defaultADCPath
	defaultADCPath = func() (string, error) { return path, nil }
	t.Cleanup(func() { defaultADCPath = originalPath })

	selector := Selector{
		CredentialsSource:   CredentialsSourceApplicationDefault,
		CredentialsProfile:  CredentialsProfileImpersonatedServiceAccount,
		CredentialsIdentity: testIdentity,
	}
	if source, err := selector.TokenSource(context.Background(), CloudPlatformScope); err != nil || source == nil {
		t.Fatalf("token source = %v, error = %v", source, err)
	}

	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", filepath.Join(directory, "static-key.json"))
	if _, err := selector.TokenSource(context.Background(), CloudPlatformScope); err == nil || !strings.Contains(err.Error(), "GOOGLE_APPLICATION_CREDENTIALS must be unset") {
		t.Fatalf("static-key override error = %v", err)
	}
}

func TestTokenSourceRejectsTransportAndPQCDowngradeOverrides(t *testing.T) {
	selector := Selector{
		CredentialsSource:   CredentialsSourceApplicationDefault,
		CredentialsProfile:  CredentialsProfileImpersonatedServiceAccount,
		CredentialsIdentity: testIdentity,
	}
	for _, tt := range []struct {
		key   string
		value string
		want  string
	}{
		{key: "HTTPS_PROXY", value: "https://attacker.invalid", want: "HTTPS_PROXY must be unset"},
		{key: "GOOGLE_API_USE_MTLS", value: "always", want: "GOOGLE_API_USE_MTLS must be unset"},
		{key: "GOOGLE_CLOUD_QUOTA_PROJECT", value: "attacker-project", want: "GOOGLE_CLOUD_QUOTA_PROJECT must be unset"},
		{key: "GODEBUG", value: "tlsmlkem=0", want: "tlsmlkem=0"},
		{key: "GODEBUG", value: "tlssecpmlkem=0", want: "tlssecpmlkem=0"},
		{key: "GODEBUG", value: "http2debug=2", want: "http2debug="},
	} {
		t.Run(tt.key+"/"+tt.value, func(t *testing.T) {
			clearAmbientGoogleEnvironment(t)
			t.Setenv(tt.key, tt.value)
			if _, err := selector.TokenSource(context.Background(), CloudPlatformScope); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func clearAmbientGoogleEnvironment(t *testing.T) {
	t.Helper()
	for _, key := range append([]string{"GODEBUG", "GOOGLE_APPLICATION_CREDENTIALS", "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", "CLOUDSDK_CONFIG"}, ambientGoogleTransportOverrideKeys[:]...) {
		t.Setenv(key, "")
	}
}
