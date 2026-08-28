package auth

import (
	"strings"
	"testing"
)

func TestGoogleSelectorAPIRequiresExactCloudPlatformScope(t *testing.T) {
	valid, err := NewGoogleSelector([]string{GoogleCloudPlatformScope})
	if err != nil {
		t.Fatalf("NewGoogleSelector(valid) error = %v", err)
	}
	if len(valid.Scopes()) != 1 || valid.Scopes()[0] != GoogleCloudPlatformScope {
		t.Fatalf("selector scopes = %#v", valid.Scopes())
	}
	caller := []string{GoogleCloudPlatformScope}
	immutable, err := NewGoogleSelector(caller)
	if err != nil {
		t.Fatal(err)
	}
	caller[0] = "https://www.googleapis.com/auth/spanner.data"
	returned := immutable.Scopes()
	returned[0] = "https://www.googleapis.com/auth/spanner.data"
	if immutable.Scopes()[0] != GoogleCloudPlatformScope {
		t.Fatal("caller mutated validated selector scope")
	}
	for name, scopes := range map[string][]string{
		"omitted":    nil,
		"empty":      {},
		"alternate":  {"https://www.googleapis.com/auth/spanner.data"},
		"extra":      {GoogleCloudPlatformScope, "https://www.googleapis.com/auth/spanner.data"},
		"duplicate":  {GoogleCloudPlatformScope, GoogleCloudPlatformScope},
		"whitespace": {" " + GoogleCloudPlatformScope},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := NewGoogleSelector(scopes); err == nil {
				t.Fatalf("NewGoogleSelector(%#v) accepted caller override", scopes)
			}
		})
	}
}

func TestGoogleCarrierDocumentRequiresExactCloudPlatformScope(t *testing.T) {
	valid := `{"type":"google_adc","scopes":["` + GoogleCloudPlatformScope + `"]}`
	document, err := ParseGoogleCarrierDocument([]byte(valid))
	if err != nil {
		t.Fatalf("ParseGoogleCarrierDocument(valid) error = %v", err)
	}
	if document.Type() != GoogleADCType || len(document.Scopes()) != 1 || document.Scopes()[0] != GoogleCloudPlatformScope {
		t.Fatalf("carrier document = %#v", document)
	}
	cases := map[string]string{
		"scope omitted":   `{"type":"google_adc"}`,
		"empty scopes":    `{"type":"google_adc","scopes":[]}`,
		"alternate":       `{"type":"google_adc","scopes":["https://www.googleapis.com/auth/spanner.data"]}`,
		"extra":           `{"type":"google_adc","scopes":["` + GoogleCloudPlatformScope + `","https://www.googleapis.com/auth/spanner.data"]}`,
		"caller override": `{"type":"google_adc","scope":"` + GoogleCloudPlatformScope + `","scopes":["` + GoogleCloudPlatformScope + `"]}`,
		"wrong carrier":   `{"type":"impersonated_service_account","scopes":["` + GoogleCloudPlatformScope + `"]}`,
		"duplicate field": `{"type":"google_adc","scopes":["` + GoogleCloudPlatformScope + `"],"scopes":["` + GoogleCloudPlatformScope + `"]}`,
		"trailing":        valid + `{}`,
	}
	for name, payload := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseGoogleCarrierDocument([]byte(payload)); err == nil {
				t.Fatalf("carrier document accepted: %s", payload)
			}
		})
	}
	if _, err := ParseGoogleCarrierDocument([]byte(valid + strings.Repeat(" ", MaxGoogleCarrierDocumentBytes))); err == nil {
		t.Fatal("oversized carrier document accepted")
	}
}
