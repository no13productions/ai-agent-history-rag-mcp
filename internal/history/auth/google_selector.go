package auth

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
)

const (
	GoogleCloudPlatformScope      = "https://www.googleapis.com/auth/cloud-platform"
	GoogleADCType                 = "google_adc"
	MaxGoogleCarrierDocumentBytes = 4 << 10
)

var ErrGoogleScopeContract = errors.New("google credential scope contract invalid")

type GoogleSelector struct {
	scope string
}

func NewGoogleSelector(scopes []string) (GoogleSelector, error) {
	if len(scopes) != 1 || scopes[0] != GoogleCloudPlatformScope {
		return GoogleSelector{}, ErrGoogleScopeContract
	}
	return GoogleSelector{scope: GoogleCloudPlatformScope}, nil
}

func (selector GoogleSelector) Scopes() []string {
	if selector.scope != GoogleCloudPlatformScope {
		return nil
	}
	return []string{GoogleCloudPlatformScope}
}

type GoogleCarrierDocument struct {
	carrierType string
	selector    GoogleSelector
}

func (document GoogleCarrierDocument) Type() string { return document.carrierType }

func (document GoogleCarrierDocument) Scopes() []string { return document.selector.Scopes() }

type googleCarrierWire struct {
	Type   string   `json:"type"`
	Scopes []string `json:"scopes"`
}

func ParseGoogleCarrierDocument(payload []byte) (GoogleCarrierDocument, error) {
	if len(payload) == 0 || len(payload) > MaxGoogleCarrierDocumentBytes {
		return GoogleCarrierDocument{}, ErrGoogleScopeContract
	}
	if err := rejectDuplicateJSONKeys(payload); err != nil {
		return GoogleCarrierDocument{}, ErrGoogleScopeContract
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	var wire googleCarrierWire
	if err := decoder.Decode(&wire); err != nil {
		return GoogleCarrierDocument{}, ErrGoogleScopeContract
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return GoogleCarrierDocument{}, ErrGoogleScopeContract
	}
	if wire.Type != GoogleADCType {
		return GoogleCarrierDocument{}, ErrGoogleScopeContract
	}
	selector, err := NewGoogleSelector(wire.Scopes)
	if err != nil {
		return GoogleCarrierDocument{}, err
	}
	return GoogleCarrierDocument{carrierType: GoogleADCType, selector: selector}, nil
}

func rejectDuplicateJSONKeys(payload []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	if err := walkJSONValue(decoder); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return ErrGoogleScopeContract
	}
	return nil
}

func walkJSONValue(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delimiter {
	case '{':
		seen := make(map[string]struct{})
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return err
			}
			key, ok := keyToken.(string)
			if !ok {
				return ErrGoogleScopeContract
			}
			if _, exists := seen[key]; exists {
				return ErrGoogleScopeContract
			}
			seen[key] = struct{}{}
			if err := walkJSONValue(decoder); err != nil {
				return err
			}
		}
		closing, err := decoder.Token()
		if err != nil || closing != json.Delim('}') {
			return ErrGoogleScopeContract
		}
	case '[':
		for decoder.More() {
			if err := walkJSONValue(decoder); err != nil {
				return err
			}
		}
		closing, err := decoder.Token()
		if err != nil || closing != json.Delim(']') {
			return ErrGoogleScopeContract
		}
	default:
		return ErrGoogleScopeContract
	}
	return nil
}
