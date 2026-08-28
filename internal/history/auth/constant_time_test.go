package auth

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func TestCredentialVerificationUsesConstantTimeCompare(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "auth.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	ast.Inspect(file, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || selector.Sel.Name != "ConstantTimeCompare" {
			return true
		}
		identifier, ok := selector.X.(*ast.Ident)
		if ok && identifier.Name == "subtle" {
			found = true
		}
		return true
	})
	if !found {
		t.Fatal("credential verification lacks crypto/subtle.ConstantTimeCompare")
	}
}
