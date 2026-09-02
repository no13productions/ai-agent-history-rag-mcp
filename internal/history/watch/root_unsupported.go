//go:build !darwin && !linux && !windows

package watch

type platformRoot struct{}

func (*platformRoot) isBound() bool          { return false }
func (*platformRoot) identity() RootIdentity { return RootIdentity{} }
func (*platformRoot) bind(string) (bool, error) {
	return false, ErrUnsupportedWatch
}
func (*platformRoot) snapshot(string, string, []string, int64) (*Snapshot, error) {
	return nil, ErrUnsupportedWatch
}
func (*platformRoot) close() error { return nil }
