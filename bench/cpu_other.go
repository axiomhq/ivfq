//go:build !unix

package bench

// CPUSeconds is 0 where getrusage is unavailable.
func CPUSeconds() float64 { return 0 }
