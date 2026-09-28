//go:build unix

package bench

import "syscall"

// CPUSeconds is this process's user+system CPU time, for a CPU versus wall
// split: wall far above CPU is time spent waiting on I/O. 0 where getrusage
// is unavailable.
func CPUSeconds() float64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return float64(ru.Utime.Sec) + float64(ru.Utime.Usec)/1e6 + float64(ru.Stime.Sec) + float64(ru.Stime.Usec)/1e6
}
