package ivfq

import (
	"encoding/json"
	"testing"
)

func TestMetricRoundTrips(t *testing.T) {
	for _, m := range []Metric{L2, Cosine, InnerProduct} {
		got, err := ParseMetric(m.String())
		if err != nil || got != m {
			t.Fatalf("ParseMetric(%q) = %v, %v", m.String(), got, err)
		}
		var back struct{ M Metric }
		data, err := json.Marshal(struct{ M Metric }{m})
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, &back); err != nil || back.M != m {
			t.Fatalf("json %s -> %v, %v", data, back.M, err)
		}
	}
	if _, err := ParseMetric("hamming"); err == nil {
		t.Fatal("ParseMetric accepted an unknown name")
	}
	if _, err := Metric(9).MarshalText(); err == nil {
		t.Fatal("MarshalText accepted an out-of-range Metric")
	}
}

func TestScorePanicsOnUnknownMetric(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Score accepted an out-of-range Metric")
		}
	}()
	Score(Metric(9), []float32{1}, []float32{1})
}
