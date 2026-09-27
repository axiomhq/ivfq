package rank

import (
	"reflect"
	"testing"
)

func TestRRF(t *testing.T) {
	rr := func(ranks ...int) float32 {
		s := 0.0
		for _, r := range ranks {
			s += 1.0 / float64(rrfK+r)
		}
		return float32(s)
	}
	vec := []Hit{{ID: "a", Score: 0.9}, {ID: "b", Score: 0.8}, {ID: "c", Score: 0.7}}
	txt := []Hit{{ID: "b", Score: 5.0}, {ID: "d", Score: 4.0}}
	cases := []struct {
		name string
		legs [][]Hit
		k    int
		want []Hit
	}{
		{
			// b: rank 2 + rank 1 beats a: rank 1 alone.
			name: "two legs fuse",
			legs: [][]Hit{vec, txt},
			k:    10,
			want: []Hit{
				{ID: "b", Score: rr(2, 1)},
				{ID: "a", Score: rr(1)},
				{ID: "d", Score: rr(2)},
				{ID: "c", Score: rr(3)},
			},
		},
		{
			// Equal fused scores break ties by id ascending: both singleton
			// legs give their doc rank 1.
			name: "tie broken by id",
			legs: [][]Hit{{{ID: "z", Score: 1}}, {{ID: "m", Score: 1}}},
			k:    10,
			want: []Hit{{ID: "m", Score: rr(1)}, {ID: "z", Score: rr(1)}},
		},
		{
			name: "k truncates",
			legs: [][]Hit{vec, txt},
			k:    2,
			want: []Hit{{ID: "b", Score: rr(2, 1)}, {ID: "a", Score: rr(1)}},
		},
		{
			name: "empty leg is a no-op",
			legs: [][]Hit{vec, nil},
			k:    10,
			want: []Hit{{ID: "a", Score: rr(1)}, {ID: "b", Score: rr(2)}, {ID: "c", Score: rr(3)}},
		},
		{
			name: "no legs",
			legs: nil,
			k:    5,
			want: nil,
		},
		{
			// A doc repeated within one leg counts its best rank once:
			// b's ranks are 1 (not 1+3) and 2 → same fusion as one clean copy.
			name: "duplicate id within a leg keeps best rank",
			legs: [][]Hit{{{ID: "b", Score: 3}, {ID: "a", Score: 2}, {ID: "b", Score: 1}}, txt},
			k:    10,
			want: []Hit{
				{ID: "b", Score: rr(1, 1)},
				{ID: "a", Score: rr(2)},
				{ID: "d", Score: rr(2)},
			},
		},
		{
			// k <= 0 means no hits — never a panic (public paths pass k through).
			name: "negative k yields no hits",
			legs: [][]Hit{vec},
			k:    -1,
			want: []Hit{},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := RRF(tc.k, tc.legs...)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %+v want %+v", got, tc.want)
			}
		})
	}
}

func TestFusionRRF(t *testing.T) {
	vec := []Hit{{ID: "a", Score: 0.9}, {ID: "b", Score: 0.8}}
	txt := []Hit{{ID: "b", Score: 5.0}, {ID: "c", Score: 4.0}}
	// weights 3 and 1, constant 10: a = 3/11, b = 3/12 + 1/11, c = 1/12.
	got := Fusion{RankConstant: 10, Weights: []float64{3, 1}}.RRF(10, vec, txt)
	want := []Hit{
		{ID: "b", Score: float32(3.0/12 + 1.0/11)},
		{ID: "a", Score: float32(3.0 / 11)},
		{ID: "c", Score: float32(1.0 / 12)},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v want %+v", got, want)
	}
	if !reflect.DeepEqual(Fusion{}.RRF(10, vec, txt), RRF(10, vec, txt)) {
		t.Fatal("zero Fusion must equal RRF")
	}
	for _, bad := range []Fusion{
		{RankConstant: -1},
		{Weights: []float64{1}},
		{Weights: []float64{1, 1, 1}},
		{Weights: []float64{}},
		{Weights: []float64{1, 0}},
		{Weights: []float64{1, -2}},
	} {
		if err := bad.Validate(2); err == nil {
			t.Errorf("%+v accepted", bad)
		}
	}
	if err := (Fusion{Weights: []float64{1, 2}}).Validate(2); err != nil {
		t.Fatal(err)
	}
}
