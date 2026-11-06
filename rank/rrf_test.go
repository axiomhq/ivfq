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
	vec := []Hit{{ID: 1, Score: 0.9}, {ID: 2, Score: 0.8}, {ID: 3, Score: 0.7}}
	txt := []Hit{{ID: 2, Score: 5.0}, {ID: 4, Score: 4.0}}
	cases := []struct {
		name string
		legs [][]Hit
		k    int
		want []Hit
	}{
		{
			// 2: rank 2 + rank 1 beats 1: rank 1 alone.
			name: "two legs fuse",
			legs: [][]Hit{vec, txt},
			k:    10,
			want: []Hit{
				{ID: 2, Score: rr(2, 1)},
				{ID: 1, Score: rr(1)},
				{ID: 4, Score: rr(2)},
				{ID: 3, Score: rr(3)},
			},
		},
		{
			// Equal fused scores break ties by id ascending: both singleton
			// legs give their doc rank 1.
			name: "tie broken by id",
			legs: [][]Hit{{{ID: 26, Score: 1}}, {{ID: 13, Score: 1}}},
			k:    10,
			want: []Hit{{ID: 13, Score: rr(1)}, {ID: 26, Score: rr(1)}},
		},
		{
			name: "k truncates",
			legs: [][]Hit{vec, txt},
			k:    2,
			want: []Hit{{ID: 2, Score: rr(2, 1)}, {ID: 1, Score: rr(1)}},
		},
		{
			name: "empty leg is a no-op",
			legs: [][]Hit{vec, nil},
			k:    10,
			want: []Hit{{ID: 1, Score: rr(1)}, {ID: 2, Score: rr(2)}, {ID: 3, Score: rr(3)}},
		},
		{
			name: "no legs",
			legs: nil,
			k:    5,
			want: nil,
		},
		{
			// A doc repeated within one leg counts its best rank once:
			// 2's ranks are 1 (not 1+3) and 2 → same fusion as one clean copy.
			name: "duplicate id within a leg keeps best rank",
			legs: [][]Hit{{{ID: 2, Score: 3}, {ID: 1, Score: 2}, {ID: 2, Score: 1}}, txt},
			k:    10,
			want: []Hit{
				{ID: 2, Score: rr(1, 1)},
				{ID: 1, Score: rr(2)},
				{ID: 4, Score: rr(2)},
			},
		},
		{
			// k <= 0 means no hits — never a panic (public paths pass k through).
			name: "negative k yields no hits",
			legs: [][]Hit{vec},
			k:    -1,
			want: nil,
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
	vec := []Hit{{ID: 1, Score: 0.9}, {ID: 2, Score: 0.8}}
	txt := []Hit{{ID: 2, Score: 5.0}, {ID: 3, Score: 4.0}}
	// weights 3 and 1, constant 10: 1 = 3/11, 2 = 3/12 + 1/11, 3 = 1/12.
	got := Fusion{RankConstant: 10, Weights: []float64{3, 1}}.RRF(10, vec, txt)
	want := []Hit{
		{ID: 2, Score: float32(3.0/12 + 1.0/11)},
		{ID: 1, Score: float32(3.0 / 11)},
		{ID: 3, Score: float32(1.0 / 12)},
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
