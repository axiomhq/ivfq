package ivfq

import (
	"encoding/json"
	"math"
	"testing"
)

func TestSparseStringKeys(t *testing.T) {
	var a, b SparseVector
	if err := json.Unmarshal([]byte(`{"apple":2,"pomme":3,"dim1":4,"dim01":5,"猫":6,"indices":7,"values":8}`), &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(`{"apple":3,"dim1":2,"dim01":1,"猫":-1,"values":1}`), &b); err != nil {
		t.Fatal(err)
	}
	if got := SparseDot(a, b); got != 21 {
		t.Fatalf("dot = %v, want 21", got)
	}
	if got := SparseDot(a, SparseVector{Keys: []string{"dim1"}, Values: []float32{2}}); got != 8 {
		t.Fatalf("typed keys = %v, want 8", got)
	}
	encoded, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	var object map[string]float32
	if err := json.Unmarshal(encoded, &object); err != nil || len(object) != 7 || object["猫"] != 6 || object["dim01"] != 5 {
		t.Fatalf("string-key roundtrip = %s (%v)", encoded, err)
	}
	for _, raw := range []string{`{"apple":null}`, `{"apple":"two"}`, `{"apple":1e100}`, `{"apple":[1]}`, `[]`, `{"indices":null,"apple":1}`, `{"indices":[1],"values":[2]}`} {
		if err := json.Unmarshal([]byte(raw), &a); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
}

func TestSparseVectorValidation(t *testing.T) {
	valid := SparseVector{Keys: []string{"", "a", "b"}, Values: []float32{1, 0, -2}}
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, v := range []SparseVector{
		{Keys: []string{"a"}, Values: nil},
		{Keys: nil, Values: []float32{1}},
		{Keys: []string{"a", "a"}, Values: []float32{1, 2}},
		{Keys: []string{"b", "a"}, Values: []float32{1, 2}},
		{Keys: []string{"a"}, Values: []float32{float32(math.NaN())}},
		{Keys: []string{"a"}, Values: []float32{float32(math.Inf(1))}},
	} {
		if err := v.Validate(); err == nil {
			t.Fatalf("accepted invalid sparse vector %+v", v)
		}
	}
	if err := (SparseVector{}).Validate(); err != nil {
		t.Fatalf("empty sparse vector: %v", err)
	}
	// No cap on the key count: the caller enforces its own.
	wide := SparseVector{Keys: make([]string, 5000), Values: make([]float32, 5000)}
	for i := range wide.Keys {
		wide.Keys[i] = string(rune(0x1000 + i))
	}
	if err := wide.Validate(); err != nil {
		t.Fatalf("5,000 keys: %v", err)
	}
}
