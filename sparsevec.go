package ivfq

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
)

// SparseVector is the string-keyed sparse vector: sorted, unique Keys with
// finite Values. Its JSON form is an object of key to weight. Sparse[T] is
// the int-indexed form.
type SparseVector struct {
	Keys   []string
	Values []float32
}

func (v SparseVector) MarshalJSON() ([]byte, error) {
	if err := v.Validate(); err != nil {
		return nil, err
	}
	object := make(map[string]float32, len(v.Keys))
	for i, key := range v.Keys {
		object[key] = v.Values[i]
	}
	return json.Marshal(object)
}

func (v *SparseVector) UnmarshalJSON(data []byte) error {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(data, &object); err != nil {
		return err
	}
	decoded := SparseVector{Keys: make([]string, 0, len(object))}
	for key := range object {
		decoded.Keys = append(decoded.Keys, key)
	}
	sort.Strings(decoded.Keys)
	for _, key := range decoded.Keys {
		var weight *float32
		if err := json.Unmarshal(object[key], &weight); err != nil || weight == nil {
			return fmt.Errorf("sparse dimension %q needs a finite numeric weight", key)
		}
		decoded.Values = append(decoded.Values, *weight)
	}
	if err := decoded.Validate(); err != nil {
		return err
	}
	*v = decoded
	return nil
}

// Validate checks equal lengths, finite values, and sorted unique keys. It
// sets no cap on the number of keys; callers enforce their own.
func (v SparseVector) Validate() error {
	if len(v.Keys) != len(v.Values) {
		return errors.New("sparse vector keys and values must have equal length")
	}
	for i, x := range v.Values {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return errors.New("sparse vector values must be finite")
		}
		if i > 0 && v.Keys[i-1] >= v.Keys[i] {
			return errors.New("sparse vector keys must be sorted and unique")
		}
	}
	return nil
}

// SparseDot is the reference sparse score. Products are taken in a's key
// order and each is rounded to float32 before it is added, so any scorer
// that reduces the same matched products in the same order reproduces the
// result bit for bit regardless of how postings are physically stored.
func SparseDot(a, b SparseVector) float32 {
	weights := make(map[string]float32, len(b.Values))
	for i, weight := range b.Values {
		weights[b.Keys[i]] = weight
	}
	var score float32
	for i, weight := range a.Values {
		score += float32(weight * weights[a.Keys[i]])
	}
	return score
}
