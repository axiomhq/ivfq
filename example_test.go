package ivfq_test

import (
	"context"
	"fmt"

	"github.com/axiomhq/ivfq"
)

// Route a query to its nearest cluster centroids.
func ExampleBuild() {
	centroids := [][]float32{{0, 0}, {10, 0}, {0, 10}, {10, 10}}
	tree, err := ivfq.Build(context.Background(), centroids, 2)
	if err != nil {
		panic(err)
	}
	fmt.Println(tree.Nearest([]float32{9, 1}, 1))
	// Output: [1]
}
