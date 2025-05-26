#[cfg(test)]
use ndarray::prelude::*;

// https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/index.html

// codemoon YT github repo:
// https://github.com/codemoonsxyz/neural-net-rs

#[test]
fn test_vector_dot_product() {
    // https://en.wikipedia.org/wiki/Dot_product
    // NB: can also use Array::from_vec()
    // The first way - use Array::from_vec to ensure we get a vector (shape has a single dimension [3]) not a matrix
    let a = array![1, 3, -5];
    let b = array![4, -2, -1];
    // println!("{:?}", a.shape()); [3]
    let dp = a.dot(&b);
    assert_eq!(dp, 3);
}

#[test]
fn test_matrix_dot_product() {
    // Column vectors (i.e. 3 x 1)
    let a = array![[1], [3], [-5]];
    let b = array![[4], [-2], [-1]];
    // println!("{:?}", a.shape()); //[3, 1]
    let dp = a.t().dot(&b);
    assert_eq!(dp[[0, 0]], 3);
}

// from https://www.youtube.com/watch?v=DKbz9pNXVdE
#[test]
fn test_feedforward() {
    // 1 x 3 inputs
    let inputs = array![[1.0, 1.0, 1.0]];
    // 3 x 2 weights
    let weights = array![[0.5, -0.25], [0.7, 0.1], [-0.2, 0.1]];
    // 1 x 2 bias
    let bias = array![[0.1, -0.2]];

    // Calculate weighted sum of inputs + bias
    let weighted_sum = inputs.dot(&weights) + bias;

    let expected_result = array![[1.1, -0.25]];
    assert_eq!(weighted_sum, expected_result);
}
