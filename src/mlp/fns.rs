use std::f64::consts::E;

use ndarray::Array2;

pub fn softmax(z: &Array2<f64>) -> Array2<f64> {
    z.exp() / z.exp().sum()
}

pub fn sigmoid(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|v| (LOGISTIC.function)(&v))
}

#[derive(Clone, Copy, Debug)]
pub struct MLPFunc {
    pub function: fn(&f64) -> f64,
    pub derivative: fn(&f64) -> f64,
}

impl Default for MLPFunc {
    fn default() -> Self {
        LOGISTIC
    }
}

pub const LOGISTIC: MLPFunc = MLPFunc {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| x * (1.0 - x),
};

#[cfg(test)]
mod test {
    use ndarray::Array;

    use super::softmax;

    #[test]
    fn test_softmax() {
        let z = Array::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let result = softmax(&z);
        let expected =
            Array::from_shape_vec((3, 1), vec![0.09003057, 0.24472847, 0.66524096]).unwrap();
        assert_eq!(result.shape(), expected.shape());
        for i in 0..result.len() {
            assert!((result[[i, 0]] - expected[[i, 0]]).abs() < 1e-6);
        }
    }
}
