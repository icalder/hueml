use std::f64::consts::E;

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
