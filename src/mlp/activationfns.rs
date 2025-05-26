use std::f64::consts::E;

#[derive(Clone, Copy, Debug)]
pub struct Activation {
    pub function: fn(&f64) -> f64,
    pub derivative: fn(&f64) -> f64,
}

impl Default for Activation {
    fn default() -> Self {
        LOGISTIC
    }
}

pub const LOGISTIC: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| x * (1.0 - x),
};
