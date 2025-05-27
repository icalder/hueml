use std::default;

use super::fns::MLPFunc;
use serde::{Deserialize, Serialize};

pub struct TrainingState {
    pub total_epochs: u32,
    pub epoch: u32,
    pub mse: f64,
}

#[derive(Serialize, Deserialize)]
pub struct MLPConfig {
    pub layers: Vec<usize>,
    #[serde(skip)]
    pub activation: MLPFunc,
    pub learning_rate: f64,
    // Optional callback for training state updates
    #[serde(skip)]
    pub training_state_updated: Option<fn(TrainingState)>,
}

impl Default for MLPConfig {
    fn default() -> Self {
        Self {
            // 2 inputs, 3 neurons in hidden layer and 1 output
            layers: vec![2, 3, 1],
            activation: default::Default::default(),
            learning_rate: 0.1,
            training_state_updated: None,
        }
    }
}
