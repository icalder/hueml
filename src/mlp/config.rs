use std::default;

use super::fns::{ActivationFnTypes, MLPFunc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub struct TrainingState {
    pub total_epochs: u32,
    pub epoch: u32,
    pub mse: f64,
}

#[derive(Serialize, Deserialize)]
pub struct MLPConfig {
    pub layers: Vec<usize>,
    #[serde(
        serialize_with = "activation_serializer",
        deserialize_with = "activation_deserializer"
    )]
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

fn activation_serializer<S>(activation: &MLPFunc, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_u8(activation.fntype as u8)
}

fn activation_deserializer<'de, D>(deserializer: D) -> Result<MLPFunc, D::Error>
where
    D: Deserializer<'de>,
{
    let e: u8 = Deserialize::deserialize(deserializer)?;
    ActivationFnTypes::try_from(e)
        .map_err(serde::de::Error::custom)
        .and_then(|fntype| match fntype {
            ActivationFnTypes::Logistic => Ok(MLPFunc::default()),
            ActivationFnTypes::Tanh => Ok(super::fns::TANH),
        })
}
