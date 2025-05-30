use chrono::{DateTime, Utc};

use crate::db::LightState;

#[derive(Debug, PartialEq)]
pub struct LightSample {
    pub state: LightState,
    pub time: DateTime<Utc>,
}

impl LightSample {
    pub fn on(&self) -> f64 {
        if self.state == LightState::On {
            1.0
        } else {
            0.0
        }
    }
}
