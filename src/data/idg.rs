use chrono::{Datelike, Timelike};

use super::types::LightSample;

pub fn make_input_data_vector(sample: &LightSample) -> Vec<f64> {
    // Our independent variables are:
    // time of day (minutes since 00:00, normalised to 1)
    let mins = (sample.time.minute() + 60 * sample.time.hour()) as f64 / 1440.0;
    // day of week
    let day_of_week = (sample.time.weekday().num_days_from_monday()) as f64 / 6.0;
    // month
    let month = (sample.time.month0()) as f64 / 11.0;
    // day of year
    //let day_of_year = sample.time.ordinal() as f64 / 365.0;

    vec![mins, day_of_week, month]
}
