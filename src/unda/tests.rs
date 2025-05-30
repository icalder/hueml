#[cfg(test)]
use chrono::{Datelike, Utc};
#[cfg(test)]
use colored::Colorize;
#[cfg(test)]
use itertools::multizip;
#[cfg(test)]
use polars::prelude::*;
#[cfg(test)]
use std::fs;
#[cfg(test)]
use unda::core::{
    data::input::Input, layer::layers::InputTypes, layer::layers::LayerTypes,
    layer::methods::activations::Activations, layer::methods::errors::ErrorTypes, network::Network,
};

#[cfg(test)]
use crate::{
    data::{idg::make_input_data_vector, types::LightSample},
    db::LightState,
};

#[test]
fn hello() {
    let mut mlp = Network::new(1);
    mlp.set_input(InputTypes::DENSE(3));
    mlp.add_layer(LayerTypes::DENSE(3, Activations::SIGMOID, 0.001));
    mlp.add_layer(LayerTypes::DENSE(1, Activations::SIGMOID, 0.001));
    mlp.compile();

    let mut file = fs::File::open("data/2023-jan.parquet").unwrap();
    let df = ParquetReader::new(&mut file).finish().unwrap();

    // randomly sample the data
    let df = df
        .sample_frac(
            &Series::new(PlSmallStr::from_static("frac"), &[1]),
            false,
            true,
            None,
        )
        .unwrap();

    let mut targets = Vec::new();
    let mut inputs = Vec::new();

    let cols = df.take_columns();
    let timestamp = cols[0].datetime().unwrap().as_datetime_iter();
    let state = cols[1].bool().unwrap().iter();

    let combined = multizip((timestamp, state));
    let res: Vec<_> = combined
        .map(|(ts, st)| LightSample {
            time: ts.unwrap().and_local_timezone(Utc).unwrap(),
            state: if st.unwrap() {
                LightState::On
            } else {
                LightState::Off
            },
        })
        .collect();

    for le in res.iter() {
        let input_vec = make_input_data_vector(le);
        inputs.push(vec![
            input_vec[0] as f32,
            input_vec[1] as f32,
            input_vec[2] as f32,
        ]);
        targets.push(vec![le.on() as f32]);
    }

    mlp.fit(
        &inputs.iter().map(|v| v as &dyn Input).collect(),
        &targets,
        20,
        ErrorTypes::MeanAbsolute,
    );

    let mut count = 0;
    let mut success_count = 0;
    for le in res.iter() {
        let results = mlp.predict(&inputs[count]);
        let prediction = if results[0] > 0.5 { "on" } else { "off" };
        count += 1;
        if prediction == "on" && le.state == LightState::On
            || prediction == "off" && le.state == LightState::Off
        {
            println!(
                "{} ({}): {}",
                le.time.to_string().green(),
                le.time.weekday().to_string().green(),
                le.state.to_string().green()
            );
            success_count += 1;
        } else {
            println!(
                "{} ({}): {}",
                le.time.to_string().red(),
                le.time.weekday().to_string().green(),
                le.state.to_string().red()
            );
        }
    }
    println!(
        "Success rate: {:.1}%",
        (success_count as f64 / count as f64) * 100.0
    );
    //let results = mlp.predict(&inputs);
    //println!("{:?}", results);
}
