use std::fs;

use chrono::{Datelike, Utc};
use clap::Args;
use colored::Colorize;
use itertools::multizip;
use polars::prelude::*;

use crate::{
    data::{idg::make_input_data_vector, types::LightSample},
    db::LightState,
    mlp::mlp::MLP,
};

use super::import::ImportError;

#[derive(Args)]
pub struct PredictArgs {
    #[arg(short, long)]
    filename: String,
    #[arg(short, long, default_value = "data/mlp.json")]
    mlp_filename: String,
}

pub async fn run(args: &PredictArgs) -> Result<(), ImportError> {
    let mut file = fs::File::open(&args.filename)?;
    let df = ParquetReader::new(&mut file).finish()?;

    let cols = df.take_columns();
    let timestamp = cols[0].datetime()?.as_datetime_iter();
    let state = cols[1].bool()?.iter();

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

    let mut mlp = MLP::load(&args.mlp_filename, None)?;

    let mut count = 0;
    let mut success_count = 0;
    for le in res.iter() {
        let input_vec = make_input_data_vector(le);
        let output = mlp.feed_forward(input_vec);
        let prediction = if output[[0, 0]] > 0.5 { "on" } else { "off" };
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

    Ok(())
}
