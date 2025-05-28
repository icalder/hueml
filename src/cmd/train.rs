use std::fs;

use chrono::Utc;
use clap::Args;
use itertools::multizip;
use polars::prelude::*;

use super::import::ImportError;
use crate::{
    data::{idg::make_input_data_vector, types::LightSample},
    db::LightState,
    mlp::{
        config::{MLPConfig, TrainingState},
        fns::LOGISTIC,
        mlp::MLP,
    },
};

#[derive(Args)]
pub struct TrainArgs {
    #[arg(short, long)]
    filename: String,
    #[arg(short, long, value_delimiter = ',')]
    layers: Vec<usize>,
    #[arg(long, default_value_t = 100)]
    epochs: u32,
    #[arg(short, long, default_value = "data/mlp.json")]
    mlp_filename: String,
}

pub async fn run(args: &TrainArgs) -> Result<(), ImportError> {
    if args.layers.len() < 2 {
        return Err(ImportError::NotEnoughLayers(String::from(
            "At least 2 layers must be defined",
        )));
    }

    let mut file = fs::File::open(&args.filename)?;
    let df = ParquetReader::new(&mut file).finish()?;

    // randomly sample the data
    let df = df.sample_frac(
        &Series::new(PlSmallStr::from_static("frac"), &[1]),
        false,
        true,
        None,
    )?;

    let mut targets = Vec::new();
    let mut inputs = Vec::new();

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

    for le in res.iter() {
        let input_vec = make_input_data_vector(le);
        inputs.push(input_vec);
        targets.push(vec![le.on()]);
    }

    let mut mlp = MLP::new(MLPConfig {
        layers: args.layers.clone(),
        activation: LOGISTIC,
        learning_rate: 0.5,
        training_state_updated: Some(|ts: TrainingState| {
            println!(
                "Epoch {} of {}; mse = {}",
                ts.epoch, ts.total_epochs, ts.mse
            );
        }),
    });

    mlp.train(inputs, targets, args.epochs);
    println!("Training complete!");
    mlp.dump(&args.mlp_filename)?;

    Ok(())
}
