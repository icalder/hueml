use clap::Args;
use itertools::multizip;
use polars::prelude::*;
use thiserror::Error;

use std::fs;

use crate::db::{LightEvent, LightState};

#[derive(Args)]
pub struct ImportArgs {
    #[arg(short, long)]
    filename: String,
}

// https://pola-rs.github.io/polars-book/user-guide/

#[derive(Debug, Error)]
pub enum ImportError {
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),
    #[error("Insufficient layers: {0}")]
    NotEnoughLayers(String),
}

// NB: not lazy, polars LazyFrame::scan doesn't seem to play well with async
// see: https://github.com/pola-rs/polars/issues/22713

// let lf = LazyFrame::scan_parquet(args.filename.as_str(), Default::default())?;
// let df = lf.collect()?;
// let df_head = df.head(Some(3));
// println!("{:?}", df_head);

pub fn run(args: &ImportArgs) -> Result<(), ImportError> {
    let mut file = fs::File::open(&args.filename)?;
    let df = ParquetReader::new(&mut file).finish()?;

    // randomly sample the data
    // let df = df.sample_frac(
    //     &Series::new(PlSmallStr::from_static("frac"), &[1]),
    //     false,
    //     true,
    //     None,
    // )?;

    // https://docs.rs/polars/latest/polars/docs/eager/index.html#sort
    // let result = df
    //     .sample_frac(
    //         &Series::new(PlSmallStr::from_static("frac"), &[0.2]),
    //         false,
    //         false,
    //         None,
    //     )?
    //     .lazy()
    //     .select([all().exclude(["id"])])
    //     .collect()?;

    // See https://stackoverflow.com/questions/72440403/iterate-over-rows-polars-rust
    let cols = df.take_columns();
    let timestamp = cols[0].datetime()?.as_datetime_iter();
    let state = cols[1].bool()?.iter();

    let combined = multizip((timestamp, state));
    let res: Vec<_> = combined
        .map(|(ts, st)| LightEvent {
            id: String::new(),
            creationtime: ts.unwrap().into(),
            state: if st.unwrap() {
                LightState::On
            } else {
                LightState::Off
            },
        })
        .collect();

    for le in res.iter() {
        println!("time: {}, state: {}", le.creationtime, le.state);
    }

    //print!("{:?}", res);

    //let mut col_iterator = result.iter();
    //let timestamps = col_iterator.next()?;

    //let h = result.head(Some(3));
    Ok(())
}
