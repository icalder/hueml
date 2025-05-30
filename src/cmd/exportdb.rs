use clap::Args;
use futures::{Stream, TryStreamExt};
use polars::prelude::*;
use sqlx::postgres::PgPoolOptions;
use thiserror::Error;
use tokio::pin;

use std::fs;

use super::cli::parse_date;
use crate::{
    data::tsg::LightTimeSeriesGenerator,
    db::{self, LightEvent, LightState},
};

#[derive(Args)]
pub struct ExportDBArgs {
    /// database connection
    ///
    /// example: // postgres://huebot:huebotpw@192.168.1.46:30529/huebot
    /// $env:DATABASE_URL="postgres://huebot:huebotpw@192.168.1.46:30529/huebot"
    #[arg(long, env = "DATABASE_URL")]
    pub db_conn: String,
    #[arg(short, long)]
    filename: String,
    /// from date, example: 2022-03-21
    #[arg(short, long, value_parser = parse_date)]
    from: chrono::NaiveDate,
    /// to date, example: 2022-03-22
    #[arg(short, long, value_parser = parse_date)]
    to: chrono::NaiveDate,
}

// https://pola-rs.github.io/polars-book/user-guide/

#[derive(Debug, Error)]
pub enum ExportDBError {
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("SQL error: {0}")]
    SQLError(#[from] sqlx::Error),
    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),
}

// pub async fn write_csv(
//     results: impl Stream<Item = Result<LightEvent, sqlx::Error>>,
//     file: impl std::io::Write,
// ) -> Result<(), ExportDBError> {
//     pin!(results);

//     let schema = Schema::from_iter(vec![
//         Field::new("id".into(), DataType::String),
//         Field::new(
//             "timestamp".into(),
//             DataType::Datetime(TimeUnit::Milliseconds, None),
//         ),
//         Field::new("state".into(), DataType::Boolean),
//     ]);

//     let csvwriter = CsvWriter::new(file);
//     let mut bw = csvwriter.batched(&schema)?;
//     let mut row_ctr = 1;

//     while let Some(light_data) = results.try_next().await? {
//         println!("{}, {:?}", row_ctr, light_data);
//         let df = df!(
//             "id" => [light_data.id],
//             "timestamp" => [light_data.creationtime],
//             "state" => [if light_data.state == LightState::On { true } else { false } ]
//         )?;
//         bw.write_batch(&df)?;
//         row_ctr += 1;
//     }
//     bw.finish()?;

//     Ok(())
// }

pub async fn write_parquet(
    results: impl Stream<Item = Result<LightEvent, sqlx::Error>>,
    file: impl std::io::Write,
) -> Result<(), ExportDBError> {
    pin!(results);

    let mut tsg = LightTimeSeriesGenerator::default();

    let schema = Schema::from_iter(vec![
        Field::new(
            "timestamp".into(),
            DataType::Datetime(TimeUnit::Milliseconds, Some(TimeZone::UTC)),
        ),
        Field::new("state".into(), DataType::Boolean),
    ]);

    let pqwriter = ParquetWriter::new(file);
    let mut bw = pqwriter.batched(&schema)?;

    let mut row_ctr = 1;

    while let Some(light_data) = results.try_next().await? {
        println!("{}, {:?}", row_ctr, light_data);
        tsg.event(light_data);

        while let Some(sample) = tsg.next() {
            let df = df!(
                "timestamp" => [sample.time.naive_utc()],
                "state" => [if sample.state == LightState::On { true } else { false } ]
            )?;
            bw.write_batch(&df)?;
        }
        row_ctr += 1;
    }
    bw.finish()?;

    Ok(())
}

pub async fn run(args: &ExportDBArgs) -> Result<(), ExportDBError> {
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(&args.db_conn)
        .await?;

    let mut sql_buf = String::new();
    let results = db::stream_query(&pool, &mut sql_buf, Some(args.from), Some(args.to)).await;

    let file = fs::File::create(&args.filename)?;
    //write_csv(results, file).await?;
    write_parquet(results, file).await?;

    Ok(())
}
