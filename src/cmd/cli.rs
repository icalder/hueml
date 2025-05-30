use std::path::PathBuf;

use clap::{Parser, Subcommand};

// Nested subcommands example
// see: https://github.com/clap-rs/clap/blob/3ef784b516b2c9fbf6adb1c3603261b085561be7/examples/git-derive.rs

#[derive(Parser)]
// NB: version from cargo.toml will be used by default
#[command(version, about="ML for hue lights on/off prediction", long_about = None)]
pub struct Cli {
    /// Sets a custom config file
    #[arg(short, long, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub debug: u8,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    Explore(super::explore::ExploreArgs),
    /// example: cargo run --release export-db --filename data/2023.parquet --from 2023-01-01 --to 2023-12-31
    ExportDB(super::exportdb::ExportDBArgs),
    /// example: cargo run --release import --filename data/2023.parquet
    Import(super::import::ImportArgs),
    /// example: cargo run --release train --from 2022-12-10 --to 2022-12-24
    Train(super::train::TrainArgs),
    Predict(super::predict::PredictArgs),
}

// Common argument parsing helper functions
pub fn parse_date(arg: &str) -> Result<chrono::NaiveDate, chrono::ParseError> {
    chrono::NaiveDate::parse_from_str(arg, "%Y-%m-%d")
}
