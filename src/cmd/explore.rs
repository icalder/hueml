use clap::Args;
use futures::TryStreamExt;
use sqlx::postgres::PgPoolOptions;

use super::cli::parse_date;
use crate::db;

#[derive(Args)]
pub struct ExploreArgs {
    /// database connection
    ///
    /// example: // postgres://huebot:huebotpw@192.168.1.46:30529/huebot
    /// $env:DATABASE_URL="postgres://huebot:huebotpw@192.168.1.46:30529/huebot"
    #[arg(long, env = "DATABASE_URL")]
    pub db_conn: String,
    /// from date, example: 2022-03-21
    #[arg(short, long, value_parser = parse_date)]
    from: chrono::NaiveDate,
    /// to date, example: 2022-03-22
    #[arg(short, long, value_parser = parse_date)]
    to: chrono::NaiveDate,
}

pub async fn run(args: &ExploreArgs) -> Result<(), sqlx::Error> {
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(&args.db_conn)
        .await?;

    let mut sql_buf = String::new();
    let mut results = db::stream_query(&pool, &mut sql_buf, Some(args.from), Some(args.to)).await;
    let mut row_ctr = 1;
    while let Some(light_data) = results.try_next().await? {
        println!("{}, {:?}", row_ctr, light_data);
        row_ctr += 1;
    }

    Ok(())
}
