mod cmd;
mod data;
mod db;
mod mlp;

use cmd::cli::Commands;
use cmd::prelude::*;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = cmd::cli::Cli::parse();

    println!("debug level = {}", cli.debug);

    match &cli.command {
        Commands::Explore(args) => cmd::explore::run(&cli.db_conn, args).await?,
        Commands::ExportDB(export_dbargs) => {
            cmd::exportdb::run(&cli.db_conn, export_dbargs).await?
        }
        Commands::Import(args) => cmd::import::run(args)?,
        Commands::Train(args) => cmd::train::run(args).await?,
        Commands::Predict(args) => cmd::predict::run(args).await?,
    }

    Ok(())
}
