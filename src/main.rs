mod cmd;
mod data;
mod db;
mod mlp;
mod unda;

use cmd::cli::Commands;
use cmd::prelude::*;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = cmd::cli::Cli::parse();

    println!("debug level = {}", cli.debug);

    match &cli.command {
        Commands::Explore(args) => cmd::explore::run(args).await?,
        Commands::ExportDB(args) => cmd::exportdb::run(args).await?,
        Commands::Import(args) => cmd::import::run(args)?,
        Commands::Train(args) => cmd::train::run(args).await?,
        Commands::Predict(args) => cmd::predict::run(args).await?,
    }

    Ok(())
}
