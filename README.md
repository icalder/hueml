# hueml

<!-- Optional: Add badges here (e.g., build status, license) -->
<!-- [![Build Status](https://travis-ci.org/username/hueml.svg?branch=main)](https://travis-ci.org/username/hueml) -->
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

## About The Project

`hueml` is a project written in Rust.

It is designed for experimenting with MLP algorithms and training, using ndarray.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   **Rust toolchain:** Ensure you have Rust installed. You can get it from rust-lang.org.
    ```sh
    rustc --version
    cargo --version
    ```

### Installation & Building

1.  Clone the repo:
    ```sh
    git clone <YOUR_REPOSITORY_URL_HERE>
    cd hueml
    ```
2.  Build the project:
    ```sh
    cargo build
    ```
    For a release build:
    ```sh
    cargo build --release
    ```

## Usage

*(Provide instructions and examples for use. Include code snippets or command-line examples if applicable.)*

To run the project (if it's an executable):
```sh
cargo run
```

Or, if you've built a release binary:
```sh
./target/release/hueml  # (or hueml.exe on Windows)
```

Examples of some commands:
```powershell
cargo run --release export-db --filename data/2023-mar.parquet --from 2023-03-01 --to 2023-04-01

cargo run --release train --filename data/2023-mar.parquet --epochs 3000 --layers 3,4,2

cargo run --release predict --filename data/2024-mar.parquet
```