# Data Directory

This directory contains data files related to the Oscillation Adaptability project.

## Structure

- `raw/`: Contains raw simulation data from model runs
- `processed/`: Contains processed data ready for analysis and visualization

## Data Format

Data files are primarily stored in CSV format with the following structure:

### Raw Data

- `simulation_YYYY-MM-DD_HHMMSS.csv`: Raw simulation output with timestamps
- `parameter_sweep_YYYY-MM-DD.csv`: Results from parameter sweeps

### Processed Data

- `conservation_validation.csv`: Data validating the conservation law
- `exponential_decay.csv`: Data validating the exponential decay relationship
- `oscillation_necessity.csv`: Data demonstrating necessary oscillations
- `spectral_analysis.csv`: Frequency domain analysis results

## Usage

These data files can be loaded and analyzed using the scripts in the `code/analysis/` directory.
