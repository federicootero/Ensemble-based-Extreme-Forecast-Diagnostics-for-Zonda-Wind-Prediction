# Evaluation of Ensemble-based Extreme Forecast Diagnostics for Zonda Wind Prediction

This repository contains the Python scripts and processed datasets used in the study:

Otero, F., and Gascón, E. (2026): Evaluation of Ensemble-based Extreme Forecast Diagnostics for Zonda Wind Prediction, submitted to Journal of Geophysical Research: Atmospheres.

## Data availability

- ECMWF ensemble forecast data are subject to licensing restrictions and are available from the European Centre for Medium-Range Weather Forecasts (ECMWF) for registered users via https://www.ecmwf.int.
- The list of Zonda wind events and processed observational datasets used in this study are provided in the `data/` directory.

## Software requirements

Python ≥ 3.9  
Main libraries:
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib

Install dependencies with: pip install -r requirements.txt

## Reproducibility

The scripts allow reproduction of all figures and verification metrics presented in the manuscript, assuming access to the licensed ECMWF ensemble forecast data.
