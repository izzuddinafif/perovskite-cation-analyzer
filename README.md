# perovskite-cation-analyzer

## Overview

This repository contains a suite of Python scripts designed for the computational analysis of mixed-cation perovskite structures, particularly focusing on Cs<sub>0.5</sub>A<sub>0.5</sub>PbI<sub>3</sub> (A = FA<sup>+</sup>, MA<sup>+</sup>) systems. The tools support research into the impact of nanoscale cation arrangement on material properties such as bandgap tunability and structural stability, as detailed in the paper "Unveiling the Impact of Nanoscale Cation Arrangement on Bandgap Tunability and Structural Stability in Cs<sub>0.5</sub>A<sub>0.5</sub>PbI<sub>3</sub> (A = FA+, MA+) Perovskites."

The scripts facilitate the generation of unique crystal structures, calculation of structural parameters like bond angles and tolerance factors, analysis of lattice strain, and processing of spectroscopic data.

## Repository Contents

*   **`Structure_Generator.py`**: Generates unique perovskite crystal structure configurations by considering symmetry operations to eliminate redundant structures. Outputs CIF files.
*   **`Bond_Angle_Calculation.py`**: Reads VASP output files (e.g., POSCAR, CONTCAR) to calculate and report specified bond angles (e.g., Pb-I-Pb).
*   **`Calibration_Code.py`**: Processes UV-Vis spectroscopy data from various CSV/TXT formats. It performs calibration by fitting a linear regression model, calculates molar absorptivity, and can predict concentrations of unknown samples.
*   **`Code_Tolerance_Factor.py`**: Computes the Goldschmidt tolerance factor for perovskite structures using `pymatgen` from CIF files. Outputs results to a CSV file and generates a plot.
*   **`Strain_Calculation_Code.py`**: Analyzes lattice strain and strain gradients between a reference and a deformed crystal structure (from CIF files), focusing on specified atoms (e.g., Pb and I).
*   **`structure_generation_workflow.md`**: Contains a Mermaid diagram illustrating the algorithmic workflow used in `Structure_Generator.py` for identifying unique crystal configurations.
*   **`.gitignore`**: Specifies files and directories to be ignored by Git version control (e.g., `result/`, `strain_analysis/`).
*   **`requirements.txt`**: Lists the Python dependencies required to run the scripts.

## Basic Usage

The Python scripts are generally designed to be run from the command line. Some scripts may accept command-line arguments for input files, parameters, and output directories.

Example:
```bash
python Structure_Generator.py --num_cs 4 --system_size 2 --parallel
python Strain_Calculation_Code.py --reference ref.cif --deformed def.cif
```
Please refer to the individual scripts for specific `argparse` options or internal configuration variables.

## Dependencies

The primary Python libraries required are:

*   `pymatgen`
*   `numpy`
*   `pandas`
*   `matplotlib`
*   `scikit-learn`

A `requirements.txt` file is provided for easy installation of dependencies:
```bash
pip install -r requirements.txt
```

## Output

Many scripts generate output files, such as:
*   CIF files for crystal structures (e.g., from `Structure_Generator.py` into the `result/` directory).
*   CSV files for data (e.g., tolerance factors, calibration data).
*   Plot images (e.g., calibration curves, tolerance factor plots, strain maps).
*   `.npz` files for numerical strain data.

Default output directories include `result/` (for `Structure_Generator.py`) and `strain_analysis/` (for `Strain_Calculation_Code.py`). These are included in `.gitignore`.
