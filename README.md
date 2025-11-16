
# Capacitance Analysis Tool

A Python-based tool for analyzing capacitance recordings from electrophysiological experiments, specifically designed for processing and analyzing capacitance recordings from HEKA data files.

## Features

- **Capacitance trace analysis** with multiple exponential fitting approaches:
  - Single exponential fitting
  - Single exponential with offset 
  - Double exponential fitting
- **Calcium current analysis** including:
  - Steady-state current (Ca_ss)
  - Tail current analysis (Ca_tail)
- **Processing**: Automated detection of stimulus timing (t0, t1) from voltage command
- **Baseline subtraction** with linear trend removal
- **Median filtering** for noise reduction
- **Group analysis** 
  - Mean ± parametric SEM
  - Median ± bootstrap SEM
- **Automated batch processing** of multiple recordings
- **Comprehensive Excel export** with detailed results
- **Interactive browser interface** for trace visualization

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Data Folder Setup
Configure the paths in `config.py`. You should copy `config_template.py` to `config.py` and adjust the paths to your local paths and your initials. Note that `config.py` is excluded from version control (see `.gitignore`).

### Metadata File Requirements
Create a metadata Excel file (`metadata.xlsx`) in your IMPORT_FOLDER defined in `config.py`. Each row represents one cell/recording with the following required columns:

#### Basic Metadata Structure:
| file_name | snapshot_3ms_1_series | snapshot_30ms_1_series | sine_3ms_1_series | snapshot_3ms_2_series | snapshot_30ms_2_series | groups |
|-----------|----------------------|------------------------|-------------------|----------------------|------------------------|---------|
| cell_001.dat | 1 | 2 | 3 | 4 | 5 | control |
| cell_002.dat | 1 | 2 | 3 | 4 | 5 | treatment |
| cell_003.dat | 1 | 2 | 3 | 4 | 5 | control |

#### Columns Explained:

**Basic Recording Information:**
- `file_name`: Name of the HEKA data file (.dat extension)
- `snapshot_3ms_1_series`: Series number for 3ms snapshot protocol (first run)
- `snapshot_30ms_1_series`: Series number for 30ms snapshot protocol (first run)
- `sine_3ms_1_series`: Series number for 3ms sine wave protocol (first run)
- `snapshot_3ms_2_series`: Series number for 3ms snapshot protocol (second run)
- `snapshot_30ms_2_series`: Series number for 30ms snapshot protocol (second run)
- `groups`: Optional experimental group identifier for statistical comparisons

**Note:** You can skip analysis for any trace type by leaving the corresponding series cell empty in the metadata Excel file.

## Usage

1. **Prepare your data:**
   - Place your HEKA data files in the external data folder (as configured in `config.py`)
   - Create the metadata Excel file with all required columns for each cell

2. **Run the analysis:**
   ```bash
   python main.py
   ```
   - Performs full analysis and automatically launches the browser interface

3. **Browser interface:**
   - Interactive visualization of traces
   - Navigate through HEKA file structure
   - Select files from dropdown menu based on metadata
   - Can be launched separately: `python browser.py`

## Output

The analysis generates a **timestamped output folder** containing:

### Individual Cell Analysis
- **Individual PDF files** with analysis plots for each recording:
  - 5 rows (one per trace type) × 5 columns per cell:
    - Column 1: Original capacitance trace with baseline fit
    - Column 2: Baseline-subtracted trace with 1-exponential fit
    - Column 3: Baseline-subtracted trace with 1-exponential + offset fit
    - Column 4: Baseline-subtracted trace with 2-exponential fit
    - Column 5: Current trace around stimulus period

### Group Analysis
- **Combined analysis PDFs** (one per group):
  - 5 rows (trace types) × 3 columns:
    - Column 1: Superposition of individual traces
    - Column 2: Mean ± parametric SEM
    - Column 3: Median ± bootstrap SEM

### Results Files
- **`results.xlsx`**: Main results file with fitted parameters for each cell and trace type
- **Used data and code folder**: Copies of metadata and analysis code for reproducibility

## Analysis Results Details

### Main Results Excel File
The `results.xlsx` file contains the following parameters for each trace type:

**Fitted Parameters:**
- `{trace_type}_baseline`: Baseline capacitance value (pF)
- `{trace_type}_1exp_A`: Amplitude of single exponential fit (pF)
- `{trace_type}_1exp_tau`: Time constant of single exponential fit (s)
- `{trace_type}_1expY_A`: Amplitude of single exponential + offset fit (pF)
- `{trace_type}_1expY_tau`: Time constant of single exponential + offset fit (s)
- `{trace_type}_1expY_y0`: Offset of single exponential + offset fit (pF)
- `{trace_type}_2exp_A`: Total amplitude of double exponential fit (pF)
- `{trace_type}_2exp_tau1`: Fast time constant of double exponential fit (s)
- `{trace_type}_2exp_aRel`: Relative amplitude of slow component (0-1)
- `{trace_type}_2exp_tau2`: Slow time constant of double exponential fit (s)

**Calcium Current Analysis:**
- `{trace_type}_ca_ss`: Steady-state calcium current (pA)
- `{trace_type}_ca_tail`: Tail current amplitude (pA)

Where `{trace_type}` is one of: `snapshot_3ms_1`, `snapshot_30ms_1`, `sine_3ms_1`, `snapshot_3ms_2`, `snapshot_30ms_2`

## Technical Details

### Signal Processing
- **Baseline correction**: Linear trend removal from pre-stimulus period
- **Noise reduction**: Median filtering with configurable window size
- **Exponential fitting**: Non-linear least squares optimization with bounds
- **Statistical analysis**: Bootstrap resampling for non-parametric SEM calculation

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- pandas: Data manipulation and Excel I/O
- numpy: Numerical computations
- matplotlib: Plotting and visualization  
- scipy: Signal processing and curve fitting
- heka_reader: HEKA file format support
- PyQt5 & pyqtgraph: Interactive browser interface

## Acknowledgements

This tool uses the HEKA reader provided by: [https://github.com/campagnola/heka_reader](https://github.com/campagnola/heka_reader). It was adapted to correctly import the trace from the lock-in amplifier. The browser interface has been adapted to allow file selection from metadata

For questions, please contact stefan_jens.hallermann@uni-leipzig.de