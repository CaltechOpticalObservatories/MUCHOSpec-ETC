# MUCHOSpec-ETC

Exposure Time Calculator (ETC) for the MUCHOSpec / NGPS spectrograph on the Palomar 200-inch telescope.

The ETC computes signal-to-noise ratio (SNR), exposure time, or slit width for a source,
given the instrument configuration in `ETC/ETC_config.py` and the throughput, sky, and
source-template data bundled in the package.

## Installation

Requirements: Python 3.9 or higher and an up-to-date `pip`.

Install from GitHub:

```bash
pip install git+https://github.com/CaltechOpticalObservatories/MUCHOSpec-ETC.git
```

For development, clone the repository and install it in editable mode (preferably in a
virtual environment):

```bash
git clone https://github.com/CaltechOpticalObservatories/MUCHOSpec-ETC.git
cd MUCHOSpec-ETC
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Plotting (`-plotSNR`, `-plotslit`) needs matplotlib, which you can install with the `plots` extra:

```bash
pip install "muchospec-etc[plots] @ git+https://github.com/CaltechOpticalObservatories/MUCHOSpec-ETC.git"
```

## Command-line usage

Installing the package adds a `muchospec-etc` command. `python -m ETC` does the same thing.

```bash
muchospec-etc -h

# Example: exposure time to reach SNR=10 in the G channel between 500 and 510 nm
muchospec-etc G 500 510 SNR 10 -slit SET 0.5 -seeing 1 640 -airmass 1 -skymag 21.4 \
    -mag 18 -magsystem AB -magfilter match
```

## Python usage

The import name is `ETC`:

```python
from ETC.ETC_main import main, runETC, parser
from ETC.ETC_arguments import check_inputs_add_units

args = parser.parse_args("G 500 510 SNR 10 -slit SET 0.5 -seeing 1 640 -airmass 1 "
                         "-skymag 21.4 -mag 18 -magsystem AB -magfilter match".split())
check_inputs_add_units(args)
result = main(args)
print(result["exptime"])
```

## Package layout

```
ETC/
  ETC_main.py        main(), runETC(), and the command-line entry point cli()
  ETC_arguments.py   argument parser and input validation
  ETC_config.py      instrument parameters
  ETC_import.py      spectra, throughput, slit and PSF modeling
  ETC_plots.py       diagnostic plots
  CSV/               throughput, extinction and sky spectra
  sources/           source spectral templates (FITS)
  moffat_table_*.npz precomputed Moffat PSF integral tables
```
