# TIRF Microscopy Data Analysis – Master's Thesis

Python analysis code for my Master's thesis in Biomedical Engineering at the **Biophysics group, TU Wien** (2025–2026).

The thesis studied **supported lipid bilayers (SLBs)** functionalised via His-tag/Ni-NTA chemistry with **DNA origami platforms** and **pMHC complexes**, imaged with **total internal reflection fluorescence (TIRF) microscopy** at 26 °C and 37 °C. From the raw movies, the pipeline extracts three quantitative readouts:

| Readout | Question | Main entry point |
|---|---|---|
| **Surface density** | How many molecules per µm² are bound to the bilayer, and how does this change over time? | `surface_density_FOR.ipynb`, `TOCCSL_surface_density_AM.ipynb` |
| **Diffusion** | How mobile are the bound molecules (MSD analysis, diffusion coefficients, immobile fractions)? | `diffusion_AM_FORcycle_NEW.ipynb` |
| **Colocalisation** | Do two fluorescent species (two-channel imaging) overlap spatially? | `colocalization_AM_final.ipynb` |

A statistical comparison of the two temperature conditions is in `Significance_Test_slopes.py`.

---

## Authorship and attribution

This repository builds on the analysis toolkit shared within the Biophysics group. The original authors are:

- **Analysis notebooks** (surface density, diffusion, colocalisation): Anezka Majkova
- **`trc_handling.py`** and related trajectory utilities: Marina *(group-internal toolbox)*
- **[`sdt-python`](https://github.com/schuetzgroup/sdt-python)** (localisation, ROI, motion analysis, channel registration): Lukas Schrangl et al.
- Supporting modules (`helpers.py`, `cluster.py`, `laserprofile.py`, `mask_handling.py`, `optimal_roi.py`, `pdf_analysis.py`, `sm_handling.py`, `TOCCSL.py`, `data_analysis.py`) come from the group's shared codebase.

**My own contributions** (see commit history):

- **Single-molecule brightness** (`origami_analysis.py → brightness`, `brightness_final`)
  - Refactored the function to aggregate per imaging location first, then across locations, and to report the mean ± SEM per time point.
  - Added optional **nearest-neighbour filtering** (`sdt.spatial.has_near_neighbor`) so that overlapping signals don't inflate single-molecule brightness.
  - Made empty locations explicit (`NaN`) instead of letting them fail silently.
- **Time-course analysis**: brightness-over-time and diffusion-over-time plots with error bars and mean trends (`surface_density_FOR.ipynb`, `diffusion_AM_FORcycle_NEW.ipynb`).
- **Temperature comparison** (`boxplot1.ipynb`): box plots and forest plots for 26 °C vs 37 °C.
- **Statistical testing** (`Significance_Test_slopes.py`): well-wise slopes tested against zero (one-sample t-test, Cohen's d, 95 % CI) and between conditions (Welch's t-test, with Mann–Whitney U as a non-parametric check).
- Adapting all notebooks to my own experimental series and parameters.

---

## Pipeline overview

```
raw TIRF movies (.SPE / .tif)
        │
        ├─ ROI selection (homogeneous illumination, ImageJ coordinates)
        │
        ├─ localisation ───────── sdt-python locator → .h5 localisation tables
        │
        ├─ surface density ────── single-molecule counting (low density)
        │                         or bulk intensity ÷ single-molecule brightness (high density),
        │                         corrected for degree of labelling and ROI area
        │
        ├─ tracking & diffusion ─ trackpy linking → MSD fits (ensemble + individual),
        │                         diffusion populations (immobile / slow / mobile)
        │
        ├─ colocalisation ─────── bead-based channel registration → overlap of two channels
        │
        └─ statistics ─────────── per-well aggregation, mean ± SEM, significance tests
```

## Repository structure

```
├── surface_density_FOR.ipynb        # surface density + brightness over time (main thesis notebook)
├── TOCCSL_surface_density_AM.ipynb  # surface density for monomeric vs. clustered samples
├── diffusion_AM_FORcycle_NEW.ipynb  # tracking, MSD analysis, diffusion populations
├── colocalization_AM_final.ipynb    # two-channel colocalisation
├── boxplot1.ipynb                   # 26 °C vs 37 °C comparison plots
├── Significance_Test_slopes.py      # statistical tests on well-wise slopes
│
├── origami_analysis.py              # surface density & brightness functions
├── trc_handling.py                  # trajectory filtering, splitting, plotting, bootstrapping
├── sm_handling.py                   # single-molecule intensity distributions & filtering
├── mask_handling.py                 # pattern masks (on/off regions)
├── laserprofile.py                  # Gaussian laser-profile fit & illumination correction
├── optimal_roi.py                   # ROI optimisation for colocalisation
├── pdf_analysis.py, data_analysis.py# brightness PDFs, fitting, bootstrapping
├── cluster.py, helpers.py, TOCCSL.py# utilities
└── test.tiff                        # small example movie
```

## Setup

Python 3.10 was used.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Usage

1. Open the notebook for the readout you need.
2. In the **User input** section, set the data directory, file names, frame ranges, ROI coordinates and imaging parameters (pixel size, exposure time, degree of labelling).
3. Run the localisation step (`sdt.gui.locator` opens as a GUI) and save the `.h5` files next to the raw data.
4. Run the remaining cells. Results and plots are shown inline, and some are also saved to the data directory.

For the statistical comparison, enter the well-wise slopes in `Significance_Test_slopes.py` and run:

```bash
python Significance_Test_slopes.py
```

## Notes and limitations

- The notebooks contain **absolute local paths** from my own machine. Change these to your data location before running them.
- Raw experimental data are **not included** (only `test.tiff` as an example).
- This is research code, written for interactive analysis in notebooks. It is not a packaged library and has no automated tests.
