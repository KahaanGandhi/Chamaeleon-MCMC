<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/github/last-commit/KahaanGandhi/Chamaeleon-MCMC" alt="Last Commit">
  <img src="https://img.shields.io/github/languages/code-size/KahaanGandhi/Chamaeleon-MCMC" alt="Code Size">
  <a href="https://arxiv.org/abs/2503.01755">
    <img src="https://img.shields.io/badge/arXiv-2503.01755-b31b1b.svg" alt="arXiv">
  </a>
</p>

Spectroscopic observations enable studies of the chemical composition of star-forming regions based on their distinct molecular emission and absorption spectra. The difficulty of detecting complex organic molecules, which probe chemical complexity, scales with their size — emission from heavy molecules is distributed over many lines, often too weak to detect individually. To address this, Markov Chain Monte Carlo (MCMC) algorithms fit spectral models to sparse spectra, enabling molecular detection through multi-transition radio observations.

Contents include:
- `inference.py`: Main MCMC script; follow instructions below to run it.
- `notebooks/`: Preprocessing and postprocessing workflows.
- `scripts/`: Reference implementations of MCMC sampling.
- `spectral_simulator/`: Fast LTE spectral modeling tools.
- `catalog/`: Subset of CDMS catalog for relavent molecules.
- `data/`: Sample data, which can be used as a template for adding your own.

## Installation Instructions

First, clone this repository to your local machine:

```bash
git clone https://github.com/KahaanGandhi/Chamaeleon-MCMC.git
cd Chamaeleon-MCMC
```
Create and activate a new [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment:

```bash
conda create --name mcmc-env python=3.9
conda activate mcmc-env
```
Install the dependencies listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Running Instructions

### Step 1: Data Preparation 

- Verify your data is in the correct `.npy` format, containing frequency and intensity arrays for the molecule of interest. To reformat common file formats, refer to `notebooks/DSN_pipeline.ipynb`.
- Add your data to `data_paths` in `inference.py`, following the format:
```python
'molecule_name': os.path.join(os.getcwd(), 'your_data_folder', 'your_data_file.npy'),
```

### Step 2: Parameter Configuration

- Open the `inference.py` file and locate the `config` dictionary at the bottom of the script.
- **Observation parameters**: Adjust parameters like `dish_size`, `lower_limit`, and `upper_limit` to match your telescope and source. Ensure that your molecule has rotational transitions that fall in the specified frequency range. The current configuration is for DSS-43 observations of Chamaeleon I. 
- **Source size and column density**:
  - Source size is highly covariant with column density; in cases where source size can be estimated through other means, fixing it can better constrain column density. If unknown, it will be a free parameter.
  - Column density is initialized via maximum likelihood estimation (MLE) by default, supporting both fixed and variable source sizes.
- **Exploring sample space**: Increase `nwalkers` for broader exploration, especially if the parameter space is large or multimodal. Increase `nruns` (number of MCMC steps) to improve convergence and precision. As a rule of thumb, more walkers help with complex models, while more steps refine results in well-defined spaces. Look for signs like poor convergence or incomplete exploration to decide which to adjust.

### Step 3: Initial MCMC Sampling

- For your first run, set ```template_run``` to ```True``` to use hardcoded initial values specific to the template species.
- Cyanopolyynes tend to share source properties, so it is recommended to first obtain a fit for shorter linear molecules, as their transitions are more easily detectable above noise levels.
- Run the script from the main directory:
```bash
cd /path/to/Chamaeleon-MCMC
python inference.py
```

### Step 4: Refining the Fit

- After the initial run, set `template_run` to `False` to load posteriors from the previous run as priors.
- Rerun the script to refine the fit:
```bash
python inference.py
```
- You can redo the template run if needed; this will overwrite the previous results.
- Once satisfied, set `prior_path` to the saved location to use it for future non-template runs. You can keep multiple template runs, so check that `prior_path` points to the correct chain for each case.
- To plot the best-fit model against observed spectra or extract the peak model intensities of each rotational transition, refer to `notebooks/DSN_spectra.ipynb`.

## Citation

Please cite the software if you found it helpful with your work. If you have any questions or feedback, feel free to reach out via email at [kahaan@gmail.com](mailto:kahaan@gmail.com). 

```bibtex
@misc{lis20251825ghzspectroscopicsurvey,
  title         = {An 18-25 GHz spectroscopic survey of dense cores in the Chamaeleon I molecular cloud},
  author        = {Dariusz C. Lis and William D. Langer and Jorge L. Pineda and Kahaan Gandhi and Karen Willacy and Paul F. Goldsmith and Susanna Widicus Weaver and Liton Majumdar and Youngmin Seo and Shinji Horiuchi and Cheikh Bop and François Lique},
  year          = {2025},
  eprint        = {2503.01755},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.GA},
  url           = {https://arxiv.org/abs/2503.01755},
}