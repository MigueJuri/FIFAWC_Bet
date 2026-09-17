# FIFAWC_Bet

Reader-friendly reference implementation for converting football betting odds into calibrated exact-score probabilities.

## Project purpose

This repository demonstrates two complementary workflows:

1. **Full scoreline pipeline** (`BetfairBayesianScoreline.py`):
   - Removes margin from quoted markets using Shin de-vigging.
   - Fits latent Poisson goal intensities from liquid markets.
   - Blends exact-score board prices with a Poisson background prior.
   - Calibrates the final grid to match 1X2, O/U 2.5, and BTTS marginals.
2. **Method comparison** (`Method_Comparison.py`):
   - Compares Power vs Shin de-vigging by checking how well exact-score probabilities map back to the liquid 1X2 market.

## Repository layout

- `/home/runner/work/FIFAWC_Bet/FIFAWC_Bet/BetfairBayesianScoreline.py` – end-to-end scoreline modeling script.
- `/home/runner/work/FIFAWC_Bet/FIFAWC_Bet/Method_Comparison.py` – standalone de-vig method comparison script.
- `/home/runner/work/FIFAWC_Bet/FIFAWC_Bet/odds_devig.py` – shared de-vigging utilities used by both scripts.
- `/home/runner/work/FIFAWC_Bet/FIFAWC_Bet/WorldCupPool.ipynb` – exploratory companion notebook.
- `/home/runner/work/FIFAWC_Bet/FIFAWC_Bet/BetfairBayesianScoreline_grid.png` – generated score-grid visualization artifact.

## Prerequisites

- Python 3.10+
- `pip`

## Installation

```bash
cd /home/runner/work/FIFAWC_Bet/FIFAWC_Bet
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the scripts

### 1) Full scoreline pipeline

```bash
cd /home/runner/work/FIFAWC_Bet/FIFAWC_Bet
python BetfairBayesianScoreline.py
```

Expected outputs:
- Console summary with fitted lambdas, marginal checks, top scorelines, and Monte Carlo pick.
- Updated image artifact: `BetfairBayesianScoreline_grid.png`.

### 2) De-vig method comparison

```bash
cd /home/runner/work/FIFAWC_Bet/FIFAWC_Bet
python Method_Comparison.py
```

Expected outputs:
- Baseline vs recovered 1X2 distributions for Power and Shin.
- KL divergence values for both methods.

## Notebook status

`WorldCupPool.ipynb` is kept as an exploratory companion. The Python scripts are the canonical, reproducible execution path.

## Modeling notes and limitations

- Inputs are assumed to be decimal odds with valid positive values.
- Score distributions are truncated to a finite goal grid and then renormalized.
- Tail outcomes are represented with grouped buckets (home/draw/away) beyond explicit exact-score quotes.
- Calibration quality depends on market consistency; contradictory market signals can force compromises.

## Contributing

See `CONTRIBUTING.md` for lightweight contribution and validation guidance.
