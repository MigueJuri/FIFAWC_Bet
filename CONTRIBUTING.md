# Contributing

## Scope

This repository focuses on readable, reproducible betting-market probability workflows.

## Guidelines

- Keep scripts focused on one responsibility.
- Prefer shared utilities (for example, in `odds_devig.py`) over duplicated logic.
- Add concise docstrings for non-obvious math/model assumptions.
- Keep notebook changes aligned with script terminology.

## Validation checklist

Before opening a PR:

1. Install dependencies from `requirements.txt`.
2. Run:
   - `python BetfairBayesianScoreline.py`
   - `python Method_Comparison.py`
3. Confirm commands in `README.md` still match actual behavior and outputs.
