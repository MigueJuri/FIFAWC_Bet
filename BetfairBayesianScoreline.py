"""Betfair exact-score probability pipeline.

This script turns raw decimal odds into a coherent exact-score distribution.
It uses:
- Shin devigging on each quoted market.
- A Poisson score prior fitted to liquid markets.
- The correct-score board as a soft prior.
- Iterative proportional fitting to match 1X2, O/U 2.5, and BTTS.

The result is a finite, normalized score grid that sums exactly to 1.0 and
can be sampled for a final picked scoreline.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar

from odds_devig import normalize, shin_devig


Score = Tuple[int, int]
MAX_EXACT_SCORE = 3
TAIL_HOME = "Any other home victory"
TAIL_DRAW = "Any other draw"
TAIL_AWAY = "Any other away victory"
TAIL_KEY_HOME = "H_OTHER"
TAIL_KEY_DRAW = "D_OTHER"
TAIL_KEY_AWAY = "A_OTHER"


def result_category(score: Score) -> int:
    """Map a scoreline to outcome category: 0=home, 1=draw, 2=away."""
    if score[0] > score[1]:
        return 0
    if score[0] == score[1]:
        return 1
    return 2


def tail_label(category: int) -> str:
    """Return a human-readable label for a tail outcome category."""
    if category == 0:
        return TAIL_HOME
    if category == 1:
        return TAIL_DRAW
    return TAIL_AWAY


@dataclass(frozen=True)
class MarketOdds:
    """Input market odds required by the scoreline calibration pipeline."""
    match_odds: Sequence[float]
    over_under_25: Sequence[float]
    btts: Sequence[float]
    correct_score: Sequence[Tuple[object, float]]

def score_grid(max_goals: int = MAX_EXACT_SCORE) -> List[str]:
    """Return canonical exact-score labels plus tail buckets."""
    labels = [score_label((home_goals, away_goals)) for home_goals in range(max_goals + 1) for away_goals in range(max_goals + 1)]
    labels.extend([TAIL_HOME, TAIL_DRAW, TAIL_AWAY])
    return labels


def score_label(score: Score, max_exact_score: int = MAX_EXACT_SCORE) -> str:
    """Format a scoreline label, collapsing beyond-grid scores to tail labels."""
    if score[0] <= max_exact_score and score[1] <= max_exact_score:
        return f"{score[0]}-{score[1]}"
    return tail_label(result_category(score))


def exact_score_key(value: object) -> object:
    """Normalize exact-score keys from tuple-like values to integer tuples."""
    if isinstance(value, tuple) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return value


def tail_market_category(value: object) -> int | None:
    """Map market tail tokens to outcome categories."""
    if value == TAIL_KEY_HOME:
        return 0
    if value == TAIL_KEY_DRAW:
        return 1
    if value == TAIL_KEY_AWAY:
        return 2
    return None


def devig_exact_score_market(quoted_odds: Sequence[Tuple[object, float]]) -> Dict[object, float]:
    """Apply Shin de-vigging to all quoted exact-score outcomes."""
    keys = [exact_score_key(key) for key, _ in quoted_odds]
    odds = [float(odds_value) for _, odds_value in quoted_odds]
    probs, _ = shin_devig(odds)
    return {key: float(prob) for key, prob in zip(keys, probs)}


def poisson_pmf_vector(lam: float, max_goals: int) -> np.ndarray:
    """Build a truncated Poisson PMF and renormalize the finite support."""
    values = np.array([exp(-lam) * (lam**goal) / factorial(goal) for goal in range(max_goals + 1)], dtype=float)
    return normalize(values)


def poisson_joint_grid(lambda_home: float, lambda_away: float, max_goals: int) -> np.ndarray:
    """Create the independent home/away Poisson joint score grid."""
    home_pmf = poisson_pmf_vector(lambda_home, max_goals)
    away_pmf = poisson_pmf_vector(lambda_away, max_goals)
    joint = np.outer(home_pmf, away_pmf)
    return normalize(joint.reshape(-1)).reshape(joint.shape)


def model_marginals(lambda_home: float, lambda_away: float, max_goals: int) -> Dict[str, np.ndarray]:
    """Derive 1X2, goal-band, and BTTS marginals from a score grid."""
    joint = poisson_joint_grid(lambda_home, lambda_away, max_goals)
    outcome = np.zeros(3, dtype=float)
    band = np.zeros(2, dtype=float)
    btts = np.zeros(2, dtype=float)

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = float(joint[home_goals, away_goals])
            if home_goals > away_goals:
                outcome[0] += probability
            elif home_goals == away_goals:
                outcome[1] += probability
            else:
                outcome[2] += probability

            total_goals = home_goals + away_goals
            if total_goals <= 2:
                band[0] += probability
            else:
                band[1] += probability

            if home_goals > 0 and away_goals > 0:
                btts[0] += probability
            else:
                btts[1] += probability

    return {
        "joint": joint,
        "outcome": normalize(outcome),
        "band": normalize(band),
        "btts": normalize(btts),
    }


def fit_goal_intensities(
    target_outcome: np.ndarray,
    target_band: np.ndarray,
    target_btts: np.ndarray,
    max_goals: int,
) -> Tuple[float, float]:
    """Fit Poisson intensities to market-implied marginals via least squares."""
    def objective(params: np.ndarray) -> float:
        lambda_home, lambda_away = float(params[0]), float(params[1])
        if lambda_home <= 0.0 or lambda_away <= 0.0:
            return 1e9

        model = model_marginals(lambda_home, lambda_away, max_goals)
        error = 0.0
        error += 4.0 * float(np.sum((model["outcome"] - target_outcome) ** 2))
        error += 2.0 * float(np.sum((model["band"] - target_band) ** 2))
        error += 2.0 * float(np.sum((model["btts"] - target_btts) ** 2))
        return error

    initial_total = 2.4
    try:
        over_25 = float(target_band[1])
        if 0.0 < over_25 < 0.999:
            solution = root_scalar(
                lambda lam: 1.0 - exp(-lam) * (1.0 + lam + 0.5 * lam * lam) - over_25,
                bracket=[0.1, 8.0],
                method="brentq",
            )
            initial_total = float(solution.root)
    except Exception:
        pass

    initial = np.array([0.55 * initial_total, 0.45 * initial_total], dtype=float)
    result = minimize(objective, initial, method="L-BFGS-B", bounds=[(0.05, 6.0), (0.05, 6.0)])
    if not result.success:
        return float(initial[0]), float(initial[1])
    return float(result.x[0]), float(result.x[1])


def build_exact_score_prior(
    base_joint: np.ndarray,
    quoted_market: Mapping[object, float],
    max_goals: int,
    blend: float = 0.65,
) -> np.ndarray:
    """Blend the market exact-score quotes with the Poisson background.

    Quoted scorelines keep their market anchor. The remaining mass from the
    correct-score book is distributed across unquoted states proportional to the
    Poisson background.
    """

    prior = np.array(base_joint, dtype=float)
    market_grid = np.zeros_like(prior)
    quoted_mask = np.zeros_like(prior, dtype=bool)
    tail_mass = np.zeros(3, dtype=float)

    for key, probability in quoted_market.items():
        score = exact_score_key(key)
        tail_category = tail_market_category(score)
        if tail_category is not None:
            tail_mass[tail_category] += float(probability)
            continue
        if not isinstance(score, tuple) or len(score) != 2:
            continue
        home_goals, away_goals = int(score[0]), int(score[1])
        if 0 <= home_goals <= max_goals and 0 <= away_goals <= max_goals:
            if home_goals <= MAX_EXACT_SCORE and away_goals <= MAX_EXACT_SCORE:
                market_grid[home_goals, away_goals] = float(probability)
                quoted_mask[home_goals, away_goals] = True
            else:
                tail_mass[result_category((home_goals, away_goals))] += float(probability)

    unquoted_mask = ~quoted_mask
    # Use the Poisson background for quoted-market gaps to keep a full support grid.
    market_grid[unquoted_mask] = prior[unquoted_mask]

    for category, mass in enumerate(tail_mass):
        if mass <= 0.0:
            continue
        if category == 0:
            mask = np.zeros_like(prior, dtype=bool)
            for home_goals in range(max_goals + 1):
                for away_goals in range(max_goals + 1):
                    # Tail buckets only cover states outside the explicit exact-score board.
                    mask[home_goals, away_goals] = home_goals > MAX_EXACT_SCORE and away_goals > MAX_EXACT_SCORE and home_goals > away_goals
        elif category == 1:
            mask = np.zeros_like(prior, dtype=bool)
            for home_goals in range(max_goals + 1):
                for away_goals in range(max_goals + 1):
                    mask[home_goals, away_goals] = home_goals > MAX_EXACT_SCORE and away_goals > MAX_EXACT_SCORE and home_goals == away_goals
        else:
            mask = np.zeros_like(prior, dtype=bool)
            for home_goals in range(max_goals + 1):
                for away_goals in range(max_goals + 1):
                    mask[home_goals, away_goals] = home_goals > MAX_EXACT_SCORE and away_goals > MAX_EXACT_SCORE and home_goals < away_goals

        tail_background = prior[mask]
        if float(np.sum(tail_background)) > 0.0:
            market_grid[mask] += mass * tail_background / float(np.sum(tail_background))

    market_grid = normalize(market_grid.reshape(-1)).reshape(prior.shape)
    prior = normalize((1.0 - blend) * prior.reshape(-1) + blend * market_grid.reshape(-1)).reshape(prior.shape)
    return prior


def calibrate_distribution(
    prior: np.ndarray,
    target_outcome: np.ndarray,
    target_band: np.ndarray,
    target_btts: np.ndarray,
    max_iterations: int = 4000,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Calibrate the prior grid to match target marginals with IPF updates."""
    calibrated = np.array(prior, dtype=float)

    outcome_index = np.zeros_like(calibrated, dtype=int)
    band_index = np.zeros_like(calibrated, dtype=int)
    btts_index = np.zeros_like(calibrated, dtype=int)

    max_goals = calibrated.shape[0] - 1
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            if home_goals > away_goals:
                outcome_index[home_goals, away_goals] = 0
            elif home_goals == away_goals:
                outcome_index[home_goals, away_goals] = 1
            else:
                outcome_index[home_goals, away_goals] = 2

            total_goals = home_goals + away_goals
            if total_goals <= 2:
                band_index[home_goals, away_goals] = 0
            else:
                band_index[home_goals, away_goals] = 1

            btts_index[home_goals, away_goals] = 0 if home_goals > 0 and away_goals > 0 else 1

    flat = calibrated.reshape(-1)
    outcome_flat = outcome_index.reshape(-1)
    band_flat = band_index.reshape(-1)
    btts_flat = btts_index.reshape(-1)

    for _ in range(max_iterations):
        previous = flat.copy()

        for category, target in enumerate(target_outcome):
            mask = outcome_flat == category
            mass = float(np.sum(flat[mask]))
            if mass > 0.0:
                flat[mask] *= float(target) / mass

        for category, target in enumerate(target_band):
            mask = band_flat == category
            mass = float(np.sum(flat[mask]))
            if mass > 0.0:
                flat[mask] *= float(target) / mass

        for category, target in enumerate(target_btts):
            mask = btts_flat == category
            mass = float(np.sum(flat[mask]))
            if mass > 0.0:
                flat[mask] *= float(target) / mass

        # Renormalize after each full IPF sweep to avoid floating-point drift.
        flat = normalize(flat)
        if float(np.max(np.abs(flat - previous))) < tolerance:
            break

    return flat.reshape(calibrated.shape)


def extract_market_targets(market: MarketOdds) -> Dict[str, np.ndarray]:
    match_odds, _ = shin_devig(market.match_odds)
    over_25, _ = shin_devig(market.over_under_25)
    btts_probs, _ = shin_devig(market.btts)

    target_outcome = match_odds.astype(float)
    target_band = np.array([float(over_25[1]), float(over_25[0])], dtype=float)
    target_btts = np.array([float(btts_probs[0]), float(btts_probs[1])], dtype=float)

    return {
        "outcome": normalize(target_outcome),
        "band": normalize(target_band),
        "btts": normalize(target_btts),
    }


def score_distribution(
    market: MarketOdds,
    max_goals: int = 10,
) -> Dict[str, object]:
    targets = extract_market_targets(market)
    lambda_home, lambda_away = fit_goal_intensities(targets["outcome"], targets["band"], targets["btts"], max_goals)
    background = model_marginals(lambda_home, lambda_away, max_goals)["joint"]

    exact_market = devig_exact_score_market(market.correct_score)
    prior = build_exact_score_prior(background, exact_market, max_goals=max_goals, blend=0.65)
    final = calibrate_distribution(prior, targets["outcome"], targets["band"], targets["btts"])

    score_mass_by_label: Dict[str, float] = {}
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            label = score_label((home_goals, away_goals))
            score_mass_by_label[label] = score_mass_by_label.get(label, 0.0) + float(final[home_goals, away_goals])
    flat_scores = sorted(score_mass_by_label.items(), key=lambda item: item[1], reverse=True)

    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "targets": targets,
        "prior": prior,
        "final": final,
        "ranked_scores": flat_scores,
    }


def reconstruct_marginals(distribution: np.ndarray) -> Dict[str, np.ndarray]:
    max_goals = distribution.shape[0] - 1
    outcome = np.zeros(3, dtype=float)
    band = np.zeros(2, dtype=float)
    btts = np.zeros(2, dtype=float)

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = float(distribution[home_goals, away_goals])
            if home_goals > away_goals:
                outcome[0] += probability
            elif home_goals == away_goals:
                outcome[1] += probability
            else:
                outcome[2] += probability

            total_goals = home_goals + away_goals
            if total_goals <= 2:
                band[0] += probability
            else:
                band[1] += probability

            if home_goals > 0 and away_goals > 0:
                btts[0] += probability
            else:
                btts[1] += probability

    return {
        "outcome": normalize(outcome),
        "band": normalize(band),
        "btts": normalize(btts),
    }


def sample_scoreline(distribution: np.ndarray, rng: np.random.Generator | None = None) -> str:
    if rng is None:
        rng = np.random.default_rng()
    flat = distribution.reshape(-1)
    index = int(rng.choice(flat.size, p=flat))
    max_goals = distribution.shape[0]
    home_goals = index // max_goals
    away_goals = index % max_goals
    return score_label((home_goals, away_goals))


def format_percentage(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_top_scores(ranked_scores: Sequence[Tuple[str, float]], top_n: int = 10) -> None:
    print("\nTop 10 scorelines")
    print("-----------------")
    for label, probability in ranked_scores[:top_n]:
        print(f"{label:<28}  {format_percentage(probability)}")


def print_marginal_check(name: str, target: np.ndarray, actual: np.ndarray) -> None:
    print(f"{name:<10} target={target.round(6)}  actual={actual.round(6)}")


def plot_score_grid(distribution: np.ndarray, output_path: Path) -> Path:
    exact_limit = MAX_EXACT_SCORE
    exact_grid = np.zeros((exact_limit + 1, exact_limit + 1), dtype=float)
    for home_goals in range(exact_limit + 1):
        for away_goals in range(exact_limit + 1):
            exact_grid[home_goals, away_goals] = float(distribution[home_goals, away_goals])

    tail_summary = {
        TAIL_HOME: 0.0,
        TAIL_DRAW: 0.0,
        TAIL_AWAY: 0.0,
    }
    max_goals = distribution.shape[0] - 1
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            if home_goals <= exact_limit and away_goals <= exact_limit:
                continue
            label = tail_label(result_category((home_goals, away_goals)))
            tail_summary[label] += float(distribution[home_goals, away_goals])

    fig, (ax_grid, ax_tail) = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [4, 1.3]},
    )

    heatmap = ax_grid.imshow(exact_grid, cmap="viridis", origin="upper")
    fig.colorbar(heatmap, ax=ax_grid, fraction=0.046, pad=0.04, label="Probability")

    ax_grid.set_xticks(range(exact_limit + 1))
    ax_grid.set_yticks(range(exact_limit + 1))
    ax_grid.set_xticklabels(range(exact_limit + 1))
    ax_grid.set_yticklabels(range(exact_limit + 1))
    ax_grid.set_xlabel("Away goals")
    ax_grid.set_ylabel("Home goals")
    ax_grid.set_title("Exact score grid (0-0 to 3-3)")

    max_exact = float(np.max(exact_grid)) if float(np.max(exact_grid)) > 0.0 else 1.0
    for home_goals in range(exact_limit + 1):
        for away_goals in range(exact_limit + 1):
            probability = float(exact_grid[home_goals, away_goals])
            text_color = "white" if probability >= max_exact * 0.45 else "black"
            ax_grid.text(
                away_goals,
                home_goals,
                f"{home_goals}-{away_goals}\n{probability:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    ax_tail.axis("off")
    tail_lines = [
        "Tail buckets",
        "",
        f"H_OTHER\n{TAIL_HOME}\n{tail_summary[TAIL_HOME]:.3%}",
        f"D_OTHER\n{TAIL_DRAW}\n{tail_summary[TAIL_DRAW]:.3%}",
        f"A_OTHER\n{TAIL_AWAY}\n{tail_summary[TAIL_AWAY]:.3%}",
    ]
    ax_tail.text(0.0, 1.0, "\n\n".join(tail_lines), va="top", ha="left", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def build_demo_market() -> MarketOdds:
    return MarketOdds(
        match_odds=[1.51, 7.4, 4.5],
        over_under_25=[1.83, 2.08],
        btts=[2.34, 1.7],
        correct_score=[
            ((0, 0), 12.5),
            ((1, 0), 6.0),
            ((0, 1), 17.0),
            ((1, 1), 9.4),
            ((2, 0), 6.8),
            ((0, 2), 6.0),
            ((2, 1), 9.8),
            ((1, 2), 26.0),
            ((2, 2), 6.6),
            ((3, 0), 12.0),
            ((0, 3), 6.2),
            ((3, 1), 17.0),
            ((1, 3), 6.2),
            ((3, 2), 6.0),
            ((2, 3), 6.0),
            (TAIL_KEY_HOME, 7.6),
            (TAIL_KEY_DRAW, 6.2),
            (TAIL_KEY_AWAY, 6.2),
        ],
    )


def main() -> None:
    market = build_demo_market()
    result = score_distribution(market, max_goals=10)
    final = result["final"]
    targets = result["targets"]
    reconstructed = reconstruct_marginals(final)
    plot_path = Path(__file__).with_name("BetfairBayesianScoreline_grid.png")
    plot_score_grid(final, plot_path)

    print("Fitted latent goal intensities")
    print("------------------------------")
    print(f"Home lambda: {result['lambda_home']:.4f}")
    print(f"Away lambda: {result['lambda_away']:.4f}")
    print(f"Total mass : {float(final.sum()):.12f}")

    print("\nMarginal check")
    print("--------------")
    print_marginal_check("1X2", targets["outcome"], reconstructed["outcome"])
    print_marginal_check("O/U", targets["band"], reconstructed["band"])
    print_marginal_check("BTTS", targets["btts"], reconstructed["btts"])

    print_top_scores(result["ranked_scores"], top_n=10)
    print(f"\nScore grid plot saved to: {plot_path}")

    rng = np.random.default_rng()
    picked = sample_scoreline(final, rng=rng)
    print("\nMonte Carlo pick")
    print("----------------")
    print(f"Picked scoreline: {picked}")


if __name__ == "__main__":
    main()