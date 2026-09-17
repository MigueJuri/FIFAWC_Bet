"""Compare de-vigging methods on 1X2 vs exact-score consistency.

The script removes bookmaker margin from:
- A liquid 1X2 market (baseline)
- A heavier-margin exact-score market (reconstructed to 1X2)

It then reports KL divergence to show which de-vigging method better preserves
the 1X2 distribution when applied to exact-score prices.
"""

import numpy as np
from scipy.stats import entropy

from odds_devig import power_devig, shin_devig


def recover_1x2_from_exact(
    exact_probs: np.ndarray,
    home_len: int,
    draw_len: int,
) -> list[float]:
    """Aggregate exact-score probabilities back into Home/Draw/Away buckets."""
    home_prob = np.sum(exact_probs[:home_len])
    draw_prob = np.sum(exact_probs[home_len : home_len + draw_len])
    away_prob = np.sum(exact_probs[home_len + draw_len :])
    return [float(home_prob), float(draw_prob), float(away_prob)]


def main() -> None:
    """Run the method comparison and print formatted results."""
    # Input: highly liquid 1X2 odds [Home, Draw, Away].
    odds_1x2 = [1.71, 5.0, 4.3] 
    
    # Input: illiquid exact-score odds grouped by final outcome.
    # The list is exhaustive by using "Any Other ..." buckets.
    exact_odds = {
        "Home": [11.0, 11.0, 9.4, 16.5, 13.5, 18.5, 6.8],  # 1-0, 2-0, 2-1, 3-0, 3-1, 3-2, Any Other Home
        "Draw": [20.0, 9.8, 15.0, 55, 95.0],  # 0-0, 1-1, 2-2, Any Other Draw
        "Away": [22.0, 38.0, 18.0, 40.0, 28.0, 46.0, 27.0],  # 0-1, 0-2, 1-2, 0-3, 1-3, 2-3, Any Other Away
    }
    
    # Flatten exact odds to de-vig them all simultaneously
    flat_exact_odds = exact_odds["Home"] + exact_odds["Draw"] + exact_odds["Away"]
    
    # Step A: de-vig the 1X2 market (baseline distribution).
    p_1x2_power, _ = power_devig(odds_1x2)
    p_1x2_shin, _ = shin_devig(odds_1x2)
    
    # Step B: de-vig the exact-score market.
    p_exact_power, _ = power_devig(flat_exact_odds)
    p_exact_shin, _ = shin_devig(flat_exact_odds)
    
    # Step C: recover Home/Draw/Away probabilities from exact scores.
    h_len = len(exact_odds["Home"])
    d_len = len(exact_odds["Draw"])
    q_1x2_power = recover_1x2_from_exact(p_exact_power, h_len, d_len)
    q_1x2_shin = recover_1x2_from_exact(p_exact_shin, h_len, d_len)
    
    # Step D: measure KL divergence, D_KL(P || Q).
    kl_power = entropy(p_1x2_power, q_1x2_power)
    kl_shin = entropy(p_1x2_shin, q_1x2_shin)
    
    # Output
    print(f"{'Distribution':<14} | {'Method':<6} | {'Home':<6} | {'Draw':<6} | {'Away':<6}")
    print("-" * 47)
    print(f"{'Baseline (1X2)':<14} | {'Power':<6} | {p_1x2_power[0]:.4f} | {p_1x2_power[1]:.4f} | {p_1x2_power[2]:.4f}")
    print(f"{'Baseline (1X2)':<14} | {'Shin':<6} | {p_1x2_shin[0]:.4f} | {p_1x2_shin[1]:.4f} | {p_1x2_shin[2]:.4f}")
    print("-" * 47)
    print(f"{'Recovered':<14} | {'Power':<6} | {q_1x2_power[0]:.4f} | {q_1x2_power[1]:.4f} | {q_1x2_power[2]:.4f}")
    print(f"{'Recovered':<14} | {'Shin':<6} | {q_1x2_shin[0]:.4f} | {q_1x2_shin[1]:.4f} | {q_1x2_shin[2]:.4f}\n")
    
    print(f"KL Divergence (Power): {kl_power:.6f}")
    print(f"KL Divergence (Shin):  {kl_shin:.6f}")
    print("\nResult: The method with the lower KL divergence maps the heavy-margin")
    print("exact score market back to the liquid 1X2 baseline more accurately.")


if __name__ == "__main__":
    main()