import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import entropy

def de_vig_power(odds):
    """
    Applies the Power Method to remove the bookmaker margin.
    Finds k such that sum( (1/odds)^k ) = 1
    """
    odds = np.array(odds)
    implied = 1.0 / odds
    
    def objective(k):
        return np.sum(implied**k) - 1.0
        
    sol = root_scalar(objective, bracket=[1.0, 10.0], method='brentq')
    k = sol.root
    true_probs = implied**k
    return true_probs, k

def de_vig_shin(odds):
    """
    Applies Shin's Method to remove the bookmaker margin.
    Finds z (proportion of insider money) such that sum(p_i) = 1
    """
    odds = np.array(odds)
    implied = 1.0 / odds
    implied_sum = np.sum(implied)
    
    def shin_probs(z):
        return (np.sqrt(z**2 + 4 * (1 - z) * (implied**2 / implied_sum)) - z) / (2 * (1 - z))

    def objective(z):
        return np.sum(shin_probs(z)) - 1.0

    lower = 0.0
    upper = 1.0 - 1e-8
    f_lower = objective(lower)
    f_upper = objective(upper)

    if f_lower * f_upper > 0:
        true_probs = implied / implied_sum
        return true_probs, 0.0

    sol = root_scalar(objective, bracket=[lower, upper], method='brentq')
    z = sol.root
    true_probs = shin_probs(z)
    return true_probs, z

def main():
    # 1. Input: Highly liquid 1X2 Odds [Home, Draw, Away]
    odds_1x2 = [1.71, 5.0, 4.3] 
    
    # 2. Input: Illiquid Exact Score Odds (categorized by outcome)
    # The list must be exhaustive. Real markets use "Any Other Home/Away/Draw" to catch the rest.
    exact_odds = {
        'Home': [11.0, 11.0, 9.4, 16.5, 13.5, 18.5, 6.8], # 1-0, 2-0, 2-1, 3-0, 3-1, 3-2, Any Other Home
        'Draw': [20.0, 9.8, 15.0, 55, 95.0],                    # 0-0, 1-1, 2-2, Any Other Draw
        'Away': [22.0, 38.0, 18.0, 40.0, 28.0, 46.0, 27.0]  # 0-1, 0-2, 1-2, 0-3, 1-3, 2-3, Any Other Away
    }
    
    # Flatten exact odds to de-vig them all simultaneously
    flat_exact_odds = exact_odds['Home'] + exact_odds['Draw'] + exact_odds['Away']
    
    # --- Step A: De-vig the 1X2 Market (Our Baseline 'True' Distribution) ---
    p_1x2_power, _ = de_vig_power(odds_1x2)
    p_1x2_shin, _  = de_vig_shin(odds_1x2)
    
    # --- Step B: De-vig the Exact Score Market ---
    p_exact_power, _ = de_vig_power(flat_exact_odds)
    p_exact_shin, _  = de_vig_shin(flat_exact_odds)
    
    # --- Step C: Recover the 1X2 probabilities from Exact Scores ---
    # Determine the slice indices for Home, Draw, Away
    h_len = len(exact_odds['Home'])
    d_len = len(exact_odds['Draw'])
    
    def recover_1x2(exact_probs):
        home_prob = np.sum(exact_probs[:h_len])
        draw_prob = np.sum(exact_probs[h_len : h_len + d_len])
        away_prob = np.sum(exact_probs[h_len + d_len:])
        return [home_prob, draw_prob, away_prob]

    q_1x2_power = recover_1x2(p_exact_power)
    q_1x2_shin  = recover_1x2(p_exact_shin)
    
    # --- Step D: Measure the KL Divergence ---
    # D_KL(P || Q) measures how much information is lost when Q is used to approximate P.
    kl_power = entropy(p_1x2_power, q_1x2_power)
    kl_shin  = entropy(p_1x2_shin, q_1x2_shin)
    
    # --- Output Results ---
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