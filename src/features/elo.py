"""
ELO Rating System for international football.

How ELO works:
  - Every team starts at 1500 points
  - Win vs strong team  → big gain
  - Win vs weak team    → small gain
  - Lose vs weak team   → big loss
  - Lose vs strong team → small loss
  
K-factor controls how fast ratings change.
We use K=32 for World Cup matches, K=20 for friendlies.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# Starting ELO for every team
default_elo = 1500

# How much ratings shift per match (by tournament importance)
k_factors = {
    "FIFA World Cup":          40,
    "UEFA Euro":               35,
    "Copa América":            35,
    "Africa Cup of Nations":   30,
    "AFC Asian Cup":           30,
    "CONCACAF Gold Cup":       28,
    "UEFA Nations League":     25,
    "FIFA World Cup qualification": 25,
    "Friendly":                15,
}
default_k = 20


def get_k_factor(tournament: str) -> float:
    """Return the K-factor for a given tournament name."""
    for key, k in k_factors.items():
        if key.lower() in tournament.lower():
            return k
    return default_k

def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Probability that team A beats team B given their ELO ratings.
    Formula: 1 / (1 + 10^((B - A) / 400))
    
    Example:
        A=1600, B=1400 → expected = 0.76  (A is 76% likely to win)
        A=1500, B=1500 → expected = 0.50  (50/50)
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

def actual_score(home_goals: int, away_goals: int, neutral: bool) -> tuple:
    """
    Convert match result to ELO score values.
    Win=1.0, Draw=0.5, Loss=0.0
    Home advantage adds a 100-point ELO bonus if NOT neutral venue.
    Returns (home_actual, away_actual, home_advantage_bonus)
    """
    home_advantage = 0 if neutral else 100

    if home_goals > away_goals:
        return 1.0, 0.0, home_advantage
    elif home_goals < away_goals:
        return 0.0, 1.0, home_advantage
    else:
        return 0.5, 0.5, home_advantage
