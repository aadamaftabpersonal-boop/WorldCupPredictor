"""
ELO Rating System for international football.

How ELO works:
  - Every team starts at 1500 points
  - Win vs strong team  → big gain
  - Win vs weak team    → small gain
  - Lose vs weak team   → big loss
  - Lose vs strong team → small loss
"""

import pandas as pd
import numpy as np
from collections import defaultdict

DEFAULT_ELO = 1500

K_FACTORS = {
    "FIFA World Cup":               40,
    "UEFA Euro":                    35,
    "Copa América":                 35,
    "Africa Cup of Nations":        30,
    "AFC Asian Cup":                30,
    "CONCACAF Gold Cup":            28,
    "UEFA Nations League":          25,
    "FIFA World Cup qualification": 25,
}
DEFAULT_K = 20


def get_k_factor(tournament: str) -> float:
    for key, k in K_FACTORS.items():
        if key.lower() in tournament.lower():
            return k
    return DEFAULT_K


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B. Formula: 1 / (1 + 10^((B-A)/400))"""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_elo_ratings(results_df: pd.DataFrame):
    """
    Walk every match chronologically and update ELO ratings.
    Returns:
        ratings     : dict  {team → final ELO}
        history_df  : DataFrame with before/after ELO per match
    """
    df = results_df.sort_values("date").reset_index(drop=True)
    ratings = defaultdict(lambda: DEFAULT_ELO)
    history = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        k    = get_k_factor(row["tournament"])

        home_elo = ratings[home]
        away_elo = ratings[away]

        # Home advantage: +100 ELO points for expected score calc only
        home_advantage = 0 if row["neutral"] else 100
        exp_home = expected_score(home_elo + home_advantage, away_elo)
        exp_away = 1.0 - exp_home

        # Actual outcome
        if row["home_score"] > row["away_score"]:
            act_home, act_away = 1.0, 0.0
        elif row["home_score"] < row["away_score"]:
            act_home, act_away = 0.0, 1.0
        else:
            act_home, act_away = 0.5, 0.5

        new_home = home_elo + k * (act_home - exp_home)
        new_away = away_elo + k * (act_away - exp_away)

        ratings[home] = new_home
        ratings[away] = new_away

        history.append({
            "date":            row["date"],
            "home_team":       home,
            "away_team":       away,
            "home_elo_before": home_elo,
            "away_elo_before": away_elo,
            "home_elo_after":  new_home,
            "away_elo_after":  new_away,
        })

    return dict(ratings), pd.DataFrame(history)


def get_elo_before_date(elo_history: pd.DataFrame,
                        team: str,
                        before_date: pd.Timestamp) -> float:
    """
    Look up a team's ELO rating just before a given date.
    Used during feature building to prevent data leakage.
    """
    past = elo_history[elo_history["date"] < before_date]

    home_rows = past[past["home_team"] == team]
    away_rows = past[past["away_team"] == team]

    last_home = home_rows["home_elo_after"].iloc[-1] if len(home_rows) else None
    last_away = away_rows["away_elo_after"].iloc[-1] if len(away_rows) else None

    # Pick the most recent of the two
    if last_home is None and last_away is None:
        return DEFAULT_ELO
    if last_home is None:
        return last_away
    if last_away is None:
        return last_home

    last_home_date = home_rows["date"].iloc[-1]
    last_away_date = away_rows["date"].iloc[-1]

    return last_home if last_home_date > last_away_date else last_away