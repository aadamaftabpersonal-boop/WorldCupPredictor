"""
Team-level statistics derived from clean match history.

All functions take a cutoff date so we never leak future data.
"""

import pandas as pd
import numpy as np


def get_team_matches(df: pd.DataFrame,
                     team: str,
                     before_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    All matches for a team, standardised so the team is always
    in the 'team' column regardless of home/away.
    Optionally filtered to before a cutoff date.
    """
    if before_date is not None:
        df = df[df["date"] < before_date]

    home = df[df["home_team"] == team].copy()
    home["team"]          = home["home_team"]
    home["opponent"]      = home["away_team"]
    home["goals_for"]     = home["home_score"]
    home["goals_against"] = home["away_score"]
    home["is_home"]       = True
    home["result"]        = home.apply(
        lambda r: "W" if r["home_score"] > r["away_score"]
        else ("L" if r["home_score"] < r["away_score"] else "D"), axis=1
    )

    away = df[df["away_team"] == team].copy()
    away["team"]          = away["away_team"]
    away["opponent"]      = away["home_team"]
    away["goals_for"]     = away["away_score"]
    away["goals_against"] = away["home_score"]
    away["is_home"]       = False
    away["result"]        = away.apply(
        lambda r: "W" if r["away_score"] > r["home_score"]
        else ("L" if r["away_score"] < r["home_score"] else "D"), axis=1
    )

    cols = ["date", "team", "opponent", "goals_for",
            "goals_against", "is_home", "result", "tournament", "neutral"]
    combined = pd.concat([home[cols], away[cols]]).sort_values("date")
    return combined.reset_index(drop=True)


def compute_form(team_matches: pd.DataFrame, n: int = 10) -> dict:
    """
    Rolling form over last N matches.

    Returns:
        attack_strength  : avg goals scored
        defence_strength : avg goals conceded (lower = better)
        win_rate         : fraction of wins
        form_score       : normalised points (W=3 D=1 L=0), range 0-1
        clean_sheet_rate : fraction with 0 goals conceded
        avg_goal_diff    : avg (scored - conceded)
        scoring_rate     : fraction of matches where team scored >= 1
    """
    recent = team_matches.tail(n)

    if len(recent) == 0:
        return {
            "attack_strength":  1.2,
            "defence_strength": 1.2,
            "win_rate":         0.33,
            "form_score":       0.33,
            "clean_sheet_rate": 0.25,
            "avg_goal_diff":    0.0,
            "scoring_rate":     0.65,
            "matches_used":     0,
        }

    wins  = (recent["result"] == "W").sum()
    draws = (recent["result"] == "D").sum()

    return {
        "attack_strength":  recent["goals_for"].mean(),
        "defence_strength": recent["goals_against"].mean(),
        "win_rate":         wins / len(recent),
        "form_score":       (wins * 3 + draws) / (len(recent) * 3),
        "clean_sheet_rate": (recent["goals_against"] == 0).mean(),
        "avg_goal_diff":    (recent["goals_for"] - recent["goals_against"]).mean(),
        "scoring_rate":     (recent["goals_for"] >= 1).mean(),
        "matches_used":     len(recent),
    }


def compute_h2h(df: pd.DataFrame,
                team_a: str,
                team_b: str,
                before_date: pd.Timestamp = None,
                n: int = 10) -> dict:
    """
    Head-to-head record between two teams.
    Returns win rate, avg goals, and number of past meetings.
    """
    if before_date is not None:
        df = df[df["date"] < before_date]

    mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
        ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    )
    h2h = df[mask].sort_values("date").tail(n)

    if len(h2h) == 0:
        return {
            "h2h_win_rate_a": 0.33,
            "h2h_avg_goals_a": 1.2,
            "h2h_avg_goals_b": 1.2,
            "h2h_meetings":    0,
        }

    wins_a, goals_a, goals_b = 0, [], []

    for _, row in h2h.iterrows():
        if row["home_team"] == team_a:
            ga, gb = row["home_score"], row["away_score"]
        else:
            ga, gb = row["away_score"], row["home_score"]
        goals_a.append(ga)
        goals_b.append(gb)
        if ga > gb:
            wins_a += 1

    return {
        "h2h_win_rate_a":  wins_a / len(h2h),
        "h2h_avg_goals_a": np.mean(goals_a),
        "h2h_avg_goals_b": np.mean(goals_b),
        "h2h_meetings":    len(h2h),
    }


def compute_tournament_experience(team_matches: pd.DataFrame) -> dict:
    """
    How experienced is a team in high-pressure tournaments?
    Counts past World Cup + continental championship appearances.
    """
    wc_matches = team_matches[
        team_matches["tournament"].str.contains("World Cup", na=False)
    ]
    cont_matches = team_matches[
        team_matches["tournament"].str.contains(
            "UEFA Euro|Copa América|Africa Cup|Asian Cup|Gold Cup", na=False
        )
    ]

    wc_win_rate = 0.33
    if len(wc_matches) > 0:
        wc_wins    = (wc_matches["result"] == "W").sum()
        wc_win_rate = wc_wins / len(wc_matches)

    return {
        "wc_matches":       len(wc_matches),
        "wc_win_rate":      wc_win_rate,
        "cont_matches":     len(cont_matches),
        "total_experience": len(wc_matches) + len(cont_matches),
    }