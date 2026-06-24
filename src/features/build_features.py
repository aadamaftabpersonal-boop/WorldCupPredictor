"""
Build the final feature matrix for model training.

Each row   = one match
Features   = differences between team stats (home minus away)
Label      = 0 (away win), 1 (draw), 2 (home win)

Key design decision — WHY differences?
  Raw ELO of 1800 vs 1600 means less than the GAP of 200.
  Difference features let the model focus on relative strength,
  which generalises better to unseen team matchups.
"""

import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

from src.features.elo import compute_elo_ratings, get_elo_before_date
from src.features.team_stats import (
    get_team_matches,
    compute_form,
    compute_h2h,
    compute_tournament_experience,
)

PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

FORM_WINDOW = 10   # last N matches for rolling stats


def build_row_features(row: pd.Series,
                       all_matches: pd.DataFrame,
                       elo_history: pd.DataFrame) -> dict:
    """
    Build one feature vector for one match.
    Everything is computed using only data BEFORE row["date"].
    """
    home      = row["home_team"]
    away      = row["away_team"]
    match_date = row["date"]

    # ── ELO ──────────────────────────────────────────────────────────────
    home_elo = get_elo_before_date(elo_history, home, match_date)
    away_elo = get_elo_before_date(elo_history, away, match_date)

    # ── Form stats ────────────────────────────────────────────────────────
    home_tm   = get_team_matches(all_matches, home, before_date=match_date)
    away_tm   = get_team_matches(all_matches, away, before_date=match_date)

    home_form = compute_form(home_tm, n=FORM_WINDOW)
    away_form = compute_form(away_tm, n=FORM_WINDOW)

    # ── Tournament experience ─────────────────────────────────────────────
    home_exp  = compute_tournament_experience(home_tm)
    away_exp  = compute_tournament_experience(away_tm)

    # ── Head to head ──────────────────────────────────────────────────────
    h2h = compute_h2h(all_matches, home, away,
                      before_date=match_date, n=10)

    # ── Confederation strength proxy ──────────────────────────────────────
    # Encoded as ordinal — UEFA/CONMEBOL historically strongest
    CONF_STRENGTH = {
        "UEFA": 6, "CONMEBOL": 5, "CONCACAF": 4,
        "AFC": 3, "CAF": 3, "OFC": 1,
    }

    # ── Assemble feature dict ─────────────────────────────────────────────
    features = {
        # Raw ELO (model can use absolute values too)
        "home_elo":                 home_elo,
        "away_elo":                 away_elo,
        "elo_diff":                 home_elo - away_elo,

        # Attack / defence
        "attack_diff":              home_form["attack_strength"]  - away_form["attack_strength"],
        "defence_diff":             away_form["defence_strength"] - home_form["defence_strength"],
        "home_attack":              home_form["attack_strength"],
        "away_attack":              away_form["attack_strength"],
        "home_defence":             home_form["defence_strength"],
        "away_defence":             away_form["defence_strength"],

        # Form
        "form_diff":                home_form["form_score"]    - away_form["form_score"],
        "win_rate_diff":            home_form["win_rate"]      - away_form["win_rate"],
        "goal_diff_diff":           home_form["avg_goal_diff"] - away_form["avg_goal_diff"],
        "clean_sheet_diff":         home_form["clean_sheet_rate"] - away_form["clean_sheet_rate"],
        "scoring_rate_diff":        home_form["scoring_rate"]  - away_form["scoring_rate"],

        # Experience
        "wc_experience_diff":       home_exp["wc_matches"]    - away_exp["wc_matches"],
        "wc_win_rate_diff":         home_exp["wc_win_rate"]   - away_exp["wc_win_rate"],
        "total_experience_diff":    home_exp["total_experience"] - away_exp["total_experience"],

        # Head to head
        "h2h_win_rate_home":        h2h["h2h_win_rate_a"],
        "h2h_goal_diff":            h2h["h2h_avg_goals_a"] - h2h["h2h_avg_goals_b"],
        "h2h_meetings":             h2h["h2h_meetings"],

        # Venue
        "is_neutral":               int(row.get("neutral", False)),

        # Data quality flag
        "home_matches_available":   home_form["matches_used"],
        "away_matches_available":   away_form["matches_used"],

        # Label
        "result":                   int(row["result"]),
    }

    return features


def build_training_matrix(clean_df: pd.DataFrame,
                           elo_history: pd.DataFrame,
                           start_year: int = 2000) -> pd.DataFrame:
    """
    Build feature matrix for all competitive matches from start_year.

    We use start_year=2000 here (not 1993) because matches between
    1993-1999 are used to warm up rolling stats — they provide
    historical context but aren't included as training rows.
    """
    train_set = clean_df[clean_df["date"].dt.year >= start_year].copy()

    print(f"Building features for {len(train_set):,} matches...")
    print(f"(Using 1993–1999 data as warm-up for rolling stats)\n")

    records = []
    for _, row in tqdm(train_set.iterrows(),
                       total=len(train_set),
                       desc="Building features"):
        feat = build_row_features(row, clean_df, elo_history)
        records.append(feat)

    df = pd.DataFrame(records)

    print(f"\nFeature matrix shape : {df.shape}")
    print(f"Features per match   : {df.shape[1] - 1}  (excl. label)")

    label_dist = df["result"].value_counts(normalize=True).sort_index() * 100
    print(f"\nLabel distribution:")
    for cls, name in [(0, "Away Win"), (1, "Draw"), (2, "Home Win")]:
        print(f"  {name:10s}: {label_dist[cls]:.1f}%")

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("Loading clean data...")
    clean = pd.read_csv(
        os.path.join(PROCESSED_DIR, "results_clean.csv"),
        parse_dates=["date"]
    )

    print("Computing ELO ratings on clean data...")
    elo_ratings, elo_history = compute_elo_ratings(clean)

    # Save current ELO snapshot
    elo_snap = (
        pd.DataFrame(list(elo_ratings.items()), columns=["team", "elo"])
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    elo_snap.to_csv(os.path.join(PROCESSED_DIR, "elo_ratings.csv"), index=False)

    print("\nTop 20 teams by ELO:")
    print(elo_snap.head(20).to_string())

    elo_history.to_csv(
        os.path.join(PROCESSED_DIR, "elo_history.csv"), index=False
    )

    # Build feature matrix
    features = build_training_matrix(clean, elo_history, start_year=2000)
    out = os.path.join(PROCESSED_DIR, "features.csv")
    features.to_csv(out, index=False)

    print(f"\nSaved → {out}")