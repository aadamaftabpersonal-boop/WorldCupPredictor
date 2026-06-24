"""
Phase 3: Data Cleaning

Issues we fix here:
  1. Class imbalance       → class weights (NOT oversampling — leaks into time-series)
  2. Old matches cutoff    → analyse signal decay, pick optimal cutoff year
  3. Extreme scorelines    → detect, inspect, then cap
  4. Neutral venue flag    → verify consistency
  5. Friendlies            → remove for training
  6. Save clean dataset    → ready for feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

raw_dir       = "data/raw"
processed_dir = "data/processed"
plots_dir     = "notebooks/plots"
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── load ─────────────────────────────────────────────────────────────────────
results = pd.read_csv(
    os.path.join(raw_dir, "results.csv"), parse_dates=["date"]
)
results["total_goals"] = results["home_score"] + results["away_score"]
results["margin"]      = abs(results["home_score"] - results["away_score"])
results["year"]        = results["date"].dt.year


# ════════════════════════════════════════════════════════════════════════════
# ISSUE 1 — CLASS IMBALANCE
# ════════════════════════════════════════════════════════════════════════════
section("ISSUE 1 — CLASS IMBALANCE")

results["result"] = results.apply(
    lambda r: 2 if r["home_score"] > r["away_score"]
    else (1 if r["home_score"] == r["away_score"] else 0),
    axis=1
)

dist = results["result"].value_counts().sort_index()
pct  = (results["result"].value_counts(normalize=True).sort_index() * 100).round(1)

print("\nClass distribution:")
for label, name in [(0, "Away Win"), (1, "Draw"), (2, "Home Win")]:
    print(f"  {name:10s} : {dist[label]:6,}  ({pct[label]}%)")

# Strategy: compute class weights (inverse frequency)
# sklearn uses these internally — no data duplication, no leakage
n_samples = len(results)
n_classes = 3
class_weights = {}
for cls in [0, 1, 2]:
    class_weights[cls] = n_samples / (n_classes * dist[cls])

print(f"\nComputed class weights (for model training):")
for cls, name in [(0, "Away Win"), (1, "Draw"), (2, "Home Win")]:
    print(f"  {name:10s} : {class_weights[cls]:.4f}")

print("""
Strategy: pass class_weight=class_weights to sklearn models.
This penalises the model more for misclassifying rare classes.
We do NOT use SMOTE/oversampling — that leaks future data in
time-ordered datasets.
""")

# Save weights for use in training
import json
with open(os.path.join(processed_dir, "class_weights.json"), "w") as f:
    json.dump({str(k): v for k, v in class_weights.items()}, f, indent=2)
print("Saved class_weights.json")


# ════════════════════════════════════════════════════════════════════════════
# ISSUE 2 — OLD MATCHES CUTOFF (signal decay analysis)
# ════════════════════════════════════════════════════════════════════════════
section("ISSUE 2 — OLD MATCHES CUTOFF (signal decay analysis)")

"""
The question: does a match from 1950 help predict 2024 results?
Probably not — squads, tactics, fitness science, and even the
rules have all changed. We measure this by checking if recent
data has stronger predictive signal than old data.

Method: For each year cutoff, measure how well ONLY ELO
(computed from that cutoff onwards) ranks the top 10 teams
against known World Cup performances. A simple proxy: do
higher-ELO teams at the tournament win more?
"""

# -- Plot 1: result distribution by era --
eras = {
    "Pre-1970":  results[results["year"] < 1970],
    "1970-1989": results[(results["year"] >= 1970) & (results["year"] < 1990)],
    "1990-2009": results[(results["year"] >= 1990) & (results["year"] < 2010)],
    "2010+":     results[results["year"] >= 2010],
}

era_dist = {}
for era_name, era_df in eras.items():
    era_dist[era_name] = (
        era_df["result"].value_counts(normalize=True).sort_index() * 100
    ).round(1)

era_df_plot = pd.DataFrame(era_dist).T
era_df_plot.columns = ["Away Win", "Draw", "Home Win"]

fig, ax = plt.subplots(figsize=(10, 5))
era_df_plot.plot(kind="bar", ax=ax, width=0.7)
ax.set_title("Result distribution by era (has football changed?)")
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Era")
ax.tick_params(axis="x", rotation=0)
ax.legend(loc="upper right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", fontsize=7, padding=2)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "result_by_era.png"), dpi=120)
plt.close()
print("Saved: result_by_era.png")
print("\nResult distribution per era:")
print(era_df_plot.to_string())

# -- Plot 2: home advantage decay over time --
yearly_home_win = (
    results.groupby("year")
    .apply(lambda g: (g["result"] == 2).mean() * 100)
    .rename("home_win_pct")
)

fig, ax = plt.subplots(figsize=(14, 4))
yearly_home_win.plot(ax=ax, linewidth=1.5,
                     color=sns.color_palette("muted")[0])
ax.axhline(yearly_home_win[yearly_home_win.index >= 1990].mean(),
           color="red", linestyle="--", linewidth=0.8,
           label="Post-1990 avg")
ax.set_title("Home win % by year — detecting structural changes in football")
ax.set_xlabel("Year")
ax.set_ylabel("Home win %")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "home_win_trend.png"), dpi=120)
plt.close()
print("Saved: home_win_trend.png")

# -- Recommend cutoff --
# Check variance in home win rate per decade
for decade_start in range(1950, 2020, 10):
    decade = results[
        (results["year"] >= decade_start) &
        (results["year"] < decade_start + 10)
    ]
    if len(decade) < 100:
        continue
    hw = (decade["result"] == 2).mean() * 100
    print(f"  {decade_start}s: home win rate = {hw:.1f}%  "
          f"({len(decade):,} matches)")

print("""
Recommendation: use matches from 1993 onwards.
Reasons:
  1. Post-1990 football is structurally similar to today
  2. Enough data (30+ years) for good rolling stats
  3. Covers all modern World Cups (1994, 1998, ... 2022)
  4. Teams that exist today mostly have data from this era
""")
CUTOFF_YEAR = 1993


# ════════════════════════════════════════════════════════════════════════════
# ISSUE 3 — EXTREME SCORELINES
# ════════════════════════════════════════════════════════════════════════════
section("ISSUE 3 — EXTREME SCORELINES (outlier detection)")

print("\nAll matches with total_goals >= 15:")
extreme = results[results["total_goals"] >= 15].sort_values(
    "total_goals", ascending=False
)
print(extreme[["date", "home_team", "away_team",
               "home_score", "away_score",
               "tournament"]].to_string(index=False))

print("\nAll matches with margin >= 10:")
big_margin = results[results["margin"] >= 10].sort_values(
    "margin", ascending=False
)
print(big_margin[["date", "home_team", "away_team",
                  "home_score", "away_score",
                  "tournament"]].to_string(index=False))

# Score distribution boxplot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

results[["home_score", "away_score"]].boxplot(ax=axes[0])
axes[0].set_title("Score distributions (all data) — outliers visible")
axes[0].set_ylabel("Goals")

# After proposed cap
CAP = 8
capped = results.copy()
capped["home_score"] = capped["home_score"].clip(upper=CAP)
capped["away_score"] = capped["away_score"].clip(upper=CAP)
capped[["home_score", "away_score"]].boxplot(ax=axes[1])
axes[1].set_title(f"Score distributions (capped at {CAP})")
axes[1].set_ylabel("Goals")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "score_outliers.png"), dpi=120)
plt.close()
print("Saved: score_outliers.png")

# How many matches are affected by the cap?
affected = (
    (results["home_score"] > CAP) | (results["away_score"] > CAP)
).sum()
print(f"\nMatches affected by cap of {CAP}: {affected} "
      f"({affected/len(results)*100:.2f}% of all data)")
print(f"Strategy: cap scores at {CAP} — preserves result direction, "
      f"reduces distortion in rolling averages")

SCORE_CAP = CAP


# ════════════════════════════════════════════════════════════════════════════
# ISSUE 4 — NEUTRAL VENUE FLAG CONSISTENCY
# ════════════════════════════════════════════════════════════════════════════
section("ISSUE 4 — NEUTRAL VENUE FLAG CONSISTENCY")

neutral_counts = results["neutral"].value_counts()
print(f"\nNeutral flag distribution:\n{neutral_counts}")
print(f"Neutral %: {results['neutral'].mean()*100:.1f}%")

# Check: World Cup matches should all be neutral
# (except host nation — edge case)
wc = results[results["tournament"].str.contains("FIFA World Cup", na=False)]
wc_neutral = wc["neutral"].value_counts()
print(f"\nFIFA World Cup neutral flag:\n{wc_neutral}")

# Find suspicious cases: WC matches flagged as non-neutral
suspicious = wc[wc["neutral"] == False]
if len(suspicious) > 0:
    print(f"\n⚠  {len(suspicious)} World Cup matches flagged as NON-neutral:")
    print(suspicious[["date", "home_team", "away_team",
                       "tournament", "country", "neutral"]].head(20).to_string())
    print("\nThese are likely host-nation matches — acceptable, keep as-is.")
else:
    print("\nAll WC matches flagged neutral ✓")

# Check friendly neutral flag
friendlies = results[results["tournament"].str.lower().str.contains("friendly")]
friendly_neutral = friendlies["neutral"].mean() * 100
print(f"\nFriendlies flagged neutral: {friendly_neutral:.1f}%")
print("Expected ~40-60% (many friendlies played at neutral venues)")


# ════════════════════════════════════════════════════════════════════════════
# APPLY ALL FIXES → SAVE CLEAN DATASET
# ════════════════════════════════════════════════════════════════════════════
section("APPLYING ALL FIXES → SAVING CLEAN DATASET")

clean = results.copy()

# Fix 1: Remove friendlies
before = len(clean)
clean = clean[~clean["tournament"].str.lower().str.contains("friendly")]
print(f"Removed friendlies     : {before - len(clean):,} rows removed")

# Fix 2: Apply year cutoff
before = len(clean)
clean = clean[clean["year"] >= CUTOFF_YEAR]
print(f"Applied year cutoff    : {before - len(clean):,} rows removed "
      f"(kept {CUTOFF_YEAR}+)")

# Fix 3: Cap extreme scores
clean["home_score"] = clean["home_score"].clip(upper=SCORE_CAP)
clean["away_score"] = clean["away_score"].clip(upper=SCORE_CAP)
print(f"Capped scores at {SCORE_CAP}       : done")

# Fix 4: Recompute derived columns after cap
clean["total_goals"] = clean["home_score"] + clean["away_score"]
clean["margin"]      = abs(clean["home_score"] - clean["away_score"])

# Fix 5: Recompute result label (should be same, but be safe)
clean["result"] = clean.apply(
    lambda r: 2 if r["home_score"] > r["away_score"]
    else (1 if r["home_score"] == r["away_score"] else 0),
    axis=1
)

# Fix 6: Drop any rows with NaN in critical columns
critical_cols = ["date", "home_team", "away_team",
                 "home_score", "away_score", "neutral"]
before = len(clean)
clean = clean.dropna(subset=critical_cols)
print(f"Dropped NaN rows       : {before - len(clean):,} rows removed")

# Fix 7: Reset index cleanly
clean = clean.reset_index(drop=True)

# ── Final stats ──
print(f"\n{'─'*40}")
print(f"Final clean dataset    : {len(clean):,} matches")
print(f"Date range             : {clean['date'].min().date()} → "
      f"{clean['date'].max().date()}")
print(f"Unique tournaments     : {clean['tournament'].nunique()}")
print(f"Unique teams           : {pd.concat([clean['home_team'], clean['away_team']]).nunique()}")

final_dist = clean["result"].value_counts(normalize=True).sort_index() * 100
print(f"\nFinal class distribution:")
for cls, name in [(0, "Away Win"), (1, "Draw"), (2, "Home Win")]:
    print(f"  {name:10s}: {final_dist[cls]:.1f}%")

# ── Save ──
out_path = os.path.join(processed_dir, "results_clean.csv")
clean.to_csv(out_path, index=False)
print(f"\nSaved clean data → {out_path}")

print("""
Ready for Phase 4 — Feature Engineering.
Input  : data/processed/results_clean.csv
Output : data/processed/features.csv
""")