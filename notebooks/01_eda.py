"""
Phase 2: Exploratory Data Analysis (EDA)

Goal: Understand the data BEFORE touching it.
Ask questions like:
  - How many matches do we have?
  - Are there missing values?
  - Which teams appear most?
  - How are results distributed? (balanced classes?)
  - Are there outliers in scores?
  - How does data volume change over time?
  - Which tournaments are represented?
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

raw_dir    = "data/raw"
plots_dir  = "notebooks/plots"
os.makedirs(plots_dir, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

# ── helper ──────────────────────────────────────────────────────────────────
def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

#helps format printing

# ── load ─────────────────────────────────────────────────────────────────────
results     = pd.read_csv(os.path.join(raw_dir, "results.csv"),
                          parse_dates=["date"])
shootouts   = pd.read_csv(os.path.join(raw_dir, "shootouts.csv"),
                          parse_dates=["date"])
goalscorers = pd.read_csv(os.path.join(raw_dir, "goalscorers.csv"),
                          parse_dates=["date"])
wc_teams    = pd.read_csv(os.path.join(raw_dir, "wc2026_teams.csv"))


# ════════════════════════════════════════════════════════════════════════════
# 1. BASIC SHAPE & TYPES
# ════════════════════════════════════════════════════════════════════════════
section("1. BASIC SHAPE & DATA TYPES")

print("\n--- results.csv ---")
print(f"Shape : {results.shape}  ({results.shape[0]} rows, {results.shape[1]} columns)")
print(f"\nColumns:\n{results.dtypes}")

print("\n--- First 5 rows ---")
print(results.head())

print("\n--- Last 5 rows ---")
print(results.tail())

# ════════════════════════════════════════════════════════════════════════════
# 2. MISSING VALUES
# ════════════════════════════════════════════════════════════════════════════
section("2. MISSING VALUES")

for name, df in [("results", results),
                 ("shootouts", shootouts),
                 ("goalscorers", goalscorers)]:
    missing = df.isnull().sum()
    pct     = (df.isnull().mean() * 100).round(2) #percentage of missing values
    report  = pd.DataFrame({"missing": missing, "pct": pct})
    report  = report[report["missing"] > 0]
    print(f"\n{name}:")
    if len(report) == 0:
        print("  No missing values ✓")
    else:
        print(report.to_string())
#name is used to identify the dataset nothing more


# ════════════════════════════════════════════════════════════════════════════
# 3. DATE RANGE & YEARLY VOLUME
# ════════════════════════════════════════════════════════════════════════════
section("3. DATE RANGE & MATCH VOLUME OVER TIME")

print(f"\nEarliest match : {results['date'].min().date()}")
print(f"Latest match   : {results['date'].max().date()}")
print(f"Total years    : {results['date'].dt.year.nunique()}")

yearly = results.groupby(results["date"].dt.year).size()

fig, ax = plt.subplots(figsize=(14, 4))
yearly.plot(kind="bar", ax=ax, color=sns.color_palette("muted")[0], width=0.8)
ax.set_title("Number of international matches per year")
ax.set_xlabel("Year")
ax.set_ylabel("Matches")
ax.tick_params(axis="x", rotation=90, labelsize=6)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "matches_per_year.png"), dpi=120)
plt.close()
print("Saved: matches_per_year.png")
#yearwise match data plotting


# Spot any suspicious gaps
low_years = yearly[yearly < 50]
if len(low_years):
    print(f"\n⚠  Years with unusually few matches:\n{low_years}")

# ════════════════════════════════════════════════════════════════════════════
# 4. CLASS DISTRIBUTION (WIN / DRAW / LOSS)
# ════════════════════════════════════════════════════════════════════════════
section("4. RESULT DISTRIBUTION (are classes balanced?)")

results["result"] = results.apply(
    lambda r: "Home Win" if r["home_score"] > r["away_score"]
    else ("Draw" if r["home_score"] == r["away_score"] else "Away Win"),
    axis=1
)

dist = results["result"].value_counts()
pct  = (results["result"].value_counts(normalize=True) * 100).round(1)
print(pd.DataFrame({"count": dist, "pct%": pct}).to_string())

# IMPORTANT for ML: if classes are very imbalanced we need to handle it
print(f"\n⚠  Class imbalance note:")
print(f"   If one class is >50% the model will be biased toward it.")
print(f"   We'll address this in Phase 3 (cleaning).")

fig, ax = plt.subplots(figsize=(6, 4))
dist.plot(kind="bar", ax=ax,
          color=[sns.color_palette("muted")[i] for i in range(3)])
ax.set_title("Match result distribution (all time)")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
for i, v in enumerate(dist):
    ax.text(i, v + 50, f"{pct.iloc[i]}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "result_distribution.png"), dpi=120)
plt.close()
print("Saved: result_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
# 5. SCORE DISTRIBUTIONS & OUTLIERS
# ════════════════════════════════════════════════════════════════════════════
section("5. SCORE DISTRIBUTIONS & OUTLIERS")

print(results[["home_score", "away_score"]].describe().round(2))

print(f"\nHighest scoring matches:")
results["total_goals"] = results["home_score"] + results["away_score"]
print(results.nlargest(10, "total_goals")[
    ["date", "home_team", "away_team", "home_score", "away_score",
     "tournament"]].to_string(index=False))

print(f"\nBiggest winning margins:")
results["margin"] = abs(results["home_score"] - results["away_score"])
print(results.nlargest(10, "margin")[
    ["date", "home_team", "away_team", "home_score", "away_score",
     "tournament"]].to_string(index=False))

# Score heatmap — how common is each scoreline?
common_scores = results.groupby(
    ["home_score", "away_score"]
).size().reset_index(name="count")
pivot = common_scores[
    (common_scores["home_score"] <= 8) &
    (common_scores["away_score"] <= 8)
].pivot(index="home_score", columns="away_score", values="count").fillna(0)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.3, ax=ax)
ax.set_title("Scoreline frequency heatmap (home_score vs away_score)")
ax.set_xlabel("Away score")
ax.set_ylabel("Home score")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "scoreline_heatmap.png"), dpi=120)
plt.close()
print("Saved: scoreline_heatmap.png")

# ════════════════════════════════════════════════════════════════════════════
# 6. TOURNAMENT BREAKDOWN
# ════════════════════════════════════════════════════════════════════════════
section("6. TOURNAMENT BREAKDOWN")

tourn = results["tournament"].value_counts()
print(f"Unique tournament names: {results['tournament'].nunique()}")
print(f"\nTop 25 tournaments:")
print(tourn.head(25).to_string())

# Are friendlies dominant?
friendly_pct = (results["tournament"].str.lower().str.contains("friendly").mean() * 100)
print(f"\nFriendly matches: {friendly_pct:.1f}% of all data")
print("Note: we'll filter these OUT for model training (low signal)")

# World Cup matches specifically
wc_mask = results["tournament"].str.contains("FIFA World Cup", na=False)
print(f"\nFIFA World Cup matches: {wc_mask.sum():,}")
print(results[wc_mask]["result"].value_counts().to_string())

# ════════════════════════════════════════════════════════════════════════════
# 7. TEAM COVERAGE
# ════════════════════════════════════════════════════════════════════════════
section("7. TEAM COVERAGE")

all_teams_in_data = pd.concat([
    results["home_team"], results["away_team"]
]).value_counts()

print(f"Unique teams in dataset: {all_teams_in_data.nunique()}")
print(f"\nTop 20 most frequent teams:")
print(all_teams_in_data.head(20).to_string())

# Check all WC2026 teams are present
wc_team_list = wc_teams["team"].tolist()
missing_from_data = [t for t in wc_team_list
                     if t not in all_teams_in_data.index]
print(f"\n⚠  WC2026 teams NOT found in historical data:")
if missing_from_data:
    for t in missing_from_data:
        print(f"   - {t}  ← needs name mapping")
else:
    print("   All teams found ✓")

# Minimum match count check
low_data_teams = [(t, all_teams_in_data.get(t, 0))
                  for t in wc_team_list
                  if all_teams_in_data.get(t, 0) < 50]
if low_data_teams:
    print(f"\n⚠  WC2026 teams with < 50 historical matches:")
    for t, n in sorted(low_data_teams, key=lambda x: x[1]):
        print(f"   {t}: {n} matches")

# ════════════════════════════════════════════════════════════════════════════
# 8. HOME ADVANTAGE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
section("8. HOME ADVANTAGE ANALYSIS")

# Is playing at home actually an advantage?
home_win_all     = (results["result"] == "Home Win").mean() * 100
home_win_neutral = (results[results["neutral"] == True]["result"]
                    == "Home Win").mean() * 100
home_win_nonneutral = (results[results["neutral"] == False]["result"]
                       == "Home Win").mean() * 100

print(f"Home win rate (all matches)         : {home_win_all:.1f}%")
print(f"Home win rate (non-neutral venues)  : {home_win_nonneutral:.1f}%")
print(f"Home win rate (neutral venues)      : {home_win_neutral:.1f}%")
print(f"\nConclusion: {'Home advantage EXISTS ✓' if home_win_nonneutral > home_win_neutral + 3 else 'Home advantage is WEAK'}")

# ════════════════════════════════════════════════════════════════════════════
# 9. GOALS TREND OVER TIME
# ════════════════════════════════════════════════════════════════════════════
section("9. GOALS PER GAME TREND OVER TIME")

results["year"] = results["date"].dt.year
goals_by_year = results.groupby("year")["total_goals"].mean()

fig, ax = plt.subplots(figsize=(14, 4))
goals_by_year.plot(ax=ax, color=sns.color_palette("muted")[2], linewidth=1.5)
ax.set_title("Average goals per match over time")
ax.set_xlabel("Year")
ax.set_ylabel("Avg goals")
ax.axhline(goals_by_year.mean(), color="gray", linestyle="--",
           linewidth=0.8, label=f"Overall avg: {goals_by_year.mean():.2f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "goals_trend.png"), dpi=120)
plt.close()
print("Saved: goals_trend.png")

# ════════════════════════════════════════════════════════════════════════════
# 10. SHOOTOUT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
section("10. SHOOTOUT (PENALTY) ANALYSIS")

print(f"Total shootout records : {len(shootouts):,}")
print(f"\nSample:\n{shootouts.head()}")

# Which teams are best/worst at penalties?
shootout_wins = shootouts.groupby("winner").size().sort_values(ascending=False)
print(f"\nMost penalty shootout wins:")
print(shootout_wins.head(15).to_string())

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
section("EDA COMPLETE — ISSUES TO FIX IN PHASE 3")

print("""
Issues found that need cleaning:
  1. Team name mismatches  → standardise names (e.g. 'Korea Republic' vs 'South Korea')
  2. Friendlies            → filter out for training (low signal)
  3. Very old matches      → consider cutoff (pre-1990 football is different)
  4. Extreme scorelines    → flag or cap outliers (50-0 type matches)
  5. Neutral venue flag    → verify it's consistently set
  6. Missing scores        → check if any rows have NaN in home/away score
  7. Class imbalance       → home wins dominate; we need a strategy
  8. Low-data teams        → teams with < 50 matches need special handling
""")