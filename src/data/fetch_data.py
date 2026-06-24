"""
Phase 1: Fetching data of international match data + FIFA rankings.
Data source: martj42's international football results dataset on GitHub
(~47,000 matches from 1872 to present — completely free)
"""

import pandas as pd
import requests
import os

raw_dir = "data/raw"
#To make data folder and raw folder if doesnt exist and ignore and move on if exists and not crash the script exists_ok = True
os.makedirs(raw_dir, exist_ok=True)

def fetch_match_results():
    """
    Download historical international match results.
    Returns a DataFrame with columns like: date, home_team, away_team,
    home_score, away_score, tournament, country, city, neutral
    """
    print("Downloading match results....")
    url = (
        "https://raw.githubusercontent.com/martj42/"
        "international_results/master/results.csv"
    )
    response = requests.get(url, timeout = 30) #GET request
    response.raise_for_status() #crash loudly if fails
    save_path = os.path.join(raw_dir, "results.csv")
    with open(save_path, "wb") as f:
        f.write(response.content)
    df = pd.read_csv(save_path, parse_dates=["date"])
    print(f"Downloaded {len(df):,} matches spanning {df['date'].min().year} "
          f"to {df['date'].max().year}")
    return df

def fetch_shootouts():
    """
    Download penalty shootout results.
    Important! A draw in 90 mins that goes to penalties needs a winner.
    """
    print("Downloading shootout data...")

    url = (
        "https://raw.githubusercontent.com/martj42/"
        "international_results/master/shootouts.csv"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    save_path = os.path.join(raw_dir, "shootouts.csv")
    with open(save_path, "wb") as f:
        f.write(response.content)

    df = pd.read_csv(save_path, parse_dates=["date"])
    print(f"Downloaded {len(df):,} shootout records")
    return df

def fetch_goalscorers():
    """
    Download goalscorer data — useful for player-level features later.
    """
    print("Downloading goalscorer data...")

    url = (
        "https://raw.githubusercontent.com/martj42/"
        "international_results/master/goalscorers.csv"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    save_path = os.path.join(raw_dir, "goalscorers.csv")
    with open(save_path, "wb") as f:
        f.write(response.content)

    df = pd.read_csv(save_path, parse_dates=["date"])
    print(f"Downloaded {len(df):,} goalscorer records")
    return df

def load_wc2026_teams():
    """
    The 48 qualified teams for FIFA World Cup 2026.
    We hardcode these since official draws are finalised.
    Returns a dict: {confederation: [team, ...]}
    """
    teams = {
        "UEFA": [
            "Germany", "Spain", "France", "England", "Portugal",
            "Netherlands", "Belgium", "Italy", "Croatia", "Serbia",
            "Austria", "Switzerland", "Denmark", "Poland", "Slovakia",
            "Turkey", "Czech Republic", "Scotland", "Slovenia", "Ukraine",
            "Hungary", "Albania", "Romania", "Georgia"
        ],
        "CONMEBOL": [
            "Brazil", "Argentina", "Colombia", "Uruguay", "Ecuador",
            "Paraguay", "Bolivia", "Chile", "Venezuela"
        ],
        "CONCACAF": [
            "United States", "Mexico", "Canada", "Jamaica",
            "Panama", "Honduras", "Costa Rica", "Cuba"
        ],
        "CAF": [
            "Morocco", "Egypt", "Senegal", "Cameroon",
            "Ivory Coast", "South Africa", "Algeria", "DR Congo",
            "Ghana", "Nigeria", "Tunisia", "Mali", "Guinea"
        ],
        "AFC": [
            "Japan", "South Korea", "Australia", "Iran",
            "Saudi Arabia", "Uzbekistan", "Jordan", "Iraq",
            "Oman", "Indonesia", "Qatar"
        ],
        "OFC": ["New Zealand"],
    }

    # Flatten to a simple set for easy lookup
    all_teams = []
    for conf, team_list in teams.items():
        for team in team_list:
            all_teams.append({"team": team, "confederation": conf})

    df = pd.DataFrame(all_teams)
    df.to_csv(os.path.join(raw_dir, "wc2026_teams.csv"), index=False)
    print(f"Saved {len(df)} WC2026 teams")
    return df

def load_wc2026_venues():
    """
    The 16 stadiums hosting WC2026 across USA, Canada, Mexico.
    """
    venues = [
        {"stadium": "MetLife Stadium",       "city": "New York",       "country": "USA",    "capacity": 82500},
        {"stadium": "AT&T Stadium",          "city": "Dallas",         "country": "USA",    "capacity": 80000},
        {"stadium": "SoFi Stadium",          "city": "Los Angeles",    "country": "USA",    "capacity": 70240},
        {"stadium": "Levi's Stadium",        "city": "San Francisco",  "country": "USA",    "capacity": 68500},
        {"stadium": "Hard Rock Stadium",     "city": "Miami",          "country": "USA",    "capacity": 65326},
        {"stadium": "Arrowhead Stadium",     "city": "Kansas City",    "country": "USA",    "capacity": 76416},
        {"stadium": "Rose Bowl",             "city": "Los Angeles",    "country": "USA",    "capacity": 92000},
        {"stadium": "Geodis Park",           "city": "Nashville",      "country": "USA",    "capacity": 30000},
        {"stadium": "Lincoln Financial",     "city": "Philadelphia",   "country": "USA",    "capacity": 69796},
        {"stadium": "Century Link Field",    "city": "Seattle",        "country": "USA",    "capacity": 72000},
        {"stadium": "Allegiant Stadium",     "city": "Las Vegas",      "country": "USA",    "capacity": 65000},
        {"stadium": "BC Place",              "city": "Vancouver",      "country": "Canada", "capacity": 54500},
        {"stadium": "BMO Field",             "city": "Toronto",        "country": "Canada", "capacity": 30000},
        {"stadium": "Estadio Azteca",        "city": "Mexico City",    "country": "Mexico", "capacity": 87523},
        {"stadium": "Estadio BBVA",          "city": "Monterrey",      "country": "Mexico", "capacity": 53500},
        {"stadium": "Estadio Akron",         "city": "Guadalajara",    "country": "Mexico", "capacity": 49850},
    ]
    df = pd.DataFrame(venues)
    df.to_csv(os.path.join(raw_dir, "venues.csv"), index=False)
    print(f"Saved {len(df)} venue records")
    return df


if __name__ == "__main__":
    results = fetch_match_results()
    shootouts = fetch_shootouts()
    goalscorers = fetch_goalscorers()
    teams = load_wc2026_teams()
    venues = load_wc2026_venues()

    print("\n=== DATA COLLECTION COMPLETE ===")
    print(f"Results shape:     {results.shape}")
    print(f"Shootouts shape:   {shootouts.shape}")
    print(f"Goalscorers shape: {goalscorers.shape}")
    print(f"Teams:             {len(teams)}")
    print(f"Venues:            {len(venues)}")
    print("\nSample match data:")
    print(results.tail(5).to_string())
