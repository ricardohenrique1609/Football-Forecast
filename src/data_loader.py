"""
Data loading and preprocessing for Football Betting Value Finder.
"""
import pandas as pd
import numpy as np
import streamlit as st

LEAGUE_NAMES = {
    "E0": "Premier League (ENG)", "E1": "Championship (ENG)", "E2": "League One (ENG)",
    "E3": "League Two (ENG)", "EC": "Conference (ENG)",
    "SP1": "La Liga (ESP)", "SP2": "Segunda División (ESP)",
    "D1": "Bundesliga (GER)", "D2": "2. Bundesliga (GER)",
    "I1": "Serie A (ITA)", "I2": "Serie B (ITA)",
    "F1": "Ligue 1 (FRA)", "F2": "Ligue 2 (FRA)",
    "N1": "Eredivisie (NED)", "B1": "Pro League (BEL)",
    "P1": "Primeira Liga (POR)", "T1": "Süper Lig (TUR)",
    "G1": "Super League (GRE)", "SC0": "Premiership (SCO)",
    "SC1": "Championship (SCO)", "SC2": "League One (SCO)",
    "SC3": "League Two (SCO)", "ARG": "Primera División (ARG)",
    "BRA1": "Brasileirao (BRA)", "MX1": "Liga MX (MEX)",
    "USA1": "MLS (USA)",
}

RESULT_MAP = {"H": 0, "D": 1, "A": 2}
RESULT_LABEL = {0: "Home Win", 1: "Draw", 2: "Away Win"}


@st.cache_data(show_spinner="Carregando dados de partidas...")
def load_matches(path: str = "Matches.csv") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")
    df = df.dropna(subset=["FTResult", "MatchDate"])
    df["Season"] = df["MatchDate"].dt.year.where(
        df["MatchDate"].dt.month >= 7,
        df["MatchDate"].dt.year - 1
    )
    df["LeagueName"] = df["Division"].map(LEAGUE_NAMES).fillna(df["Division"])
    df["ResultCode"] = df["FTResult"].map(RESULT_MAP)

    # Odds-based features
    mask = df[["OddHome", "OddDraw", "OddAway"]].notna().all(axis=1)
    df.loc[mask, "ImpliedHome"] = 1 / df.loc[mask, "OddHome"]
    df.loc[mask, "ImpliedDraw"] = 1 / df.loc[mask, "OddDraw"]
    df.loc[mask, "ImpliedAway"] = 1 / df.loc[mask, "OddAway"]
    df.loc[mask, "Margin"] = (
        df.loc[mask, "ImpliedHome"]
        + df.loc[mask, "ImpliedDraw"]
        + df.loc[mask, "ImpliedAway"]
    )
    # Fair (no-vig) probabilities
    df.loc[mask, "FairHome"] = df.loc[mask, "ImpliedHome"] / df.loc[mask, "Margin"]
    df.loc[mask, "FairDraw"] = df.loc[mask, "ImpliedDraw"] / df.loc[mask, "Margin"]
    df.loc[mask, "FairAway"] = df.loc[mask, "ImpliedAway"] / df.loc[mask, "Margin"]

    # Elo diff feature
    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]

    return df


@st.cache_data(show_spinner="Carregando ratings Elo...")
def load_elo(path: str = "EloRatings.csv") -> pd.DataFrame:
    elo = pd.read_csv(path)
    elo["date"] = pd.to_datetime(elo["date"], errors="coerce")
    return elo


def get_top_leagues(df: pd.DataFrame, n: int = 10) -> list:
    return df["Division"].value_counts().head(n).index.tolist()
