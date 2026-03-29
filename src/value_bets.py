"""
Value bet identification and market efficiency analysis.
"""
import pandas as pd
import numpy as np


# ── Elo win expectancy ────────────────────────────────────────────────────────
def elo_expected_home(elo_diff: pd.Series, home_advantage: float = 65.0) -> pd.Series:
    """
    Classic Elo win-expectancy for the home team.
    Home advantage is modelled as +65 Elo points (typical football estimate).
    """
    return 1 / (1 + 10 ** (-(elo_diff + home_advantage) / 400))


def add_elo_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Elo-derived win probabilities as an alternative to bookmaker implied probs.

    For football (3-way), we distribute the draw probability as follows:
      P_draw ≈ k * P_home^(1-P_home) (empirical approximation)
    and renormalize so P_home + P_draw + P_away = 1.
    """
    df = df.copy()
    valid = df["EloDiff"].notna()

    # Base Elo win expectancy (ignoring draws)
    base_home = elo_expected_home(df.loc[valid, "EloDiff"])
    base_away = 1 - base_home

    # Estimate draw probability: peaks ~30% when P_home ≈ 0.5
    draw_weight = 0.28 * 4 * base_home * base_away  # max ≈ 0.28 when evenly matched
    draw_weight = draw_weight.clip(0.10, 0.40)

    df.loc[valid, "EloHome"] = base_home  * (1 - draw_weight)
    df.loc[valid, "EloDraw"] = draw_weight
    df.loc[valid, "EloAway"] = base_away  * (1 - draw_weight)

    return df


# ── Value edge: Elo model vs bookmaker ───────────────────────────────────────
def compute_value_bets(df: pd.DataFrame, min_edge: float = 0.03) -> pd.DataFrame:
    """
    Identify value bets where the Elo-based probability exceeds the bookmaker's
    implied (de-vigged) probability by at least min_edge.

    Value edge = EloModel_prob − FairProb (bookmaker, de-vigged)
    """
    df = add_elo_probs(df)
    needed = ["EloHome", "EloDraw", "EloAway", "FairHome", "FairDraw", "FairAway"]
    df = df.dropna(subset=needed)

    df["EdgeHome"] = df["EloHome"] - df["FairHome"]
    df["EdgeDraw"] = df["EloDraw"] - df["FairDraw"]
    df["EdgeAway"] = df["EloAway"] - df["FairAway"]

    df["ValueHome"] = df["EdgeHome"] >= min_edge
    df["ValueDraw"] = df["EdgeDraw"] >= min_edge
    df["ValueAway"] = df["EdgeAway"] >= min_edge

    return df


# ── ROI analysis ──────────────────────────────────────────────────────────────
def _roi(df: pd.DataFrame, odd_col: str, result_val: str) -> dict:
    n = len(df)
    if n == 0:
        return {}
    wins = (df["FTResult"] == result_val).sum()
    profit = (
        (df["FTResult"] == result_val).astype(float) * (df[odd_col] - 1)
    ).sum() - (df["FTResult"] != result_val).astype(float).sum()
    return {"n": n, "wins": int(wins), "profit": profit, "roi": profit / n * 100}


def roi_by_market(df: pd.DataFrame) -> pd.DataFrame:
    """
    ROI for three strategies per market:
      1. Bet ALL games
      2. Bet only when Elo edge >= 3%
      3. Bet only when Elo edge >= 5%
    """
    df_v = add_elo_probs(df)
    rows = []

    for market, odd_col, elo_col, fair_col, result_val in [
        ("Home Win", "OddHome", "EloHome", "FairHome", "H"),
        ("Draw",     "OddDraw",  "EloDraw",  "FairDraw",  "D"),
        ("Away Win", "OddAway",  "EloAway",  "FairAway",  "A"),
    ]:
        base = df_v.dropna(subset=[odd_col, elo_col, fair_col, "FTResult"])
        base = base[base[odd_col] > 1.01]  # remove corrupt odds

        for strategy, sub in [
            ("All bets",      base),
            ("Elo edge >= 3%", base[base[elo_col] - base[fair_col] >= 0.03]),
            ("Elo edge >= 5%", base[base[elo_col] - base[fair_col] >= 0.05]),
        ]:
            r = _roi(sub, odd_col, result_val)
            if r:
                rows.append({
                    "Market": market,
                    "Strategy": strategy,
                    "Bets": r["n"],
                    "Wins": r["wins"],
                    "Win%": round(r["wins"] / r["n"] * 100, 2),
                    "Profit (units)": round(r["profit"], 2),
                    "ROI%": round(r["roi"], 2),
                })
    return pd.DataFrame(rows)


def roi_by_league(df: pd.DataFrame, market: str = "Home", min_edge: float = 0.03) -> pd.DataFrame:
    """ROI breakdown by league for a given market using Elo edge filter."""
    odd_col    = f"Odd{market}"
    elo_col    = f"Elo{market}"
    fair_col   = f"Fair{market}"
    result_val = {"Home": "H", "Draw": "D", "Away": "A"}[market]

    df_v = add_elo_probs(df)
    base = df_v.dropna(subset=[odd_col, elo_col, fair_col, "FTResult", "LeagueName"])
    base = base[(base[odd_col] > 1.01) & (base[elo_col] - base[fair_col] >= min_edge)]

    rows = []
    for league, grp in base.groupby("LeagueName"):
        n = len(grp)
        if n < 30:
            continue
        r = _roi(grp, odd_col, result_val)
        rows.append({
            "League": league,
            "Bets": r["n"],
            "Win%": round(r["wins"] / r["n"] * 100, 2),
            "ROI%": round(r["roi"], 2),
        })
    return pd.DataFrame(rows).sort_values("ROI%", ascending=False)


# ── Calibration: implied prob vs actual rate ──────────────────────────────────
def actual_vs_implied(df: pd.DataFrame, bins: int = 15) -> pd.DataFrame:
    rows = []
    for market, odd_col, result_val in [
        ("Home", "OddHome", "H"),
        ("Draw", "OddDraw",  "D"),
        ("Away", "OddAway",  "A"),
    ]:
        sub = df.dropna(subset=[odd_col, "FTResult"]).copy()
        sub = sub[sub[odd_col] > 1.01]
        sub["implied"] = 1 / sub[odd_col]
        sub["win"] = (sub["FTResult"] == result_val).astype(int)
        sub["bucket"] = pd.cut(sub["implied"], bins=bins)
        agg = (
            sub.groupby("bucket", observed=True)
            .agg(implied_mid=("implied", "mean"),
                 actual_rate=("win", "mean"),
                 count=("win", "count"))
            .reset_index()
        )
        agg["market"] = market
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


# ── Bookmaker margin ──────────────────────────────────────────────────────────
def bookmaker_margin_by_league(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["Margin", "LeagueName"])
    agg = (
        sub.groupby("LeagueName")
        .agg(AvgMargin=("Margin", "mean"), Games=("Margin", "count"))
        .reset_index()
    )
    agg["Vig%"] = ((agg["AvgMargin"] - 1) * 100).round(2)
    return agg.sort_values("Vig%")


# ── Elo diff vs win rate ──────────────────────────────────────────────────────
def elo_diff_win_rate(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["EloDiff", "FTResult"]).copy()
    sub["EloBracket"] = pd.cut(
        sub["EloDiff"],
        bins=[-np.inf, -300, -200, -100, -50, 0, 50, 100, 200, 300, np.inf],
        labels=["<-300", "-300:-200", "-200:-100", "-100:-50", "-50:0",
                "0:50", "50:100", "100:200", "200:300", ">300"],
    )
    agg = (
        sub.groupby("EloBracket", observed=True)
        .agg(
            HomeWin=("FTResult", lambda x: (x == "H").mean()),
            Draw   =("FTResult", lambda x: (x == "D").mean()),
            AwayWin=("FTResult", lambda x: (x == "A").mean()),
            Count  =("FTResult", "count"),
        )
        .reset_index()
    )
    return agg


# ── Elo edge distribution ─────────────────────────────────────────────────────
def elo_edge_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Edge (Elo prob − fair prob) distribution by bucket."""
    df_v = add_elo_probs(df)
    rows = []
    for market, elo_col, fair_col, result_val in [
        ("Home Win", "EloHome", "FairHome", "H"),
        ("Draw",     "EloDraw",  "FairDraw",  "D"),
        ("Away Win", "EloAway",  "FairAway",  "A"),
    ]:
        sub = df_v.dropna(subset=[elo_col, fair_col, "FTResult"]).copy()
        sub["edge"] = sub[elo_col] - sub[fair_col]
        sub["win"]  = (sub["FTResult"] == result_val).astype(int)
        sub["bucket"] = pd.cut(sub["edge"], bins=20)
        agg = (
            sub.groupby("bucket", observed=True)
            .agg(edge_mid=("edge", "mean"),
                 actual_rate=("win", "mean"),
                 count=("win", "count"))
            .reset_index()
        )
        agg["market"] = market
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)
