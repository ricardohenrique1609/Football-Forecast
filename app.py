"""
Football Betting Value Finder — Dashboard Principal
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_loader import load_matches, load_elo, LEAGUE_NAMES
from src.value_bets import (
    compute_value_bets, roi_by_market, roi_by_league,
    actual_vs_implied, bookmaker_margin_by_league,
    elo_diff_win_rate, elo_edge_distribution,
)
from src.ml_model import train_model, model_roi

# ── Configuração ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Football Value Finder",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ─────────────────────────────────────────────────────────────
C = {
    "bg":       "#0a0e1a",
    "surface":  "#111827",
    "card":     "#1a2235",
    "border":   "#2d3748",
    "home":     "#4f9eff",
    "draw":     "#f6c90e",
    "away":     "#ff5c5c",
    "green":    "#34d399",
    "purple":   "#a78bfa",
    "text":     "#f1f5f9",
    "muted":    "#94a3b8",
    "dim":      "#64748b",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {{ box-sizing: border-box; }}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}

/* ── App background ── */
.stApp {{ background: {C["bg"]}; color: {C["text"]}; }}
.main .block-container {{ padding: 1.5rem 2rem 3rem; max-width: 1400px; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {C["surface"]};
    border-right: 1px solid {C["border"]};
}}
section[data-testid="stSidebar"] .block-container {{ padding: 1.5rem 1rem; }}

/* ── Header banner ── */
.hero-banner {{
    background: linear-gradient(135deg, #0f1f40 0%, #1a2d5a 50%, #0f1f40 100%);
    border: 1px solid {C["border"]};
    border-radius: 16px;
    padding: 28px 32px 24px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}}
.hero-banner::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(79,158,255,0.08) 0%, transparent 60%);
    pointer-events: none;
}}
.hero-title {{
    font-size: 26px; font-weight: 800; color: {C["text"]};
    margin: 0 0 4px; letter-spacing: -0.5px;
}}
.hero-title span {{ color: {C["home"]}; }}
.hero-sub {{ font-size: 13px; color: {C["muted"]}; margin: 0; }}

/* ── KPI Cards ── */
.kpi-row {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }}
.kpi-card {{
    flex: 1; min-width: 130px;
    background: {C["card"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 16px 18px;
    position: relative; overflow: hidden;
}}
.kpi-card::after {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 12px 12px 0 0;
}}
.kpi-card.blue::after  {{ background: {C["home"]}; }}
.kpi-card.yellow::after {{ background: {C["draw"]}; }}
.kpi-card.red::after   {{ background: {C["away"]}; }}
.kpi-card.green::after {{ background: {C["green"]}; }}
.kpi-card.purple::after {{ background: {C["purple"]}; }}
.kpi-label {{ font-size: 11px; font-weight: 600; color: {C["muted"]}; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px; }}
.kpi-value {{ font-size: 22px; font-weight: 700; color: {C["text"]}; line-height: 1; }}
.kpi-sub   {{ font-size: 11px; color: {C["dim"]}; margin-top: 4px; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {C["surface"]};
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid {C["border"]};
}}
.stTabs [role="tab"] {{
    color: {C["muted"]};
    font-weight: 600;
    font-size: 13px;
    border-radius: 8px;
    padding: 8px 16px;
    border: none !important;
    transition: all 0.2s;
}}
.stTabs [aria-selected="true"] {{
    color: {C["text"]} !important;
    background: {C["card"]} !important;
    border: none !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
.stTabs [data-baseweb="tab-border"]    {{ display: none; }}

/* ── Section titles ── */
.section-title {{
    font-size: 16px; font-weight: 700; color: {C["text"]};
    margin: 24px 0 12px;
    display: flex; align-items: center; gap: 8px;
}}
.section-title::before {{
    content: '';
    display: inline-block; width: 3px; height: 18px;
    background: {C["home"]}; border-radius: 2px;
}}

/* ── Info box ── */
.info-box {{
    background: rgba(79,158,255,0.08);
    border: 1px solid rgba(79,158,255,0.25);
    border-left: 3px solid {C["home"]};
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 20px;
    font-size: 13px;
    color: {C["text"]};
    line-height: 1.6;
}}
.info-box code {{
    background: rgba(79,158,255,0.15);
    color: {C["home"]};
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Fira Code', monospace;
    font-size: 12px;
}}

/* ── Metric native override ── */
[data-testid="stMetric"] {{
    background: {C["card"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 14px 16px;
}}
[data-testid="stMetricLabel"] {{ color: {C["muted"]}; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
[data-testid="stMetricValue"] {{ color: {C["home"]}; font-weight: 700; }}

/* ── Divider ── */
hr {{ border-color: {C["border"]} !important; margin: 20px 0; }}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {C["border"]};
    border-radius: 10px;
    overflow: hidden;
}}

/* ── Selectbox / multiselect ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {{
    background: {C["card"]};
    border: 1px solid {C["border"]};
    border-radius: 8px;
    color: {C["text"]};
}}

/* ── Sidebar widgets ── */
.sidebar-section {{
    background: {C["card"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 14px;
    margin: 12px 0;
}}

/* ── Caption / footer ── */
.footer {{
    text-align: center;
    font-size: 12px;
    color: {C["dim"]};
    padding: 24px 0 8px;
    border-top: 1px solid {C["border"]};
    margin-top: 32px;
}}
</style>
""", unsafe_allow_html=True)

# ── Chart theme helpers ────────────────────────────────────────────────────────
_CHART_BASE = dict(
    plot_bgcolor=C["bg"],
    paper_bgcolor=C["card"],
    font_color=C["text"],
    font_family="Inter",
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=C["border"],
        borderwidth=1,
        font_size=12,
    ),
    margin=dict(l=10, r=10, t=48, b=10),
    title_font_size=14,
    title_font_color=C["text"],
    hoverlabel=dict(
        bgcolor=C["surface"],
        bordercolor=C["border"],
        font_color=C["text"],
        font_size=13,
    ),
)

_AXES = dict(
    gridcolor=C["border"],
    zerolinecolor=C["border"],
    linecolor=C["border"],
    tickcolor=C["muted"],
    tickfont_color=C["muted"],
    tickfont_size=11,
)


def theme(fig, height: int = 380) -> go.Figure:
    """Apply unified dark theme to a figure."""
    fig.update_layout(height=height, **_CHART_BASE)
    fig.update_xaxes(**_AXES)
    fig.update_yaxes(**_AXES)
    return fig


def bar_colors(values, pos_color=None, neg_color=None):
    pos = pos_color or C["green"]
    neg = neg_color or C["away"]
    return [pos if v >= 0 else neg for v in values]


# ── Data ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
df_raw = load_matches(os.path.join(BASE_DIR, "data", "Matches.csv"))
elo_df  = load_elo(os.path.join(BASE_DIR,  "data", "EloRatings.csv"))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 8px 0 16px;">
        <div style="font-size:32px;">⚽</div>
        <div style="font-size:15px; font-weight:700; color:{C['text']};">Value Finder</div>
        <div style="font-size:11px; color:{C['muted']};">Football Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    all_leagues = sorted(df_raw["Division"].unique())
    league_display = {d: LEAGUE_NAMES.get(d, d) for d in all_leagues}
    selected_divs = st.multiselect(
        "Ligas",
        options=all_leagues,
        default=["E0", "SP1", "D1", "I1", "F1"],
        format_func=lambda x: league_display[x],
    )
    if not selected_divs:
        selected_divs = all_leagues

    seasons = sorted(df_raw["Season"].dropna().unique().astype(int))
    season_range = st.slider(
        "Temporadas",
        min_value=int(seasons[0]),
        max_value=int(seasons[-1]),
        value=(2015, int(seasons[-1])),
    )

    st.divider()
    st.markdown(f"<div style='font-size:12px; font-weight:600; color:{C['muted']}; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;'>Modelo ML</div>", unsafe_allow_html=True)
    edge_threshold = st.slider(
        "Edge mínimo",
        0.01, 0.15, 0.05, 0.01,
        help="Diferença mínima entre prob. do modelo XGBoost e a fair prob para considerar value bet",
    )
    run_ml = st.toggle("Treinar XGBoost", value=True)

    st.divider()
    st.markdown(f"<div style='font-size:11px; color:{C['dim']}; text-align:center; line-height:1.6;'>230k+ jogos · 27 países<br>42 ligas · 2000–2025</div>", unsafe_allow_html=True)

# ── Filter ────────────────────────────────────────────────────────────────────
df = df_raw[
    df_raw["Division"].isin(selected_divs) &
    df_raw["Season"].between(season_range[0], season_range[1])
].copy()
df_odds = df.dropna(subset=["OddHome", "OddDraw", "OddAway", "FairHome"])

label_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

# ── Hero Banner ───────────────────────────────────────────────────────────────
n_leagues = len(selected_divs)
n_seasons = season_range[1] - season_range[0] + 1
st.markdown(f"""
<div class="hero-banner">
  <div class="hero-title">⚽ Football Betting <span>Value Finder</span></div>
  <div class="hero-sub">
    Identificação de value bets · {len(df):,} jogos · {n_seasons} temporadas · {n_leagues} liga(s) selecionada(s)
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
hw = (df["FTResult"] == "H").mean() * 100
dw = (df["FTResult"] == "D").mean() * 100
aw = (df["FTResult"] == "A").mean() * 100
avg_vig  = (df_odds["Margin"] - 1).mean() * 100 if len(df_odds) else 0
avg_goals = df.dropna(subset=["FTHome","FTAway"]).eval("FTHome + FTAway").mean()

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card blue">
    <div class="kpi-label">Jogos c/ Odds</div>
    <div class="kpi-value">{len(df_odds):,}</div>
    <div class="kpi-sub">de {len(df):,} total</div>
  </div>
  <div class="kpi-card blue">
    <div class="kpi-label">Home Win</div>
    <div class="kpi-value">{hw:.1f}%</div>
    <div class="kpi-sub">mandante vence</div>
  </div>
  <div class="kpi-card yellow">
    <div class="kpi-label">Draw</div>
    <div class="kpi-value">{dw:.1f}%</div>
    <div class="kpi-sub">empate</div>
  </div>
  <div class="kpi-card red">
    <div class="kpi-label">Away Win</div>
    <div class="kpi-value">{aw:.1f}%</div>
    <div class="kpi-sub">visitante vence</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-label">Média Gols</div>
    <div class="kpi-value">{avg_goals:.2f}</div>
    <div class="kpi-sub">por jogo</div>
  </div>
  <div class="kpi-card purple">
    <div class="kpi-label">Vig Médio</div>
    <div class="kpi-value">{avg_vig:.2f}%</div>
    <div class="kpi-sub">margem da casa</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Visão Geral",
    "💰  Value Bets",
    "📈  Mercado",
    "🤖  Modelo ML",
    "🔍  Times",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Visão Geral
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns([1, 1.6])

    with c1:
        st.markdown('<div class="section-title">Distribuição de Resultados</div>', unsafe_allow_html=True)
        counts = df["FTResult"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Home Win", "Draw", "Away Win"],
            values=[counts.get("H",0), counts.get("D",0), counts.get("A",0)],
            hole=0.62,
            marker=dict(
                colors=[C["home"], C["draw"], C["away"]],
                line=dict(color=C["bg"], width=3),
            ),
            textinfo="label+percent",
            textfont_size=13,
            hovertemplate="<b>%{label}</b><br>%{value:,} jogos (%{percent})<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b>{len(df):,}</b><br><span style='font-size:11px'>jogos</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color=C["text"]),
        )
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.08))
        theme(fig, 360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Resultados por Liga</div>', unsafe_allow_html=True)
        league_res = (
            df.groupby(["LeagueName", "FTResult"]).size().reset_index(name="count")
        )
        total_per_league = league_res.groupby("LeagueName")["count"].transform("sum")
        league_res["pct"] = league_res["count"] / total_per_league * 100
        top_leagues_list = df["LeagueName"].value_counts().head(14).index.tolist()
        lrt = league_res[league_res["LeagueName"].isin(top_leagues_list)].copy()
        lrt["label"] = lrt["FTResult"].map(label_map)
        fig = px.bar(
            lrt, x="pct", y="LeagueName", color="label", orientation="h",
            barmode="stack",
            color_discrete_map={"Home Win": C["home"], "Draw": C["draw"], "Away Win": C["away"]},
            labels={"pct": "(%)", "LeagueName": "", "label": ""},
            custom_data=["label","count"],
        )
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b>: %{x:.1f}% (%{customdata[1]:,})<extra></extra>",
        )
        fig.update_layout(legend_title="", bargap=0.25)
        theme(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Tendência de Resultados (% por Temporada)</div>', unsafe_allow_html=True)
    time_res = df.groupby(["Season","FTResult"]).size().reset_index(name="count")
    time_res["pct"] = time_res["count"] / time_res.groupby("Season")["count"].transform("sum") * 100
    time_res["label"] = time_res["FTResult"].map(label_map)
    fig = px.line(
        time_res, x="Season", y="pct", color="label",
        color_discrete_map={"Home Win": C["home"], "Draw": C["draw"], "Away Win": C["away"]},
        markers=True, labels={"pct": "%", "Season": "Temporada", "label": ""},
    )
    fig.update_traces(line_width=2.5, marker_size=5)
    theme(fig, 320)
    st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="section-title">Distribuição das Odds</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for label, col, color in [
            ("Home Win","OddHome",C["home"]),
            ("Draw","OddDraw",C["draw"]),
            ("Away Win","OddAway",C["away"]),
        ]:
            vals = df_odds[col].dropna()
            vals = vals[vals < vals.quantile(0.98)]
            fig.add_trace(go.Histogram(
                x=vals, name=label, opacity=0.70,
                marker_color=color, nbinsx=60,
                hovertemplate=f"<b>{label}</b><br>Odd: %{{x:.2f}}<br>Freq: %{{y:,}}<extra></extra>",
            ))
        fig.update_layout(barmode="overlay", xaxis_title="Odd", yaxis_title="Frequência")
        theme(fig, 340)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Média de Gols por Temporada</div>', unsafe_allow_html=True)
        goals = df.dropna(subset=["FTHome","FTAway"]).copy()
        goals["TotalGoals"] = goals["FTHome"] + goals["FTAway"]
        gs = goals.groupby("Season")["TotalGoals"].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=gs["Season"], y=gs["TotalGoals"],
            marker=dict(
                color=gs["TotalGoals"],
                colorscale=[[0, C["surface"]], [1, C["green"]]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}</b><br>Média: %{y:.2f} gols<extra></extra>",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=gs["Season"], y=gs["TotalGoals"],
            mode="lines", line=dict(color=C["green"], width=2),
            showlegend=False,
        ))
        fig.update_layout(xaxis_title="Temporada", yaxis_title="Gols / Jogo")
        theme(fig, 340)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Value Bets
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"""
    <div class="info-box">
        <strong>O que é Value Bet?</strong> Ocorre quando a probabilidade <em>real</em> de um evento é maior que
        a probabilidade implícita da odd — ou seja, a casa de apostas está <em>subprecificando</em> o evento.<br><br>
        Aqui usamos o <strong>modelo Elo</strong> como estimativa da probabilidade real e comparamos contra a
        <em>fair probability</em> (odd sem vig) do mercado.<br>
        Fórmula: <code>Value = (Prob. Elo &times; Odd) &minus; 1 &gt; 0</code>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">ROI por Mercado × Estratégia</div>', unsafe_allow_html=True)
        roi_mkt = roi_by_market(df_odds)
        if not roi_mkt.empty:
            strat_colors = {
                "All bets":       C["dim"],
                "Elo edge >= 3%": C["draw"],
                "Elo edge >= 5%": C["green"],
            }
            fig = px.bar(
                roi_mkt, x="Market", y="ROI%", color="Strategy",
                barmode="group",
                color_discrete_map=strat_colors,
                text=roi_mkt["ROI%"].apply(lambda x: f"{x:+.1f}%"),
                custom_data=["Bets","Win%"],
                labels={"ROI%": "ROI (%)", "Market": "", "Strategy": ""},
            )
            fig.update_traces(
                textposition="outside",
                hovertemplate="<b>%{x}</b> · %{data.name}<br>ROI: %{y:+.2f}%<br>Apostas: %{customdata[0]:,}<br>Win: %{customdata[1]:.1f}%<extra></extra>",
            )
            fig.add_hline(y=0, line_dash="dot", line_color=C["muted"], line_width=1)
            fig.update_layout(yaxis_title="ROI (%)", bargap=0.2, bargroupgap=0.06)
            theme(fig, 380)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Calibração: Implied Prob vs Taxa Real</div>', unsafe_allow_html=True)
        cal = actual_vs_implied(df_odds, bins=15)
        mcolors = {"Home": C["home"], "Draw": C["draw"], "Away": C["away"]}
        fig = go.Figure()
        for mkt in ["Home","Draw","Away"]:
            sub = cal[cal["market"]==mkt].dropna()
            fig.add_trace(go.Scatter(
                x=sub["implied_mid"], y=sub["actual_rate"],
                mode="lines+markers", name=mkt,
                line=dict(color=mcolors[mkt], width=2.5),
                marker=dict(size=7, color=mcolors[mkt],
                            line=dict(width=1.5, color=C["bg"])),
                hovertemplate=f"<b>{mkt}</b><br>Implied: %{{x:.2%}}<br>Real: %{{y:.2%}}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(dash="dot", color=C["muted"], width=1.5),
            name="Calibração perfeita",
        ))
        fig.update_layout(
            xaxis_title="Probabilidade Implícita (1/odd)",
            yaxis_title="Taxa Real de Vitórias",
        )
        theme(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">ROI por Liga com Filtro de Edge Elo</div>', unsafe_allow_html=True)
    c3, c4 = st.columns([1, 3])
    with c3:
        market_sel = st.selectbox("Mercado:", ["Home","Draw","Away"], key="vb_market")
    roi_lg = roi_by_league(df_odds, market_sel)
    if not roi_lg.empty:
        fig = px.bar(
            roi_lg.head(22), x="ROI%", y="League", orientation="h",
            color="ROI%",
            color_continuous_scale=[[0,"#ff5c5c"],[0.5,"#f6c90e"],[1,"#34d399"]],
            color_continuous_midpoint=0,
            text=roi_lg.head(22)["ROI%"].apply(lambda x: f"{x:+.1f}%"),
            custom_data=["Bets","Win%"],
            labels={"ROI%":"ROI (%)","League":""},
        )
        fig.update_traces(
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>ROI: %{x:+.2f}%<br>Apostas: %{customdata[0]:,}<br>Win: %{customdata[1]:.1f}%<extra></extra>",
        )
        fig.update_layout(coloraxis_showscale=False, bargap=0.2)
        theme(fig, 520)
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)

    with c5:
        st.markdown('<div class="section-title">Resultados por Diferença Elo</div>', unsafe_allow_html=True)
        elo_wr = elo_diff_win_rate(df)
        if not elo_wr.empty:
            fig = go.Figure()
            for col, label, color in [
                ("HomeWin","Home Win",C["home"]),
                ("Draw","Draw",C["draw"]),
                ("AwayWin","Away Win",C["away"]),
            ]:
                fig.add_trace(go.Bar(
                    x=elo_wr["EloBracket"].astype(str),
                    y=(elo_wr[col]*100).round(1),
                    name=label, marker_color=color,
                    hovertemplate=f"<b>{label}</b><br>Bracket: %{{x}}<br>%{{y:.1f}}%<extra></extra>",
                ))
            fig.update_layout(barmode="stack", xaxis_title="Elo Diff (Home − Away)", yaxis_title="%")
            theme(fig, 380)
            st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.markdown('<div class="section-title">Edge Elo vs Taxa Real de Vitórias</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px; color:{C['muted']}; margin: -4px 0 10px;'>Quanto maior o edge do modelo Elo, mais alta a taxa real?</div>", unsafe_allow_html=True)
        eed = elo_edge_distribution(df_odds)
        if not eed.empty:
            mce = {"Home Win":C["home"],"Draw":C["draw"],"Away Win":C["away"]}
            fig = go.Figure()
            for mkt in eed["market"].unique():
                sub = eed[(eed["market"]==mkt) & eed["count"].ge(20)].dropna()
                fig.add_trace(go.Scatter(
                    x=sub["edge_mid"], y=sub["actual_rate"]*100,
                    mode="lines+markers", name=mkt,
                    line=dict(color=mce[mkt], width=2.5),
                    marker=dict(size=6, color=mce[mkt],
                                line=dict(width=1.5, color=C["bg"])),
                    hovertemplate=f"<b>{mkt}</b><br>Edge: %{{x:.3f}}<br>Win: %{{y:.1f}}%<extra></extra>",
                ))
            fig.add_vline(x=0, line_dash="dot", line_color=C["muted"], line_width=1.5)
            fig.update_layout(
                xaxis_title="Elo Edge (EloProb − FairProb)",
                yaxis_title="Taxa Real (%)",
            )
            theme(fig, 380)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Eficiência de Mercado
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="section-title">Margem da Casa (Vig) por Liga</div>', unsafe_allow_html=True)
        vig = bookmaker_margin_by_league(df_odds)
        vig = vig[vig["Games"] >= 100].head(25)
        fig = px.bar(
            vig, x="Vig%", y="LeagueName", orientation="h",
            color="Vig%",
            color_continuous_scale=[[0,"#34d399"],[0.5,"#f6c90e"],[1,"#ff5c5c"]],
            text=vig["Vig%"].apply(lambda x: f"{x:.2f}%"),
            custom_data=["Games"],
            labels={"Vig%":"Margem (%)","LeagueName":""},
        )
        fig.update_traces(
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Vig: %{x:.2f}%<br>Jogos: %{customdata[0]:,}<extra></extra>",
        )
        fig.update_layout(coloraxis_showscale=False)
        theme(fig, 560)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Over 2.5 — Implied vs Real</div>', unsafe_allow_html=True)
        ou = df_odds.dropna(subset=["Over25","Under25","FTHome","FTAway"]).copy()
        ou["TotalGoals"] = ou["FTHome"] + ou["FTAway"]
        ou["IsOver"] = (ou["TotalGoals"] > 2.5).astype(int)
        ou["ImpliedOver"] = 1 / ou["Over25"]
        ou_season = ou.groupby("Season").agg(
            ImpliedOver=("ImpliedOver","mean"),
            ActualOver=("IsOver","mean"),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ou_season["Season"], y=ou_season["ImpliedOver"]*100,
            name="Implied Over 2.5 %", mode="lines+markers",
            line=dict(color=C["draw"], width=2.5),
            marker=dict(size=6, color=C["draw"]),
        ))
        fig.add_trace(go.Scatter(
            x=ou_season["Season"], y=ou_season["ActualOver"]*100,
            name="Real Over 2.5 %", mode="lines+markers",
            line=dict(color=C["green"], width=2.5, dash="dot"),
            marker=dict(size=6, color=C["green"]),
        ))
        fig.update_layout(xaxis_title="Temporada", yaxis_title="%")
        theme(fig, 270)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Ligas: Implied vs Real Over 2.5</div>', unsafe_allow_html=True)
        ou_lg = ou.groupby("LeagueName").agg(
            ImpliedOver=("ImpliedOver","mean"),
            ActualOver=("IsOver","mean"),
            Games=("IsOver","count"),
        ).reset_index()
        ou_lg = ou_lg[ou_lg["Games"] >= 100]
        fig = px.scatter(
            ou_lg, x="ImpliedOver", y="ActualOver", size="Games",
            color="ActualOver",
            color_continuous_scale=[[0,"#ff5c5c"],[0.5,"#f6c90e"],[1,"#34d399"]],
            text="LeagueName",
            custom_data=["Games"],
            labels={"ImpliedOver":"Implied Over 2.5","ActualOver":"Real Over 2.5"},
        )
        fig.add_shape(type="line", x0=0.45, y0=0.45, x1=0.72, y1=0.72,
                      line=dict(color=C["muted"], dash="dot", width=1.5))
        fig.update_traces(
            textposition="top center", textfont_size=9,
            hovertemplate="<b>%{text}</b><br>Implied: %{x:.2%}<br>Real: %{y:.2%}<br>Jogos: %{customdata[0]:,}<extra></extra>",
        )
        fig.update_layout(coloraxis_showscale=False)
        theme(fig, 280)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Evolução das Odds Médias por Temporada</div>', unsafe_allow_html=True)
    odds_time = df_odds.groupby("Season").agg(
        Home=("OddHome","mean"), Draw=("OddDraw","mean"), Away=("OddAway","mean"),
    ).reset_index()
    fig = go.Figure()
    for col, color, label in [
        ("Home", C["home"], "Home Win"),
        ("Draw", C["draw"], "Draw"),
        ("Away", C["away"], "Away Win"),
    ]:
        fig.add_trace(go.Scatter(
            x=odds_time["Season"], y=odds_time[col],
            name=label, mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=5, color=color),
            hovertemplate=f"<b>{label}</b> %{{x}}: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(xaxis_title="Temporada", yaxis_title="Odd Média")
    theme(fig, 340)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Heatmap · Home Win % por Liga × Temporada</div>', unsafe_allow_html=True)
    top10 = df["LeagueName"].value_counts().head(10).index.tolist()
    hm_data = (
        df[df["LeagueName"].isin(top10)]
        .groupby(["LeagueName","Season"])
        .apply(lambda x: (x["FTResult"]=="H").mean()*100)
        .reset_index(name="HomeWin%")
    )
    hm_pivot = hm_data.pivot(index="LeagueName", columns="Season", values="HomeWin%")
    fig = go.Figure(go.Heatmap(
        z=hm_pivot.values,
        x=hm_pivot.columns.astype(str),
        y=hm_pivot.index,
        colorscale=[[0,"#ff5c5c"],[0.45,"#f6c90e"],[1,"#34d399"]],
        hovertemplate="<b>%{y}</b><br>Temporada %{x}<br>Home Win: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Home%", tickfont_color=C["muted"], title_font_color=C["muted"]),
        xgap=1, ygap=1,
    ))
    fig.update_layout(xaxis_title="Temporada")
    theme(fig, 380)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Modelo ML
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if not run_ml:
        st.markdown(f'<div class="info-box">Ative o treinamento do modelo no painel lateral.</div>', unsafe_allow_html=True)
    else:
        result = train_model(df)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Treino", f"{result['n_train']:,} jogos")
        mc2.metric("Teste",  f"{result['n_test']:,} jogos")
        mc3.metric("AUC (macro OvR)", f"{result['auc']:.4f}")
        mc4.metric("Log Loss", f"{result['log_loss']:.4f}")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
            imp = result["importance"]
            fig = go.Figure(go.Bar(
                x=imp["Importance"],
                y=imp["Feature"],
                orientation="h",
                marker=dict(
                    color=imp["Importance"],
                    colorscale=[[0, C["surface"]], [1, C["home"]]],
                    line=dict(width=0),
                ),
                text=imp["Importance"].apply(lambda x: f"{x:.3f}"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Importância: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(xaxis_title="Importância (gain)")
            theme(fig, 420)
            fig.update_yaxes(autorange="reversed")  # fix: separate call avoids key conflict
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-title">Matriz de Confusão (normalizada)</div>', unsafe_allow_html=True)
            cm = result["cm"]
            labels_cm = ["Home Win", "Draw", "Away Win"]
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            text_vals = [[f"{cm_norm[i][j]*100:.1f}%<br>({cm[i][j]:,})" for j in range(3)] for i in range(3)]
            fig = go.Figure(go.Heatmap(
                z=cm_norm,
                x=[f"Pred: {l}" for l in labels_cm],
                y=[f"Real: {l}" for l in labels_cm],
                colorscale=[[0, C["bg"]], [0.5, "#1a3a6b"], [1, C["home"]]],
                text=text_vals,
                texttemplate="%{text}",
                hovertemplate="Real: %{y}<br>Pred: %{x}<br>%{text}<extra></extra>",
                colorbar=dict(title="Prob", tickfont_color=C["muted"], title_font_color=C["muted"]),
                xgap=3, ygap=3,
            ))
            theme(fig, 420)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Curvas de Calibração do Modelo</div>', unsafe_allow_html=True)
        calib_colors = {"Home Win":C["home"],"Draw":C["draw"],"Away Win":C["away"]}
        fig = go.Figure()
        for cls_name, data in result["calib"].items():
            fig.add_trace(go.Scatter(
                x=data["mean_pred"], y=data["frac_pos"],
                name=cls_name, mode="lines+markers",
                line=dict(color=calib_colors[cls_name], width=2.5),
                marker=dict(size=8, color=calib_colors[cls_name],
                            line=dict(width=2, color=C["bg"])),
            ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(dash="dot", color=C["muted"], width=1.5),
            name="Calibração perfeita",
        ))
        fig.update_layout(
            xaxis_title="Probabilidade Prevista",
            yaxis_title="Fração Positiva Real",
        )
        theme(fig, 360)
        st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns([1.2, 1])

        with c3:
            st.markdown('<div class="section-title">Relatório de Classificação</div>', unsafe_allow_html=True)
            report_df = pd.DataFrame(result["report"]).T.iloc[:3]
            report_df = report_df[["precision","recall","f1-score","support"]].round(3)
            report_df.index = ["Home Win","Draw","Away Win"]
            st.dataframe(
                report_df.style
                    .format({"precision":"{:.3f}","recall":"{:.3f}","f1-score":"{:.3f}","support":"{:.0f}"})
                    .background_gradient(cmap="Blues", subset=["precision","recall","f1-score"]),
                use_container_width=True,
            )

        with c4:
            st.markdown('<div class="section-title">ROI com Edge do Modelo XGBoost</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:12px;color:{C['muted']};margin:-6px 0 10px;'>Edge > {edge_threshold:.0%} vs fair prob</div>", unsafe_allow_html=True)
            pred_df = result["predictions"]
            mroi = model_roi(pred_df, edge_threshold)
            if not mroi.empty:
                fig = go.Figure(go.Bar(
                    x=mroi["Market"], y=mroi["ROI%"],
                    marker=dict(
                        color=bar_colors(mroi["ROI%"]),
                        line=dict(width=0),
                    ),
                    text=[f"{v:+.2f}%" for v in mroi["ROI%"]],
                    textposition="outside",
                    customdata=mroi[["Bets","Win%"]].values,
                    hovertemplate="<b>%{x}</b><br>ROI: %{y:+.2f}%<br>Apostas: %{customdata[0]:,}<br>Win: %{customdata[1]:.1f}%<extra></extra>",
                ))
                fig.add_hline(y=0, line_dash="dot", line_color=C["muted"], line_width=1)
                fig.update_layout(yaxis_title="ROI (%)", showlegend=False)
                theme(fig, 300)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Distribuição do Edge do Modelo por Resultado</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;color:{C['muted']};margin:-6px 0 10px;'>Jogos onde o modelo acertou vs errou — distribuição do edge previsto</div>", unsafe_allow_html=True)
        sub_titles = ["Home Win", "Draw", "Away Win"]
        fig = make_subplots(rows=1, cols=3, subplot_titles=sub_titles,
                            horizontal_spacing=0.08)
        for i, (edge_col, result_val, color) in enumerate([
            ("ModelEdgeHome","H",C["home"]),
            ("ModelEdgeDraw","D",C["draw"]),
            ("ModelEdgeAway","A",C["away"]),
        ], start=1):
            sub = pred_df.dropna(subset=[edge_col]).copy()
            sub = sub[sub[edge_col].abs() < 0.3]
            won  = sub[sub["FTResult"]==result_val][edge_col]
            lost = sub[sub["FTResult"]!=result_val][edge_col]
            fig.add_trace(go.Histogram(x=won,  name="Acertou", marker_color=C["green"],
                                       opacity=0.8, nbinsx=40, showlegend=(i==1)), row=1, col=i)
            fig.add_trace(go.Histogram(x=lost, name="Errou",   marker_color=C["away"],
                                       opacity=0.5, nbinsx=40, showlegend=(i==1)), row=1, col=i)
        fig.update_layout(barmode="overlay", **_CHART_BASE, height=320)
        fig.update_xaxes(**_AXES)
        fig.update_yaxes(**_AXES)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Explorador de Times
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    all_teams = sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))
    default_idx = all_teams.index("Arsenal") if "Arsenal" in all_teams else 0
    team = st.selectbox("Selecione um time:", all_teams, index=default_idx)

    home_df = df[df["HomeTeam"]==team].copy()
    away_df = df[df["AwayTeam"]==team].copy()
    home_df["Venue"] = "Casa"
    away_df["Venue"] = "Fora"
    home_df["TeamResult"] = home_df["FTResult"].map({"H":"Vitória","D":"Empate","A":"Derrota"})
    away_df["TeamResult"] = away_df["FTResult"].map({"H":"Derrota","D":"Empate","A":"Vitória"})
    home_df["GF"] = home_df["FTHome"]
    home_df["GC"] = home_df["FTAway"]
    away_df["GF"] = away_df["FTAway"]
    away_df["GC"] = away_df["FTHome"]
    all_team = pd.concat([home_df, away_df]).sort_values("MatchDate")

    if len(all_team) == 0:
        st.warning("Nenhum jogo encontrado para este time no período selecionado.")
    else:
        total  = len(all_team)
        wins   = (all_team["TeamResult"]=="Vitória").sum()
        draws  = (all_team["TeamResult"]=="Empate").sum()
        losses = (all_team["TeamResult"]=="Derrota").sum()
        avg_gf = all_team["GF"].mean()
        avg_gc = all_team["GC"].mean()

        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi-card blue">
            <div class="kpi-label">Total Jogos</div>
            <div class="kpi-value">{total}</div>
          </div>
          <div class="kpi-card green">
            <div class="kpi-label">Vitórias</div>
            <div class="kpi-value">{wins}</div>
            <div class="kpi-sub">{wins/total*100:.1f}%</div>
          </div>
          <div class="kpi-card yellow">
            <div class="kpi-label">Empates</div>
            <div class="kpi-value">{draws}</div>
            <div class="kpi-sub">{draws/total*100:.1f}%</div>
          </div>
          <div class="kpi-card red">
            <div class="kpi-label">Derrotas</div>
            <div class="kpi-value">{losses}</div>
            <div class="kpi-sub">{losses/total*100:.1f}%</div>
          </div>
          <div class="kpi-card blue">
            <div class="kpi-label">Gols/Jogo</div>
            <div class="kpi-value">{avg_gf:.2f}</div>
            <div class="kpi-sub">marcados</div>
          </div>
          <div class="kpi-card red">
            <div class="kpi-label">Sofridos/Jogo</div>
            <div class="kpi-value">{avg_gc:.2f}</div>
            <div class="kpi-sub">concedidos</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-title">Resultados Casa vs Fora</div>', unsafe_allow_html=True)
            vc = all_team.groupby(["Venue","TeamResult"]).size().reset_index(name="count")
            vc["pct"] = vc["count"] / vc.groupby("Venue")["count"].transform("sum") * 100
            fig = px.bar(
                vc, x="Venue", y="pct", color="TeamResult",
                color_discrete_map={"Vitória":C["green"],"Empate":C["draw"],"Derrota":C["away"]},
                barmode="stack",
                text=vc["pct"].apply(lambda x: f"{x:.1f}%"),
                labels={"pct":"%","Venue":"","TeamResult":""},
            )
            fig.update_traces(textposition="inside", textfont_size=12)
            fig.update_layout(bargap=0.3)
            theme(fig, 360)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-title">Gols Marcados vs Sofridos</div>', unsafe_allow_html=True)
            gt = all_team.dropna(subset=["GF","GC"]).groupby("Season").agg(
                GF=("GF","mean"), GC=("GC","mean")
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=gt["Season"], y=gt["GF"],
                                  name="Marcados", marker_color=C["green"],
                                  hovertemplate="<b>%{x}</b><br>Marcados: %{y:.2f}<extra></extra>"))
            fig.add_trace(go.Bar(x=gt["Season"], y=-gt["GC"],
                                  name="Sofridos", marker_color=C["away"],
                                  hovertemplate="<b>%{x}</b><br>Sofridos: %{y:.2f}<extra></extra>"))
            fig.add_hline(y=0, line_color=C["muted"], line_width=1)
            fig.update_layout(barmode="overlay", yaxis_title="Gols/Jogo")
            theme(fig, 360)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Evolução do Rating Elo</div>', unsafe_allow_html=True)
        team_elo = all_team.dropna(subset=["HomeElo","AwayElo","MatchDate"]).copy()
        team_elo["TeamElo"] = np.where(team_elo["Venue"]=="Casa", team_elo["HomeElo"], team_elo["AwayElo"])
        if not team_elo.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=team_elo["MatchDate"], y=team_elo["TeamElo"],
                mode="lines",
                line=dict(color=C["home"], width=2),
                fill="tozeroy",
                fillcolor="rgba(79,158,255,0.07)",
                hovertemplate="%{x|%Y-%m-%d}<br><b>Elo: %{y:.0f}</b><extra></extra>",
            ))
            fig.update_layout(xaxis_title="Data", yaxis_title="Rating Elo")
            theme(fig, 300)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Últimos 20 Jogos</div>', unsafe_allow_html=True)
        recent = all_team.tail(20)[[
            "MatchDate","Venue","HomeTeam","AwayTeam",
            "FTHome","FTAway","TeamResult","OddHome","OddDraw","OddAway"
        ]].copy()
        recent["MatchDate"] = recent["MatchDate"].dt.strftime("%Y-%m-%d")
        recent.columns = ["Data","Local","Casa","Fora","GF Casa","GF Fora","Resultado","Odd H","Odd D","Odd A"]

        def color_result(val):
            if val == "Vitória":  return f"color: {C['green']}; font-weight:700"
            if val == "Derrota": return f"color: {C['away']}; font-weight:700"
            return f"color: {C['draw']}; font-weight:600"

        st.dataframe(
            recent.style.applymap(color_result, subset=["Resultado"])
                        .format({"Odd H":"{:.2f}","Odd D":"{:.2f}","Odd A":"{:.2f}"}, na_rep="-"),
            use_container_width=True, hide_index=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    ⚽ Football Betting Value Finder &nbsp;·&nbsp;
    230k+ jogos · 2000–2025 · 27 países · 42 ligas &nbsp;·&nbsp;
    Streamlit + XGBoost + Plotly
</div>
""", unsafe_allow_html=True)
