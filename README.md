# ⚽ Football Betting Value Finder

> **Identificação de Value Bets com Dados Históricos** — análise de ineficiências no mercado de apostas esportivas usando dados de 230.000+ jogos, ratings Elo e machine learning.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189AB4?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75?style=flat-square&logo=plotly&logoColor=white)

---

## 🎯 O Projeto

A ideia central é replicar o raciocínio inverso de um **Analista de Risco** de casa de apostas: identificar onde o mercado estava *errado* — ou seja, onde a probabilidade implícita de uma odd era **menor** que a probabilidade real do evento.

Isso é exatamente o que define um **value bet**:

```
Value = (Probabilidade Real × Odd) − 1 > 0
```

Para estimar a probabilidade "real", o projeto usa o **modelo Elo** — o mesmo sistema de rating utilizado no xadrez e adaptado para o futebol — como alternativa independente à precificação das casas de apostas.

---

## 📊 Dataset

| Arquivo | Descrição |
|--------|-----------|
| `data/Matches.csv` | 230.000+ jogos de 27 países e 42 ligas (2000–2025) |
| `data/EloRatings.csv` | Ratings Elo de ~500 times europeus, atualizados 2x/mês |

**Ligas incluídas:** Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Pro League, Primeira Liga, Süper Lig e mais.

**Principais colunas de `Matches.csv`:**

| Coluna | Descrição |
|--------|-----------|
| `Division` | Código da liga (ex: `E0` = Premier League) |
| `MatchDate` | Data da partida |
| `HomeTeam / AwayTeam` | Nome dos times |
| `HomeElo / AwayElo` | Rating Elo na data do jogo |
| `FTHome / FTAway` | Gols no tempo regulamentar |
| `FTResult` | Resultado (`H`, `D`, `A`) |
| `OddHome / OddDraw / OddAway` | Odds médias do mercado |
| `MaxHome / MaxDraw / MaxAway` | Odds máximas disponíveis |
| `Over25 / Under25` | Odds para Over/Under 2.5 gols |
| `Form3Home / Form5Away...` | Forma recente (últimos 3 e 5 jogos) |

---

## 🚀 Como Rodar

### 1. Clone o repositório

```bash
git clone https://github.com/ricardohenrique1609/Football-Forecast.git
cd Football-Forecast
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Inicie o dashboard

```bash
python -m streamlit run app.py
```

Ou clique duas vezes em **`run.bat`** (Windows).

O app abre em `http://localhost:8501`.

---

## 🧠 Análises e Funcionalidades

### 📊 Visão Geral
- Distribuição de resultados (donut + barras por liga)
- Tendência de vitórias em casa / empates / vitórias fora ao longo das temporadas
- Distribuição das odds e média de gols por temporada

### 💰 Value Bets
- **ROI por mercado × estratégia**: compara apostar em tudo vs filtrar por edge Elo ≥ 3% e ≥ 5%
- **Calibração**: probabilidade implícita da odd vs taxa real de vitórias
- **ROI por liga** com filtro de edge do modelo Elo
- **Win rate por diferença Elo** (mandante muito mais forte vs visitante favorito)
- **Curva de edge**: quanto maior o edge do modelo, maior a taxa real de acerto?

### 📈 Eficiência de Mercado
- Margem (vig) média por liga — quem cobra mais caro?
- Mercado Over 2.5: probabilidade implícita vs real, por temporada e por liga
- Evolução das odds médias ao longo do tempo
- Heatmap: % de vitórias em casa por liga × temporada

### 🤖 Modelo XGBoost
- Treinado com features: Elo mandante/visitante, diferença Elo, forma recente, fair probabilities
- **Métricas**: AUC macro OvR, Log Loss, precisão, recall, F1
- **Feature importance** (gain)
- **Matriz de confusão** normalizada
- **Curvas de calibração** por classe
- **ROI simulado**: aposta apenas quando o modelo supera a fair prob por um edge mínimo configurável
- Histogramas de distribuição do edge (acertos vs erros)

### 🔍 Explorador de Times
- Busca qualquer time do dataset
- Resultados em casa vs fora (%)
- Gols marcados vs sofridos por temporada
- Evolução do rating Elo ao longo do tempo
- Tabela dos últimos 20 jogos com destaque por resultado

---

## 🏗️ Estrutura do Projeto

```
Football-Forecast/
├── app.py                  # Dashboard Streamlit (ponto de entrada)
├── run.bat                 # Lançador Windows
├── requirements.txt        # Dependências Python
├── data/
│   ├── Matches.csv         # Dataset principal de partidas
│   └── EloRatings.csv      # Ratings Elo históricos
└── src/
    ├── __init__.py
    ├── data_loader.py       # Carga, pré-processamento e cache
    ├── value_bets.py        # Modelo Elo, value bets, ROI, calibração
    └── ml_model.py          # XGBoost: treino, métricas, ROI simulado
```

---

## ⚙️ Detalhes Técnicos

### Modelo Elo para probabilidade
```python
P_home = 1 / (1 + 10 ^ (-(EloDiff + 65) / 400))
```
O ajuste de +65 pontos representa a **vantagem de mandante** estimada empiricamente.

Para o futebol (3 resultados possíveis), a probabilidade de empate é estimada como:
```python
P_draw ≈ 0.28 × 4 × P_home × P_away   (máx ≈ 28% em jogos equilibrados)
```

### Probabilidade Justa (sem vig)
```python
FairHome = (1/OddHome) / (1/OddHome + 1/OddDraw + 1/OddAway)
```

### Edge do Modelo
```python
Edge = EloProb − FairProb
```
Um edge positivo indica que o modelo acredita que o evento é mais provável do que o mercado está precificando.

---

## 📦 Dependências

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.20.0
xgboost>=2.0.0
scikit-learn>=1.4.0
```

---

## 📄 Licença

MIT License — livre para uso, estudo e adaptação.

---

<div align="center">
  Desenvolvido com ⚽ · Python + Streamlit + XGBoost + Plotly
</div>
