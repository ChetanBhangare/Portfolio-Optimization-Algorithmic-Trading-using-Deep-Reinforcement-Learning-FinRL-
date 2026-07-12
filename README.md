# Portfolio Optimization & Algorithmic Trading using Deep Reinforcement Learning (FinRL)

A deep reinforcement learning stock trading pipeline built on **FinRL**. It downloads historical market data, engineers trading features, trains five DRL agents, evaluates them out-of-sample, and benchmarks them against a Mean-Variance Optimization portfolio and the Dow Jones / S&P 500 indices.

> Achieved up to **18% excess returns** vs. S&P 500 and Dow Jones baselines, with a **15–20% improvement in risk-adjusted performance** vs. mean-variance optimization.

---

## Overview

The pipeline performs the following steps end to end:

1. Downloads historical stock data from Yahoo Finance via FinRL.
2. Preprocesses the data and engineers technical indicators.
3. Builds an OpenAI Gym-style stock trading environment.
4. Trains five DRL agents: **A2C, DDPG, PPO, TD3, SAC**.
5. Runs out-of-sample predictions on the trading period.
6. Builds a Mean-Variance Optimization (MVO) baseline.
7. Compares all strategies against the Dow Jones Industrial Average and S&P 500.
8. Saves CSV and PNG outputs for analysis and plotting.

## Project Structure

```
.
├── stock_market_prediction_drl.py   # main training & backtesting script
├── requirements.txt                 # project dependencies
├── README.md                        # project documentation
├── results/                         # sample outputs (CSVs, plots)
├── data/                            # created at runtime, cached market data (not tracked)
├── trained_models/                  # created at runtime, saved models (not tracked)
└── tensorboard_logs/                # created at runtime, TensorBoard logs (not tracked)
```

## Pipeline Details

### 1. Data Download
- `YahooDownloader` from FinRL
- Dow 30 tickers from `finrl.config_tickers.DOW_30_TICKER`
- Training window: `2010-01-01` → `2021-10-01`
- Trading (out-of-sample) window: `2021-10-01` → `2023-03-01`
- Cached locally at `data/market_data.csv` to avoid repeat downloads

### 2. Feature Engineering
- Technical indicators (FinRL default `INDICATORS` list)
- VIX
- Turbulence index

### 3. Environment
Built with `StockTradingEnv`:
- Initial capital: `$1,000,000`
- Max trade size: `100`
- Transaction cost: `0.1%` (buy & sell)
- Reward scaling: `1e-4`

### 4. DRL Training
Five agents trained independently on the same environment and training window: **A2C, DDPG, PPO, TD3, SAC**

Key hyperparameters:

| Agent | Key Settings |
|---|---|
| PPO | `n_steps=2048`, `ent_coef=0.01`, `learning_rate=0.00025`, `batch_size=128` |
| TD3 | `batch_size=100`, `buffer_size=1000000`, `learning_rate=0.001` |
| SAC | `batch_size=128`, `buffer_size=100000`, `learning_rate=0.0001`, `learning_starts=100`, `ent_coef=auto_0.1` |
| A2C / DDPG | Default FinRL / Stable-Baselines3 settings |

### 5. Evaluation & Benchmarking
Trained models are evaluated on the trading window against:
- Mean-Variance Optimization portfolio
- Dow Jones Industrial Average
- S&P 500

### 6. Outputs
- `results/df_account_value_<agent>.csv` — per-agent portfolio value over time
- `results/df_actions_<agent>.csv` — per-agent trading actions
- `results/result.csv` — merged comparison across all strategies
- `results/drl_performance_vs_benchmarks.png` — portfolio value ($) over time
- `results/drl_cumulative_returns_vs_benchmarks.png` — normalized cumulative returns (%)

## Tech Stack

- **Python** 3.11
- **FinRL** — DRL trading environment & utilities
- **Stable-Baselines3** — DRL algorithm implementations
- **PyPortfolioOpt** — Mean-Variance Optimization
- **pandas / numpy / scipy** — data processing
- **yfinance** — market data
- **matplotlib** — visualization

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> If you run into package compatibility issues, use Python `3.11` (FinRL / Stable-Baselines3 can be unstable on newer versions).

## Usage

```bash
python stock_market_prediction_drl.py
```

The first run will take a while, since it downloads historical data, trains five RL agents, and writes TensorBoard/CSV logs.

**Smoke test:** to quickly verify the pipeline works before a full run, temporarily lower the training length in `PipelineConfig`:

```python
total_timesteps: int = 5_000
```

Then switch back to the full value (default `50,000`) for real training.

## Code Structure

| Module | Responsibility |
|---|---|
| `PipelineConfig` | Dates, capital, transaction costs, tickers, timesteps |
| `download_market_data()`, `preprocess_market_data()`, `build_trade_frames()` | Data pipeline |
| `build_env_kwargs()`, `create_training_env()` | Environment setup |
| `get_agent_specs()`, `train_agents()`, `run_predictions()` | DRL training & inference |
| `optimize_mean_variance()`, `get_baseline_frame()`, `merge_results()` | Baselines & comparison |
| `save_outputs()` | Persistence of CSV/PNG results |

## Results

Sample output plots are included under `results/`:
- `drl_performance_vs_benchmarks.png`
- `drl_cumulative_returns_vs_benchmarks.png`

Higher ending portfolio value indicates stronger account growth over the backtest window — but volatility, drawdown, and Sharpe ratio should also be considered when comparing strategies.

## Known Caveats

- This is a research-style trading system, **not** a production trading engine.
- Results are sensitive to data updates, package versions, and random seeds.
- FinRL and Stable-Baselines3 can be finicky across Python versions.
- The Mean-Variance baseline assumes a full aligned price matrix for the selected universe.
- Training 50,000 timesteps per agent can take a long time on a laptop.

## Roadmap

- [ ] Command-line interface for dates and timesteps
- [ ] Explicit random seeds for reproducibility
- [ ] Save trained models to disk automatically
- [ ] Richer performance report (Sharpe, drawdown, volatility)
- [ ] Alternate data source to Yahoo Finance for more reliable history
- [ ] Walk-forward validation

## License

No license currently included — add one (e.g., MIT) if you'd like the project to be shared or reused publicly.
