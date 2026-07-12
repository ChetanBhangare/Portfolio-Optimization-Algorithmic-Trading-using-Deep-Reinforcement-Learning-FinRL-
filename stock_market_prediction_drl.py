"""Clean FinRL pipeline for stock trading with five DRL agents.

This script turns the original notebook workflow into a maintainable module:
- one data download path
- one preprocessing path
- one environment builder
- one agent training loop
- one backtest loop
- one MVO baseline

The default data source is Yahoo Finance through FinRL's downloader.
If you later want to swap the data vendor, keep the rest of the pipeline and
replace only `download_market_data`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import itertools
import os
import sys
import sysconfig
import types

import numpy as np
import pandas as pd
import yfinance as yf

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_finrl_module(module_name: str, relative_path: str, finrl_root: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, finrl_root / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {finrl_root / relative_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap_finrl() -> None:
    purelib = Path(sysconfig.get_paths()["purelib"])
    finrl_root = purelib / "finrl"
    if not finrl_root.exists():
        raise ImportError(
            "FinRL is not installed in this environment. Install it first with `pip install finrl`."
        )

    if "finrl" not in sys.modules:
        finrl_pkg = types.ModuleType("finrl")
        finrl_pkg.__path__ = [str(finrl_root)]
        sys.modules["finrl"] = finrl_pkg

    config = _load_finrl_module("finrl.config", "config.py", finrl_root)
    tickers = _load_finrl_module("finrl.config_tickers", "config_tickers.py", finrl_root)
    main = _load_finrl_module("finrl.main", "main.py", finrl_root)

    finrl_pkg = sys.modules["finrl"]
    finrl_pkg.config = config
    finrl_pkg.config_tickers = tickers
    finrl_pkg.main = main


bootstrap_finrl()

from finrl.config_tickers import DOW_30_TICKER
from finrl.config import (
    DATA_SAVE_DIR,
    INDICATORS,
    RESULTS_DIR,
    TENSORBOARD_LOG_DIR,
    TRAINED_MODEL_DIR,
)
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats
from stable_baselines3.common.logger import configure
from pypfopt.efficient_frontier import EfficientFrontier


@dataclass(frozen=True)
class AgentSpec:
    name: str
    algo: str
    model_kwargs: Dict[str, object] = field(default_factory=dict)
    enabled: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    train_start_date: str = "2010-01-01"
    train_end_date: str = "2021-10-01"
    trade_start_date: str = "2021-10-01"
    trade_end_date: str = "2023-03-01"
    initial_amount: int = 1_000_000
    hmax: int = 100
    buy_cost_pct: float = 0.001
    sell_cost_pct: float = 0.001
    reward_scaling: float = 1e-4
    turbulence_threshold: float = 70.0
    tickers: List[str] = field(default_factory=lambda: list(DOW_30_TICKER))
    total_timesteps: int = 50_000
    cache_file: Path = Path(DATA_SAVE_DIR) / "market_data.csv"


def ensure_project_dirs() -> None:
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])


def download_market_data(
    start_date: str,
    end_date: str,
    tickers: Iterable[str],
    cache_file: Optional[Path] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download historical prices and optionally reuse a local cache."""
    ticker_list = list(tickers)
    if cache_file and use_cache and cache_file.exists():
        cached = pd.read_csv(cache_file, low_memory=False)
        cached["date"] = pd.to_datetime(cached["date"]).dt.strftime("%Y-%m-%d")
        return cached.sort_values(["date", "tic"]).reset_index(drop=True)

    frames = []
    failures = 0
    for tic in ticker_list:
        temp_df = yf.download(
            tic,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            actions=False,
            group_by="column",
            threads=True,
        )

        if temp_df is None or temp_df.empty:
            failures += 1
            continue

        temp_df = temp_df.reset_index()
        temp_df.columns = [c[0] if isinstance(c, tuple) else c for c in temp_df.columns]
        temp_df["tic"] = tic
        temp_df = temp_df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjcp",
                "Volume": "volume",
            }
        )

        if "adjcp" not in temp_df.columns:
            temp_df["adjcp"] = temp_df["close"]

        temp_df["date"] = pd.to_datetime(temp_df["date"]).dt.strftime("%Y-%m-%d")
        temp_df["day"] = pd.to_datetime(temp_df["date"]).dt.dayofweek

        numeric_cols = ["open", "high", "low", "close", "adjcp", "volume"]
        for col in numeric_cols:
            temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")

        temp_df["close"] = temp_df["adjcp"]
        temp_df = temp_df[["date", "open", "high", "low", "close", "adjcp", "volume", "tic", "day"]]
        temp_df = temp_df.dropna(subset=["open", "high", "low", "close", "adjcp", "volume"])
        frames.append(temp_df)

    if failures == len(ticker_list):
        raise ValueError("no data is fetched.")

    df = pd.concat(frames, axis=0).sort_values(["date", "tic"]).reset_index(drop=True)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)

    return df


def add_vix_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add VIX using the current yfinance API instead of FinRL's downloader."""
    vix = yf.download(
        "^VIX",
        start=df["date"].min(),
        end=df["date"].max(),
        auto_adjust=False,
        progress=False,
        actions=False,
        group_by="column",
        threads=True,
    )

    if vix is None or vix.empty:
        raise ValueError("No VIX data was fetched.")

    vix = vix.reset_index()
    vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
    vix = vix.rename(columns={"Date": "date", "Close": "vix"})
    vix["date"] = pd.to_datetime(vix["date"]).dt.strftime("%Y-%m-%d")
    vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")
    vix = vix[["date", "vix"]].dropna()

    df = df.merge(vix, on="date", how="left")
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    return df


def preprocess_market_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["open", "high", "low", "close", "adjcp", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "adjcp", "volume"])

    feature_engineer = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = feature_engineer.preprocess_data(df)
    processed = add_vix_feature(processed)

    tickers = processed["tic"].unique().tolist()
    dates = list(pd.date_range(processed["date"].min(), processed["date"].max()).astype(str))
    panel = pd.DataFrame(itertools.product(dates, tickers), columns=["date", "tic"])
    panel = panel.merge(processed, on=["date", "tic"], how="left")
    panel = panel[panel["date"].isin(processed["date"])]
    panel = panel.sort_values(["date", "tic"]).fillna(0).reset_index(drop=True)

    return panel


def build_trade_frames(processed: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data_split(processed, config.train_start_date, config.train_end_date)
    trade = data_split(processed, config.trade_start_date, config.trade_end_date)
    return train, trade


def build_env_kwargs(train: pd.DataFrame, config: PipelineConfig) -> Dict[str, object]:
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    buy_cost_list = [config.buy_cost_pct] * stock_dimension
    sell_cost_list = [config.sell_cost_pct] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    return {
        "hmax": config.hmax,
        "initial_amount": config.initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": config.reward_scaling,
    }


def create_training_env(train: pd.DataFrame, env_kwargs: Dict[str, object]) -> StockTradingEnv:
    return StockTradingEnv(df=train, **env_kwargs)


def get_agent_specs() -> List[AgentSpec]:
    return [
        AgentSpec("a2c", "a2c"),
        AgentSpec("ddpg", "ddpg"),
        AgentSpec("ppo", "ppo", {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}),
        AgentSpec("td3", "td3", {"batch_size": 100, "buffer_size": 1_000_000, "learning_rate": 0.001}),
        AgentSpec(
            "sac",
            "sac",
            {
                "batch_size": 128,
                "buffer_size": 100_000,
                "learning_rate": 0.0001,
                "learning_starts": 100,
                "ent_coef": "auto_0.1",
            },
        ),
    ]


def train_agents(
    env_train,
    specs: List[AgentSpec],
    total_timesteps: int,
) -> Dict[str, object]:
    models: Dict[str, object] = {}

    for spec in specs:
        if not spec.enabled:
            continue

        agent = DRLAgent(env=env_train)
        model = agent.get_model(spec.algo, model_kwargs=spec.model_kwargs)

        log_dir = Path(RESULTS_DIR) / spec.name
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
        except AssertionError:
            logger = configure(str(log_dir), ["stdout", "csv"])
        model.set_logger(logger)

        trained_model = agent.train_model(
            model=model,
            tb_log_name=spec.name,
            total_timesteps=total_timesteps,
        )
        models[spec.name] = trained_model

    return models


def run_predictions(models: Dict[str, object], trade_env) -> Dict[str, pd.DataFrame]:
    account_values: Dict[str, pd.DataFrame] = {}
    actions: Dict[str, pd.DataFrame] = {}

    for name, model in models.items():
        df_account_value, df_actions = DRLAgent.DRL_prediction(model=model, environment=trade_env)
        account_values[name] = df_account_value
        actions[name] = df_actions

    return account_values, actions


def normalize_date_index(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index).strftime("%Y-%m-%d")
    return normalized


def stock_returns_computing(stock_price: np.ndarray, rows: int, columns: int) -> np.ndarray:
    stock_return = np.zeros([rows - 1, columns])
    for j in range(columns):
        for i in range(rows - 1):
            stock_return[i, j] = ((stock_price[i + 1, j] - stock_price[i, j]) / stock_price[i, j]) * 100
    return stock_return


def optimize_mean_variance(
    mvo_df: pd.DataFrame,
    train_end_date: str,
    trade_start_date: str,
    initial_amount: int,
) -> pd.DataFrame:
    stock_data = mvo_df[mvo_df["date"] < train_end_date].copy()
    trade_data = mvo_df[mvo_df["date"] >= trade_start_date].copy()

    stock_prices = (
        stock_data.pivot(index="date", columns="tic", values="close")
        .sort_index()
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .dropna(axis=1, how="any")
    )
    trade_prices = (
        trade_data.pivot(index="date", columns="tic", values="close")
        .sort_index()
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )

    valid_tickers = stock_prices.columns.intersection(trade_prices.columns)
    stock_prices = stock_prices.loc[:, valid_tickers]
    trade_prices = trade_prices.loc[:, valid_tickers].ffill().bfill()

    last_price = stock_prices.iloc[-1].replace(0, np.nan)
    valid_tickers = last_price.dropna().index
    stock_prices = stock_prices.loc[:, valid_tickers]
    trade_prices = trade_prices.loc[:, valid_tickers]

    if stock_prices.empty or trade_prices.empty:
        raise ValueError("Mean-Variance baseline has no valid price data after cleaning.")

    returns = stock_prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    returns = returns.dropna(axis=1, how="any")
    trade_prices = trade_prices.loc[:, returns.columns]
    stock_prices = stock_prices.loc[:, returns.columns]

    if returns.empty:
        raise ValueError("Mean-Variance baseline has no valid returns after cleaning.")

    mean_returns = returns.mean()
    cov_returns = returns.cov()

    try:
        max_weight = 1.0 if len(mean_returns) == 1 else 0.5
        ef = EfficientFrontier(mean_returns, cov_returns, weight_bounds=(0, max_weight))
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        portfolio_weights = pd.Series(cleaned_weights, index=mean_returns.index, dtype=float)
    except Exception as exc:
        print(f"Warning: Mean-Variance optimization failed ({exc}). Falling back to equal weights.")
        portfolio_weights = pd.Series(1.0 / len(mean_returns), index=mean_returns.index)

    latest_prices = stock_prices.iloc[-1].loc[portfolio_weights.index]
    initial_portfolio = (portfolio_weights * initial_amount) / latest_prices
    portfolio_assets = (trade_prices.loc[:, initial_portfolio.index] @ initial_portfolio).to_frame("Mean Var")
    return normalize_date_index(portfolio_assets)


def get_index_baseline(
    ticker: str,
    column_name: str,
    trade_start_date: str,
    trade_end_date: str,
    initial_amount: int,
) -> pd.DataFrame:
    df_index = yf.download(
        ticker,
        start=trade_start_date,
        end=trade_end_date,
        auto_adjust=False,
        progress=False,
        actions=False,
        group_by="column",
        threads=True,
    )

    if df_index is None or df_index.empty:
        raise ValueError(f"No benchmark data was fetched for {ticker}.")

    df_index = df_index.reset_index()
    df_index.columns = [c[0] if isinstance(c, tuple) else c for c in df_index.columns]
    df_index = df_index.rename(columns={"Date": "date", "Close": "close"})
    df_index["date"] = pd.to_datetime(df_index["date"]).dt.strftime("%Y-%m-%d")
    df_index["close"] = pd.to_numeric(df_index["close"], errors="coerce")
    df_index = df_index[["date", "close"]].dropna()

    if df_index.empty:
        raise ValueError(f"Benchmark data for {ticker} has no valid close prices.")

    baseline = pd.DataFrame(index=df_index["date"])
    baseline[column_name] = (df_index["close"] / df_index["close"].iloc[0] * initial_amount).to_numpy()
    return baseline


def get_baseline_frame(trade_start_date: str, trade_end_date: str, initial_amount: int) -> pd.DataFrame:
    dji = get_index_baseline("^DJI", "dji", trade_start_date, trade_end_date, initial_amount)
    sp500 = get_index_baseline("^GSPC", "s&p 500", trade_start_date, trade_end_date, initial_amount)
    return dji.join(sp500, how="inner")


def merge_results(result_frames: Dict[str, pd.DataFrame], mvo_result: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    merged = None
    for key in ["a2c", "ddpg", "td3", "ppo", "sac"]:
        frame = result_frames[key].set_index(result_frames[key].columns[0])
        frame = normalize_date_index(frame)
        frame.columns = [key]
        merged = frame if merged is None else merged.join(frame, how="inner")

    merged = merged.join(mvo_result, how="inner")
    merged = merged.join(normalize_date_index(baseline), how="inner")
    merged.columns = ["a2c", "ddpg", "td3", "ppo", "sac", "mean var", "dji", "s&p 500"]
    return merged


def plot_performance(merged_result: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_frame = merged_result.copy()
    plot_frame.index = pd.to_datetime(plot_frame.index)

    display_names = {
        "a2c": "A2C",
        "ddpg": "DDPG",
        "td3": "TD3",
        "ppo": "PPO",
        "sac": "SAC",
        "mean var": "Mean-Variance",
        "dji": "DJI",
        "s&p 500": "S&P 500",
    }
    colors = {
        "a2c": "#1f77b4",
        "ddpg": "#ff7f0e",
        "td3": "#2ca02c",
        "ppo": "#d62728",
        "sac": "#9467bd",
        "mean var": "#8c564b",
        "dji": "#111111",
        "s&p 500": "#17becf",
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))
    for column in plot_frame.columns:
        ax.plot(
            plot_frame.index,
            plot_frame[column],
            label=display_names.get(column, column),
            color=colors.get(column),
            linewidth=2.2 if column in {"dji", "s&p 500"} else 1.8,
            linestyle="--" if column in {"dji", "s&p 500"} else "-",
        )
    ax.set_title("DRL Agent Portfolio Value vs Market Benchmarks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(ncol=2)
    ax.margins(x=0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "drl_performance_vs_benchmarks.png", dpi=180)
    plt.close(fig)

    normalized = plot_frame.divide(plot_frame.iloc[0]).subtract(1).multiply(100)
    fig, ax = plt.subplots(figsize=(14, 8))
    for column in normalized.columns:
        ax.plot(
            normalized.index,
            normalized[column],
            label=display_names.get(column, column),
            color=colors.get(column),
            linewidth=2.2 if column in {"dji", "s&p 500"} else 1.8,
            linestyle="--" if column in {"dji", "s&p 500"} else "-",
        )
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set_title("DRL Agent Cumulative Return vs Market Benchmarks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(ncol=2)
    ax.margins(x=0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "drl_cumulative_returns_vs_benchmarks.png", dpi=180)
    plt.close(fig)


def save_outputs(
    account_values: Dict[str, pd.DataFrame],
    actions: Dict[str, pd.DataFrame],
    merged_result: pd.DataFrame,
) -> None:
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in account_values.items():
        frame.to_csv(output_dir / f"df_account_value_{name}.csv", index=False)
    for name, frame in actions.items():
        frame.to_csv(output_dir / f"df_actions_{name}.csv", index=False)
    merged_result.to_csv(output_dir / "result.csv")
    plot_performance(merged_result, output_dir)


def main() -> None:
    config = PipelineConfig()
    ensure_project_dirs()

    raw = download_market_data(
        start_date=config.train_start_date,
        end_date=config.trade_end_date,
        tickers=config.tickers,
        cache_file=config.cache_file,
    )
    processed = preprocess_market_data(raw)
    train, trade = build_trade_frames(processed, config)

    env_kwargs = build_env_kwargs(train, config)
    train_env_wrapper = create_training_env(train, env_kwargs)
    env_train, _ = train_env_wrapper.get_sb_env()

    agent_specs = get_agent_specs()
    trained_models = train_agents(env_train, agent_specs, config.total_timesteps)

    trade_env = StockTradingEnv(df=trade, turbulence_threshold=config.turbulence_threshold, risk_indicator_col="vix", **env_kwargs)
    account_values, actions = run_predictions(trained_models, trade_env)

    mvo_df = processed.sort_values(["date", "tic"], ignore_index=True)[["date", "tic", "close"]]
    mvo_result = optimize_mean_variance(
        mvo_df,
        config.train_end_date,
        config.trade_start_date,
        config.initial_amount,
    )
    baseline = get_baseline_frame(config.trade_start_date, config.trade_end_date, config.initial_amount)

    merged_result = merge_results(account_values, mvo_result, baseline)
    save_outputs(account_values, actions, merged_result)

    print("Backtest summary:")
    print(merged_result.tail())

    # Optional diagnostics
    for name, frame in account_values.items():
        stats = backtest_stats(frame.set_index(frame.columns[0]), value_col_name=frame.columns[1])
        print(f"\n{name.upper()} stats")
        print(stats)


if __name__ == "__main__":
    main()
