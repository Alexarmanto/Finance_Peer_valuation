"""
=============================================================================
  Peer Analysis Valuation Tool  -  Production-Ready Quant Implementation
=============================================================================
  Architecture : DataHandler  ->  ValuationEngine  ->  plot_results()
  Optimiser    : Optuna (TPE sampler, composite-multiple weight search)
  Stack        : pandas, yfinance, optuna, matplotlib, seaborn, sklearn
=============================================================================
"""

from __future__ import annotations   # Python 3.7-3.9 type-hint compatibility

import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import yfinance as yf
import optuna
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# CLASS 1 - DataHandler
# =============================================================================
class DataHandler:
    """
    Responsible for all market-data acquisition and pre-processing.

    LEARN: In peer (comparable-company) analysis we normalise value by
    dividing market price by an earnings or cash-flow metric.  This ratio,
    the 'multiple', allows apples-to-apples comparison across firms of
    different absolute sizes.  If a peer group's average multiple is higher
    than the target's, the target is relatively cheap (multiple contraction
    risk is low); if lower, the target looks expensive (multiple expansion
    potential is limited).
    """

    FIELDS = {
        "trailingPE": "PE Ratio (TTM)",
        "enterpriseToEbitda": "EV/EBITDA",
        "marketCap": "Market Cap",
    }

    # LEARN: In a live system you would replace this static table with a
    # Bloomberg, Refinitiv, or FactSet API call.  The *structure* of the
    # pipeline is identical regardless of data source - a key OOP design
    # benefit.  Values below are approximate Q1-2025 figures from 10-Ks.
    _FALLBACK = {
        "AAPL":  {"trailingPE": 31.2, "enterpriseToEbitda": 23.5, "marketCap": 3_400_000_000_000},
        "MSFT":  {"trailingPE": 34.8, "enterpriseToEbitda": 25.1, "marketCap": 3_100_000_000_000},
        "GOOGL": {"trailingPE": 22.6, "enterpriseToEbitda": 16.8, "marketCap": 2_100_000_000_000},
        "META":  {"trailingPE": 24.1, "enterpriseToEbitda": 17.9, "marketCap": 1_500_000_000_000},
        "AMZN":  {"trailingPE": 41.3, "enterpriseToEbitda": 19.4, "marketCap": 2_200_000_000_000},
        "NVDA":  {"trailingPE": 53.0, "enterpriseToEbitda": 44.2, "marketCap": 2_800_000_000_000},
        "TSLA":  {"trailingPE": 48.5, "enterpriseToEbitda": 31.6, "marketCap":   700_000_000_000},
    }

    def __init__(self, tickers):
        self.tickers = [t.upper().strip() for t in tickers]
        self.raw_df = None
        self.clean_df = None

    def fetch(self):
        """
        Download fundamentals via yfinance; fall back to curated static
        data if the network is unavailable; return self for method chaining.
        """
        logger.info("Fetching fundamentals for %d tickers ...", len(self.tickers))
        records = []

        for ticker in self.tickers:
            try:
                info = yf.Ticker(ticker).info
                # LEARN: yfinance may return None or an empty dict for
                # illiquid / delisted tickers.  We raise explicitly so the
                # except block can attempt the fallback rather than silently
                # propagating bad data downstream.
                if not info or info.get("trailingPE") is None:
                    raise ValueError("Empty or incomplete info dict")

                row = {"Ticker": ticker}
                for field in self.FIELDS:
                    row[field] = info.get(field, np.nan)

                records.append(row)
                logger.info(
                    "  OK [live]     %-6s  PE=%.1f  EV/EBITDA=%.1f  MCap=$%.1fB",
                    ticker,
                    row["trailingPE"],
                    row["enterpriseToEbitda"],
                    row["marketCap"] / 1e9,
                )

            except Exception as exc:
                if ticker in self._FALLBACK:
                    row = {"Ticker": ticker}
                    row.update(self._FALLBACK[ticker])
                    records.append(row)
                    logger.info(
                        "  OK [fallback] %-6s  PE=%.1f  EV/EBITDA=%.1f  MCap=$%.1fB",
                        ticker,
                        row["trailingPE"],
                        row["enterpriseToEbitda"],
                        row["marketCap"] / 1e9,
                    )
                else:
                    logger.warning("  SKIP %s - no fallback available: %s", ticker, exc)

        if not records:
            raise RuntimeError(
                "No data retrieved for any ticker. "
                "Check network connection or extend _FALLBACK."
            )

        self.raw_df = pd.DataFrame(records).set_index("Ticker")
        return self

    def clean(self):
        """
        Drop rows with any missing fundamental and log removals.

        LEARN: A single NaN in either multiple renders the composite score
        undefined.  In production you might impute with sector medians, but
        for a peer group of similar large-caps, dropping is safer than
        introducing look-ahead bias.
        """
        if self.raw_df is None:
            raise RuntimeError("Call .fetch() before .clean()")

        before = len(self.raw_df)
        self.clean_df = self.raw_df.dropna(subset=list(self.FIELDS.keys())).copy()
        dropped = before - len(self.clean_df)

        if dropped:
            logger.warning("Dropped %d ticker(s) with missing data.", dropped)

        logger.info("Clean dataset: %d tickers retained.", len(self.clean_df))
        return self

    def split(self, test_size=0.30, random_state=42):
        """
        70 / 30 In-Sample / Out-of-Sample split on the peer list.

        LEARN: Holding out a subset prevents the model from simply
        memorising the training peers.  The out-of-sample tickers become
        the 'target companies' whose fair value we infer - mirroring
        real-world practice where an analyst calibrates a model on traded
        comps and then applies it to an unpriced asset (e.g. a pre-IPO firm).
        """
        if self.clean_df is None:
            raise RuntimeError("Call .clean() before .split()")

        tickers = self.clean_df.index.tolist()
        if len(tickers) < 4:
            raise ValueError(
                "Need >= 4 clean tickers for a meaningful split; got {}.".format(len(tickers))
            )

        in_tickers, out_tickers = train_test_split(
            tickers, test_size=test_size, random_state=random_state
        )

        in_sample = self.clean_df.loc[in_tickers]
        out_sample = self.clean_df.loc[out_tickers]

        logger.info(
            "Split -> In-Sample: %s  |  Out-of-Sample: %s",
            in_sample.index.tolist(),
            out_sample.index.tolist(),
        )
        return in_sample, out_sample


# =============================================================================
# CLASS 2 - ValuationEngine
# =============================================================================
class ValuationEngine:
    """
    Fits a composite multiple model and estimates fair-value market caps.

    Model
    -----
        Score_i  =  w1 * PE_i  +  w2 * EV/EBITDA_i

    Fair-Value Market Cap
    ---------------------
        FV_MCap_i  =  (Score_i / Score_mean) * MCap_mean

    where the mean is taken over the *in-sample* peers.

    LEARN: EV/EBITDA is preferred over PE in many contexts because it is
    capital-structure neutral - it compares firms regardless of leverage.
    A highly-levered firm has depressed earnings (interest expense), making
    its PE appear cheap.  EV/EBITDA corrects for this by adding net debt
    back into the numerator, giving a cleaner signal of operating value.
    Including both metrics in a weighted composite hedges against situations
    where one ratio is distorted (e.g. one-time write-downs affecting PE).
    """

    def __init__(self, n_trials=200, random_state=42):
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_w1 = None
        self.best_w2 = None
        self.study = None
        self._in_sample = None

    @staticmethod
    def _composite_score(df, w1, w2):
        """Weighted sum of the two multiples for every ticker in df."""
        return w1 * df["trailingPE"] + w2 * df["enterpriseToEbitda"]

    @staticmethod
    def _mape(actual, predicted):
        """
        Mean Absolute Percentage Error - scale-invariant loss function.

        LEARN: MAPE is the natural loss for valuation work because we care
        about *relative* mis-pricing.  A $10B error on a $1T company is
        trivial; the same error on a $20B company is catastrophic.  MAPE
        forces the optimiser to treat all peers symmetrically by size.
        """
        mask = actual != 0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

    def _objective(self, trial):
        """
        Optuna objective: search over (w1, w2) to minimise in-sample MAPE.

        LEARN: We constrain both weights to [0, 1] and normalise them so
        the composite score is a convex combination of the two multiples.
        This prevents the optimiser from inflating one weight arbitrarily
        while collapsing the other, keeping the model interpretable.
        """
        w1_raw = trial.suggest_float("w1", 0.0, 1.0)
        w2_raw = trial.suggest_float("w2", 0.0, 1.0)

        total = w1_raw + w2_raw + 1e-9
        w1 = w1_raw / total
        w2 = w2_raw / total

        df = self._in_sample
        scores = self._composite_score(df, w1, w2)
        mean_score = scores.mean()
        mean_mcap = df["marketCap"].mean()

        if mean_score <= 0:
            return 1e9

        fv_mcap = (scores / mean_score) * mean_mcap
        return self._mape(df["marketCap"].values, fv_mcap.values)

    def fit(self, in_sample):
        """Run Optuna optimisation on the in-sample peer group."""
        self._in_sample = in_sample
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.study.optimize(self._objective, n_trials=self.n_trials)

        best = self.study.best_params
        total = best["w1"] + best["w2"] + 1e-9
        self.best_w1 = best["w1"] / total
        self.best_w2 = best["w2"] / total

        logger.info(
            "Optimisation complete -> w1(PE)=%.4f  w2(EV/EBITDA)=%.4f  In-Sample MAPE=%.2f%%",
            self.best_w1,
            self.best_w2,
            self.study.best_value,
        )
        return self

    def predict(self, df):
        """
        Apply fitted weights to any peer DataFrame; return enriched copy.

        LEARN: The anchor for absolute fair-value is the in-sample mean
        market cap mapped to the in-sample mean composite score.  We then
        extrapolate: if an out-of-sample peer's composite score is 20% above
        the in-sample mean, its fair-value MCap is 20% above the mean MCap.
        This is the essence of *relative valuation* - anchor on traded
        comps, not on DCF assumptions.
        """
        if self.best_w1 is None:
            raise RuntimeError("Call .fit() before .predict()")

        result = df.copy()
        in_scores = self._composite_score(self._in_sample, self.best_w1, self.best_w2)
        mean_score = in_scores.mean()
        mean_mcap = self._in_sample["marketCap"].mean()

        out_scores = self._composite_score(result, self.best_w1, self.best_w2)
        result["CompositeScore"] = out_scores
        result["FairValueMCap"] = (out_scores / mean_score) * mean_mcap
        result["PriceFairValue_ratio"] = result["FairValueMCap"] / result["marketCap"]

        oos_mape = self._mape(result["marketCap"].values, result["FairValueMCap"].values)
        logger.info("Out-of-Sample MAPE = %.2f%%", oos_mape)
        return result, oos_mape


# =============================================================================
# Visualiser
# =============================================================================
def plot_results(out_df, best_w1, best_w2, oos_mape, save_path="result_plot.png", dpi=300):
    """
    Seaborn grouped bar chart: Actual Market Cap vs Model Fair Value.

    LEARN: Visualising both values side-by-side immediately shows whether
    a stock is trading at a premium or discount to its model fair value.
    The percentage delta annotation quantifies the mis-pricing at a glance,
    which is exactly the output an equity analyst would present to a PM.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = {
        "Market Cap (Actual)": "#2E86AB",
        "Fair Value (Model)": "#E84855",
    }

    plot_data = pd.melt(
        out_df.reset_index()[["Ticker", "marketCap", "FairValueMCap"]],
        id_vars="Ticker",
        value_vars=["marketCap", "FairValueMCap"],
        var_name="Measure",
        value_name="Value_B",
    )
    plot_data["Value_B"] = plot_data["Value_B"] / 1e9
    plot_data["Measure"] = plot_data["Measure"].map(
        {"marketCap": "Market Cap (Actual)", "FairValueMCap": "Fair Value (Model)"}
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=plot_data,
        x="Ticker",
        y="Value_B",
        hue="Measure",
        palette=palette,
        ax=ax,
        edgecolor="white",
        linewidth=0.6,
    )

    for i, ticker in enumerate(out_df.index.tolist()):
        actual = out_df.loc[ticker, "marketCap"] / 1e9
        model = out_df.loc[ticker, "FairValueMCap"] / 1e9
        diff_pct = (model - actual) / actual * 100
        max_val = max(actual, model)
        colour = "#27AE60" if diff_pct >= 0 else "#C0392B"
        sign = "+" if diff_pct >= 0 else ""
        ax.text(
            i,
            max_val + max_val * 0.03,
            "{}{:.1f}%".format(sign, diff_pct),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=colour,
        )

    ax.set_title(
        "Peer Analysis Valuation - Out-of-Sample Comparison\n"
        "Composite Score = {:.2f} x PE  +  {:.2f} x EV/EBITDA".format(best_w1, best_w2),
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Ticker", fontsize=11)
    ax.set_ylabel("Market Cap  (USD Billions)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: "${:.0f}B".format(x)))
    ax.legend(title="", loc="upper right", framealpha=0.85)

    ax.text(
        0.01,
        0.97,
        "OOS MAPE: {:.1f}%".format(oos_mape),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3CD", edgecolor="#DAA520", alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    logger.info("Plot saved -> %s  (DPI=%d)", save_path, dpi)
    plt.close(fig)


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    logger.info("=" * 60)
    logger.info("  Peer Analysis Valuation Tool  -  Pipeline Start")
    logger.info("=" * 60)

    # LEARN: Peer selection is the most consequential step in comp analysis.
    # We use US mega-cap tech because all companies have reliable public
    # financials, high analyst coverage, and well-defined earnings - this
    # minimises data quality risk in the peer group.
    TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA"]

    # Step 1 - Data
    handler = DataHandler(TICKERS)
    handler.fetch()
    handler.clean()
    in_sample, out_sample = handler.split(test_size=0.30, random_state=42)

    print("\nIn-Sample peers   :", in_sample.index.tolist())
    print("Out-of-Sample peers:", out_sample.index.tolist(), "\n")

    # Step 2 - Fit
    engine = ValuationEngine(n_trials=300, random_state=42)
    engine.fit(in_sample)

    # Step 3 - Predict
    predictions, oos_mape = engine.predict(out_sample)

    # Step 4 - Report
    print("\n" + "=" * 65)
    print("  OUT-OF-SAMPLE VALUATION RESULTS")
    print("=" * 65)

    report = predictions[["marketCap", "FairValueMCap", "CompositeScore", "PriceFairValue_ratio"]].copy()
    report["marketCap"] = (report["marketCap"] / 1e9).map("${:.1f}B".format)
    report["FairValueMCap"] = (report["FairValueMCap"] / 1e9).map("${:.1f}B".format)
    report["CompositeScore"] = report["CompositeScore"].map("{:.2f}".format)
    report["PriceFairValue_ratio"] = report["PriceFairValue_ratio"].map("{:.3f}x".format)
    report.columns = ["Actual MCap", "Model Fair Value", "Composite Score", "Price/FV Ratio"]
    print(report.to_string())

    print("\n  Optimal weights -> w1(PE) = {:.4f}  |  w2(EV/EBITDA) = {:.4f}".format(
        engine.best_w1, engine.best_w2
    ))
    print("  Out-of-Sample MAPE = {:.2f}%".format(oos_mape))
    print("=" * 65)

    # Step 5 - Top-5 Optuna trials
    trials_df = (
        engine.study.trials_dataframe()
        .sort_values("value")
        .head(5)[["number", "value", "params_w1", "params_w2"]]
        .rename(columns={
            "number": "Trial",
            "value": "MAPE (%)",
            "params_w1": "w1 (raw)",
            "params_w2": "w2 (raw)",
        })
    )
    print("\n  Top-5 Optuna Trials (In-Sample MAPE)")
    print(trials_df.to_string(index=False))

    # Step 6 - Plot
    # LEARN: os.makedirs(exist_ok=True) is the safe, idempotent way to
    # ensure an output directory exists before writing to it - it never
    # raises an error if the folder was already created by a previous run.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "result_plot.png")

    plot_results(predictions, engine.best_w1, engine.best_w2, oos_mape, save_path=save_path)

    logger.info("Pipeline complete. Plot saved to: %s", save_path)


# =============================================================================
if __name__ == "__main__":
    main()


# =============================================================================
# THREE-SENTENCE LOGIC SUMMARY
# =============================================================================
# 1. The DataHandler fetches trailing PE and EV/EBITDA for a peer group via
#    yfinance (with a curated fallback for offline environments), discards any
#    ticker with incomplete data, and performs a 70/30 in-sample/out-of-sample
#    split so that model calibration and evaluation are cleanly separated.
#
# 2. The ValuationEngine uses Optuna's Tree-structured Parzen Estimator (TPE)
#    to find the convex combination of the two multiples (w1*PE + w2*EV/EBITDA)
#    that minimises MAPE between model-implied fair-value market caps and the
#    actual market caps on the in-sample peers.
#
# 3. The optimised weights are applied to the held-out peers to generate
#    out-of-sample fair-value estimates, and a grouped Seaborn bar chart
#    annotated with per-ticker premium/discount percentages and an overall
#    OOS MAPE badge is saved at 300 DPI for presentation-quality reporting.
# =============================================================================