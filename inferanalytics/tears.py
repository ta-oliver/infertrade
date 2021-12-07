#
# Copyright 2019 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from time import time
import empyrical as ep
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from inferanalytics import plotting
from inferanalytics import timeseries
from inferanalytics import utils

FACTOR_PARTITIONS = {
    "style": [
        "momentum",
        "size",
        "value",
        "reversal_short_term",
        "volatility",
    ],
    "sector": [
        "basic_materials",
        "consumer_cyclical",
        "financial_services",
        "real_estate",
        "consumer_defensive",
        "health_care",
        "utilities",
        "communication_services",
        "energy",
        "industrials",
        "technology",
    ],
}


def timer(msg_body, previous_time):
    current_time = time()
    run_time = current_time - previous_time
    message = "\nFinished " + msg_body + " (required {:.2f} seconds)."
    print(message.format(run_time))
    return current_time


def create_full_tear_sheet(
    returns,
    positions=None,
    transactions=None,
    market_data=None,
    benchmark_rets=None,
    slippage=None,
    live_start_date=None,
    sector_mappings=None,
    round_trips=False,
    estimate_intraday="infer",
    hide_positions=False,
    cone_std=(1.0, 1.5, 2.0),
    bootstrap=False,
    unadjusted_returns=None,
    turnover_denom="AGB",
    set_context=True,
    factor_returns=None,
    factor_loadings=None,
    pos_in_dollars=True,
    header_rows=None,
    factor_partitions=FACTOR_PARTITIONS,
    return_fig = True,
):

    ims = []
    im1=create_returns_tear_sheet(
        returns,
        positions=positions,
        transactions=transactions,
        live_start_date=live_start_date,
        cone_std=cone_std,
        benchmark_rets=benchmark_rets,
        bootstrap=bootstrap,
        turnover_denom=turnover_denom,
        header_rows=header_rows,
        set_context=set_context,
        return_fig=True,
    )

    ims.append(im1)

    im2=create_interesting_times_tear_sheet(
        returns, benchmark_rets=benchmark_rets, set_context=set_context,
        return_fig=True,
    )
    ims.append(im2)

    return ims

@plotting.customize
def create_returns_tear_sheet(
    returns,
    positions=None,
    transactions=None,
    live_start_date=None,
    cone_std=(1.0, 1.5, 2.0),
    benchmark_rets=None,
    bootstrap=False,
    turnover_denom="AGB",
    header_rows=None,
    return_fig=False,
):
    if benchmark_rets is not None:
        returns = utils.clip_returns_to_benchmark(returns, benchmark_rets)

    #plotting.show_worst_drawdown_periods(returns)

    vertical_sections = 11

    if benchmark_rets is not None:
        vertical_sections += 1

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_rolling_returns = plt.subplot(gs[:2, :])
    i = 2
    ax_rolling_returns_vol_match = plt.subplot(
        gs[i, :], sharex=ax_rolling_returns
    )
    i += 1
    ax_rolling_returns_log = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_returns = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    if benchmark_rets is not None:
        ax_rolling_beta = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
        i += 1
    ax_rolling_volatility = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_rolling_sharpe = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_drawdown = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_underwater = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_monthly_heatmap = plt.subplot(gs[i, 0])
    ax_annual_returns = plt.subplot(gs[i, 1])
    ax_monthly_dist = plt.subplot(gs[i, 2])
    i += 1
    ax_return_quantiles = plt.subplot(gs[i, :])
    i += 1

    plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        live_start_date=live_start_date,
        cone_std=cone_std,
        ax=ax_rolling_returns,
    )
    ax_rolling_returns.set_title("Cumulative returns")

    plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        live_start_date=live_start_date,
        cone_std=None,
        volatility_match=(benchmark_rets is not None),
        legend_loc=None,
        ax=ax_rolling_returns_vol_match,
    )
    ax_rolling_returns_vol_match.set_title(
        "Cumulative returns volatility matched to benchmark"
    )

    plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        logy=True,
        live_start_date=live_start_date,
        cone_std=cone_std,
        ax=ax_rolling_returns_log,
    )
    ax_rolling_returns_log.set_title("Cumulative returns on logarithmic scale")

    plotting.plot_returns(
        returns,
        live_start_date=live_start_date,
        ax=ax_returns,
    )
    ax_returns.set_title("Returns")

    if benchmark_rets is not None:
        plotting.plot_rolling_beta(returns, benchmark_rets, ax=ax_rolling_beta)

    plotting.plot_rolling_volatility(
        returns, factor_returns=benchmark_rets, ax=ax_rolling_volatility
    )

    plotting.plot_rolling_sharpe(returns, ax=ax_rolling_sharpe)

    # Drawdowns
    plotting.plot_drawdown_periods(returns, top=5, ax=ax_drawdown)

    plotting.plot_drawdown_underwater(returns=returns, ax=ax_underwater)

    plotting.plot_monthly_returns_heatmap(returns, ax=ax_monthly_heatmap)
    plotting.plot_annual_returns(returns, ax=ax_annual_returns)
    plotting.plot_monthly_returns_dist(returns, ax=ax_monthly_dist)

    plotting.plot_return_quantiles(
        returns, live_start_date=live_start_date, ax=ax_return_quantiles
    )

    for ax in fig.axes:
        ax.tick_params(
            axis="x",
            which="major",
            bottom=True,
            top=False,
            labelbottom=True,
        )

    if return_fig:
        return fig

@plotting.customize
def create_interesting_times_tear_sheet(
    returns,
    benchmark_rets=None,
    periods=None,
    legend_loc="best",
    return_fig=False,
):
    """
    Generate a number of returns plots around interesting points in time,
    like the flash crash and 9/11.

    Plots: returns around the dotcom bubble burst, Lehmann Brothers' failure,
    9/11, US downgrade and EU debt crisis, Fukushima meltdown, US housing
    bubble burst, EZB IR, Great Recession (August 2007, March and September
    of 2008, Q1 & Q2 2009), flash crash, April and October 2014.

    benchmark_rets must be passed, as it is meaningless to analyze performance
    during interesting times without some benchmark to refer to.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in create_full_tear_sheet.
    benchmark_rets : pd.Series
        Daily noncumulative returns of the benchmark.
         - This is in the same style as returns.
    periods: dict or OrderedDict, optional
        historical event dates that may have had significant
        impact on markets
    legend_loc : plt.legend_loc, optional
         The legend's location.
    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    rets_interesting = timeseries.extract_interesting_date_ranges(
        returns, periods
    )

    if not rets_interesting:
        warnings.warn(
            "Passed returns do not overlap with any" "interesting times.",
            UserWarning,
        )
        return

    if benchmark_rets is not None:
        returns = utils.clip_returns_to_benchmark(returns, benchmark_rets)

        bmark_interesting = timeseries.extract_interesting_date_ranges(
            benchmark_rets, periods
        )

    num_plots = len(rets_interesting)
    # 2 plots, 1 row; 3 plots, 2 rows; 4 plots, 2 rows; etc.
    num_rows = int((num_plots + 1) / 2.0)
    fig = plt.figure(figsize=(14, num_rows * 6.0))
    gs = gridspec.GridSpec(num_rows, 2, wspace=0.5, hspace=0.5)

    for i, (name, rets_period) in enumerate(rets_interesting.items()):
        # i=0 -> 0, i=1 -> 0, i=2 -> 1 ;; i=0 -> 0, i=1 -> 1, i=2 -> 0
        ax = plt.subplot(gs[int(i / 2.0), i % 2])

        ep.cum_returns(rets_period).plot(
            ax=ax, color="forestgreen", label="algo", alpha=0.7, lw=2
        )

        if benchmark_rets is not None:
            ep.cum_returns(bmark_interesting[name]).plot(
                ax=ax, color="gray", label="benchmark", alpha=0.6
            )
            ax.legend(
                ["Algo", "benchmark"],
                loc=legend_loc,
                frameon=True,
                framealpha=0.5,
            )
        else:
            ax.legend(["Algo"], loc=legend_loc, frameon=True, framealpha=0.5)

        ax.set_title(name)
        ax.set_ylabel("Returns")
        ax.set_xlabel("")

    if return_fig:
        return fig
