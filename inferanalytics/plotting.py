#
# Copyright 2018 Quantopian, Inc.
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

import datetime
from collections import OrderedDict
from functools import wraps
import empyrical as ep
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import scipy as sp
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from inferanalytics import timeseries
from inferanalytics import utils
from inferanalytics.utils import APPROX_BDAYS_PER_MONTH, MM_DISPLAY_UNIT


def customize(func):
    """
    Decorator to set plotting context and axes style during function call.
    """

    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop("set_context", True)
        if set_context:
            with plotting_context(), axes_style():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return call_w_context


def plotting_context(context="notebook", font_scale=1.5, rc=None):
    if rc is None:
        rc = {}

    rc_default = {"lines.linewidth": 1.5}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.plotting_context(context=context, font_scale=font_scale, rc=rc)


def axes_style(style="darkgrid", rc=None):
    if rc is None:
        rc = {}

    rc_default = {}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.axes_style(style=style, rc=rc)


def plot_monthly_returns_heatmap(returns, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    monthly_ret_table = ep.aggregate_returns(returns, "monthly")
    monthly_ret_table = monthly_ret_table.unstack().round(3)

    sns.heatmap(
        monthly_ret_table.fillna(0) * 100.0,
        annot=True,
        annot_kws={"size": 9},
        alpha=1.0,
        center=0.0,
        cbar=False,
        cmap=matplotlib.cm.RdYlGn,
        ax=ax,
        **kwargs,
    )
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")
    ax.set_title("Monthly returns (%)")
    return ax


def plot_annual_returns(returns, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    x_axis_formatter = FuncFormatter(utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis="x", which="major")

    ann_ret_df = pd.DataFrame(ep.aggregate_returns(returns, "yearly"))

    ax.axvline(
        100 * ann_ret_df.values.mean(),
        color="steelblue",
        linestyle="--",
        lw=4,
        alpha=0.7,
    )
    (100 * ann_ret_df.sort_index(ascending=False)).plot(
        ax=ax, kind="barh", alpha=0.70, **kwargs
    )
    ax.axvline(0.0, color="black", linestyle="-", lw=3)

    ax.set_ylabel("Year")
    ax.set_xlabel("Returns")
    ax.set_title("Annual returns")
    ax.legend(["Mean"], frameon=True, framealpha=0.5)
    return ax

def plot_monthly_returns_dist(returns, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    x_axis_formatter = FuncFormatter(utils.percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis="x", which="major")

    monthly_ret_table = ep.aggregate_returns(returns, "monthly")
    ax.hist(
        100 * monthly_ret_table,
        color="orangered",
        alpha=0.80,
        bins=20,
        **kwargs,
    )
    ax.axvline(
        100 * monthly_ret_table.mean(),
        color="gold",
        linestyle="--",
        lw=4,
        alpha=1.0,
    )
    ax.axvline(0.0, color="black", linestyle="-", lw=3, alpha=0.75)
    ax.legend(["Mean"], frameon=True, framealpha=0.5)
    ax.set_ylabel("Number of months")
    ax.set_xlabel("Returns")
    ax.set_title("Distribution of monthly returns")
    return ax

def plot_drawdown_periods(returns, top=10, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
    df_drawdowns = timeseries.gen_drawdown_table(returns, top=top)

    df_cum_rets.plot(ax=ax, **kwargs)

    lim = ax.get_ylim()
    colors = sns.cubehelix_palette(len(df_drawdowns))[::-1]
    for i, (peak, recovery) in df_drawdowns[
        ["Peak date", "Recovery date"]
    ].iterrows():
        if pd.isnull(recovery):
            recovery = returns.index[-1]
        ax.fill_between(
            (peak, recovery), lim[0], lim[1], alpha=0.4, color=colors[i]
        )
    ax.set_ylim(lim)
    ax.set_title("Top %i drawdown periods" % top)
    ax.set_ylabel("Cumulative returns")
    ax.legend(["Portfolio"], loc="upper left", frameon=True, framealpha=0.5)
    ax.set_xlabel("")
    return ax

def plot_drawdown_underwater(returns, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.percentage)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
    running_max = np.maximum.accumulate(df_cum_rets)
    underwater = -100 * ((running_max - df_cum_rets) / running_max)
    (underwater).plot(ax=ax, kind="area", color="coral", alpha=0.7, **kwargs)
    ax.set_ylabel("Drawdown")
    ax.set_title("Underwater plot")
    ax.set_xlabel("")
    return ax



def plot_returns(returns, live_start_date=None, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_label("")
    ax.set_ylabel("Returns")

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        is_returns = returns.loc[returns.index < live_start_date]
        oos_returns = returns.loc[returns.index >= live_start_date]
        is_returns.plot(ax=ax, color="g")
        oos_returns.plot(ax=ax, color="r")

    else:
        returns.plot(ax=ax, color="g")

    return ax


def plot_rolling_returns(
    returns,
    factor_returns=None,
    live_start_date=None,
    logy=False,
    cone_std=None,
    legend_loc="best",
    volatility_match=False,
    cone_function=timeseries.forecast_cone_bootstrap,
    ax=None,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    ax.set_xlabel("")
    ax.set_ylabel("Cumulative returns")
    ax.set_yscale("log" if logy else "linear")

    if volatility_match and factor_returns is None:
        raise ValueError(
            "volatility_match requires passing of " "factor_returns."
        )
    elif volatility_match and factor_returns is not None:
        bmark_vol = factor_returns.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    cum_rets = ep.cum_returns(returns, 1.0)

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    if factor_returns is not None:
        cum_factor_returns = ep.cum_returns(
            factor_returns[cum_rets.index], 1.0
        )
        cum_factor_returns.plot(
            lw=2,
            color="gray",
            label=factor_returns.name,
            alpha=0.60,
            ax=ax,
            **kwargs,
        )

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        is_cum_returns = cum_rets.loc[cum_rets.index < live_start_date]
        oos_cum_returns = cum_rets.loc[cum_rets.index >= live_start_date]
    else:
        is_cum_returns = cum_rets
        oos_cum_returns = pd.Series([])

    is_cum_returns.plot(
        lw=3, color="forestgreen", alpha=0.6, label="Backtest", ax=ax, **kwargs
    )

    if len(oos_cum_returns) > 0:
        oos_cum_returns.plot(
            lw=4, color="red", alpha=0.6, label="Live", ax=ax, **kwargs
        )

        if cone_std is not None:
            if isinstance(cone_std, (float, int)):
                cone_std = [cone_std]

            is_returns = returns.loc[returns.index < live_start_date]
            cone_bounds = cone_function(
                is_returns,
                len(oos_cum_returns),
                cone_std=cone_std,
                starting_value=is_cum_returns[-1],
            )

            cone_bounds = cone_bounds.set_index(oos_cum_returns.index)
            for std in cone_std:
                ax.fill_between(
                    cone_bounds.index,
                    cone_bounds[float(std)],
                    cone_bounds[float(-std)],
                    color="steelblue",
                    alpha=0.5,
                )

    if legend_loc is not None:
        ax.legend(loc=legend_loc, frameon=True, framealpha=0.5)
    ax.axhline(1.0, linestyle="--", color="black", lw=2)

    return ax


def plot_rolling_beta(
    returns, factor_returns, legend_loc="best", ax=None, **kwargs
):
    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    ax.set_title("Rolling portfolio beta to " + str(factor_returns.name))
    ax.set_ylabel("Beta")
    rb_1 = timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6
    )
    rb_1.plot(color="steelblue", lw=3, alpha=0.6, ax=ax, **kwargs)
    rb_2 = timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 12
    )
    rb_2.plot(color="grey", lw=3, alpha=0.4, ax=ax, **kwargs)
    ax.axhline(rb_1.mean(), color="steelblue", linestyle="--", lw=3)
    ax.axhline(1.0, color="black", linestyle="--", lw=1)

    ax.set_xlabel("")
    ax.legend(
        ["6-mo", "12-mo", "6-mo Average"],
        loc=legend_loc,
        frameon=True,
        framealpha=0.5,
    )
    return ax


def plot_rolling_volatility(
    returns,
    factor_returns=None,
    rolling_window=APPROX_BDAYS_PER_MONTH * 6,
    legend_loc="best",
    ax=None,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    rolling_vol_ts = timeseries.rolling_volatility(returns, rolling_window)
    rolling_vol_ts.plot(alpha=0.7, lw=3, color="orangered", ax=ax, **kwargs)
    if factor_returns is not None:
        rolling_vol_ts_factor = timeseries.rolling_volatility(
            factor_returns, rolling_window
        )
        rolling_vol_ts_factor.plot(
            alpha=0.7, lw=3, color="grey", ax=ax, **kwargs
        )

    ax.set_title("Rolling volatility (6-month)")
    ax.axhline(rolling_vol_ts.mean(), color="steelblue", linestyle="--", lw=3)

    ax.axhline(0.0, color="black", linestyle="-", lw=2)

    ax.set_ylabel("Volatility")
    ax.set_xlabel("")
    if factor_returns is None:
        ax.legend(
            ["Volatility", "Average volatility"],
            loc=legend_loc,
            frameon=True,
            framealpha=0.5,
        )
    else:
        ax.legend(
            ["Volatility", "Benchmark volatility", "Average volatility"],
            loc=legend_loc,
            frameon=True,
            framealpha=0.5,
        )
    return ax

def plot_rolling_sharpe(
    returns,
    factor_returns=None,
    rolling_window=APPROX_BDAYS_PER_MONTH * 6,
    legend_loc="best",
    ax=None,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    rolling_sharpe_ts = timeseries.rolling_sharpe(returns, rolling_window)
    rolling_sharpe_ts.plot(alpha=0.7, lw=3, color="orangered", ax=ax, **kwargs)

    if factor_returns is not None:
        rolling_sharpe_ts_factor = timeseries.rolling_sharpe(
            factor_returns, rolling_window
        )
        rolling_sharpe_ts_factor.plot(
            alpha=0.7, lw=3, color="grey", ax=ax, **kwargs
        )

    ax.set_title("Rolling Sharpe ratio (6-month)")
    ax.axhline(
        rolling_sharpe_ts.mean(), color="steelblue", linestyle="--", lw=3
    )
    ax.axhline(0.0, color="black", linestyle="-", lw=3)

    ax.set_ylabel("Sharpe ratio")
    ax.set_xlabel("")
    if factor_returns is None:
        ax.legend(
            ["Sharpe", "Average"], loc=legend_loc, frameon=True, framealpha=0.5
        )
    else:
        ax.legend(
            ["Sharpe", "Benchmark Sharpe", "Average"],
            loc=legend_loc,
            frameon=True,
            framealpha=0.5,
        )

    return ax

def plot_return_quantiles(returns, live_start_date=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    is_returns = (
        returns
        if live_start_date is None
        else returns.loc[returns.index < live_start_date]
    )
    is_weekly = ep.aggregate_returns(is_returns, "weekly")
    is_monthly = ep.aggregate_returns(is_returns, "monthly")
    sns.boxplot(
        data=[is_returns, is_weekly, is_monthly],
        palette=["#4c72B0", "#55A868", "#CCB974"],
        ax=ax,
        **kwargs,
    )

    if live_start_date is not None:
        oos_returns = returns.loc[returns.index >= live_start_date]
        oos_weekly = ep.aggregate_returns(oos_returns, "weekly")
        oos_monthly = ep.aggregate_returns(oos_returns, "monthly")

        sns.swarmplot(
            data=[oos_returns, oos_weekly, oos_monthly],
            ax=ax,
            color="red",
            marker="d",
            **kwargs,
        )
        red_dots = matplotlib.lines.Line2D(
            [],
            [],
            color="red",
            marker="d",
            label="Out-of-sample data",
            linestyle="",
        )
        ax.legend(handles=[red_dots], frameon=True, framealpha=0.5)
    ax.set_xticklabels(["Daily", "Weekly", "Monthly"])
    ax.set_title("Return quantiles")

    return ax

def plot_monthly_returns_timeseries(returns, ax=None, **kwargs):
    def cumulate_returns(x):
        return ep.cum_returns(x)[-1]

    if ax is None:
        ax = plt.gca()

    monthly_rets = returns.resample("M").apply(lambda x: cumulate_returns(x))
    monthly_rets = monthly_rets.to_period()

    sns.barplot(x=monthly_rets.index, y=monthly_rets.values, color="steelblue")

    _, labels = plt.xticks()
    plt.setp(labels, rotation=90)

    # only show x-labels on year boundary
    xticks_coord = []
    xticks_label = []
    count = 0
    for i in monthly_rets.index:
        if i.month == 1:
            xticks_label.append(i)
            xticks_coord.append(count)
            # plot yearly boundary line
            ax.axvline(count, color="gray", ls="--", alpha=0.3)

        count += 1

    ax.axhline(0.0, color="darkgray", ls="-")
    ax.set_xticks(xticks_coord)
    ax.set_xticklabels(xticks_label)

    return ax