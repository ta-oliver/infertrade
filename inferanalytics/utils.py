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

import warnings
from itertools import cycle
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
from distutils.version import StrictVersion
import empyrical.utils


APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52

MM_DISPLAY_UNIT = 1000000.0

DAILY = "daily"
WEEKLY = "weekly"
MONTHLY = "monthly"
YEARLY = "yearly"

ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR,
}

COLORMAP = "Paired"
COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000080",
    "#808080",
]

pandas_version = StrictVersion(pd.__version__)

pandas_one_point_one_or_less = pandas_version < StrictVersion("1.2")


def one_dec_places(x, pos):
    """
    Adds 1/10th decimal to plot ticks.
    """

    return "%.1f" % x


def two_dec_places(x, pos):
    """
    Adds 1/100th decimal to plot ticks.
    """

    return "%.2f" % x


def percentage(x, pos):
    """
    Adds percentage sign to plot ticks.
    """

    return "%.0f%%" % x

def vectorize(func):
    """
    Decorator so that functions can be written to work on Series but
    may still be called with DataFrames.
    """

    def wrapper(df, *args, **kwargs):
        if df.ndim == 1:
            return func(df, *args, **kwargs)
        elif df.ndim == 2:
            return df.apply(func, *args, **kwargs)

    return wrapper

def standardize_data(x):
    """
    Standardize an array with mean and standard deviation.

    Parameters
    ----------
    x : np.array
        Array to standardize.

    Returns
    -------
    np.array
        Standardized array.
    """

    return (x - np.mean(x)) / np.std(x)


def detect_intraday(positions, transactions, threshold=0.25):
    """
    Attempt to detect an intraday strategy. Get the number of
    positions held at the end of the day, and divide that by the
    number of unique stocks transacted every day. If the average quotient
    is below a threshold, then an intraday strategy is detected.

    Parameters
    ----------
    positions : pd.DataFrame
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.
         - See full explanation in create_full_tear_sheet.

    Returns
    -------
    boolean
        True if an intraday strategy is detected.
    """

    daily_txn = transactions.copy()
    daily_txn.index = daily_txn.index.date
    txn_count = daily_txn.groupby(level=0).symbol.nunique().sum()
    daily_pos = positions.drop("cash", axis=1).replace(0, np.nan)
    return daily_pos.count(axis=1).sum() / txn_count < threshold

def clip_returns_to_benchmark(rets, benchmark_rets):
    """
    Drop entries from rets so that the start and end dates of rets match those
    of benchmark_rets.

    Parameters
    ----------
    rets : pd.Series
        Daily returns of the strategy, noncumulative.
         - See pf.tears.create_full_tear_sheet for more details

    benchmark_rets : pd.Series
        Daily returns of the benchmark, noncumulative.

    Returns
    -------
    clipped_rets : pd.Series
        Daily noncumulative returns with index clipped to match that of
        benchmark returns.
    """

    if (rets.index[0] < benchmark_rets.index[0]) or (
        rets.index[-1] > benchmark_rets.index[-1]
    ):
        clipped_rets = rets[benchmark_rets.index]
    else:
        clipped_rets = rets

    return clipped_rets


def to_utc(df):
    """
    For use in tests; applied UTC timestamp to DataFrame.
    """

    try:
        df.index = df.index.tz_localize("UTC")
    except TypeError:
        df.index = df.index.tz_convert("UTC")

    return df


def to_series(df):
    """
    For use in tests; converts DataFrame's first column to Series.
    """

    return df[df.columns[0]]


# This functions is simply a passthrough to empyrical, but is
# required by the register_returns_func and get_symbol_rets.
default_returns_func = empyrical.utils.default_returns_func

# Settings dict to store functions/values that may
# need to be overridden depending on the users environment
SETTINGS = {"returns_func": default_returns_func}


def register_return_func(func):
    """
    Registers the 'returns_func' that will be called for
    retrieving returns data.

    Parameters
    ----------
    func : function
        A function that returns a pandas Series of asset returns.
        The signature of the function must be as follows

        >>> func(symbol)

        Where symbol is an asset identifier

    Returns
    -------
    None
    """

    SETTINGS["returns_func"] = func


def get_symbol_rets(symbol, start=None, end=None):
    """
    Calls the currently registered 'returns_func'

    Parameters
    ----------
    symbol : object
        An identifier for the asset whose return
        series is desired.
        e.g. ticker symbol or database ID
    start : date, optional
        Earliest date to fetch data for.
        Defaults to earliest date available.
    end : date, optional
        Latest date to fetch data for.
        Defaults to latest date available.

    Returns
    -------
    pandas.Series
        Returned by the current 'returns_func'
    """

    return SETTINGS["returns_func"](symbol, start=start, end=end)


def configure_legend(
    ax, autofmt_xdate=True, change_colors=False, rotation=30, ha="right"
):
    """
    Format legend for perf attribution plots:
    - put legend to the right of plot instead of overlapping with it
    - make legend order match up with graph lines
    - set colors according to colormap
    """
    chartBox = ax.get_position()
    ax.set_position(
        [chartBox.x0, chartBox.y0, chartBox.width * 0.75, chartBox.height]
    )

    # make legend order match graph lines
    handles, labels = ax.get_legend_handles_labels()
    handles_and_labels_sorted = sorted(
        zip(handles, labels), key=lambda x: x[0].get_ydata()[-1], reverse=True
    )

    handles_sorted = [h[0] for h in handles_and_labels_sorted]
    labels_sorted = [h[1] for h in handles_and_labels_sorted]

    if change_colors:
        for handle, color in zip(handles_sorted, cycle(COLORS)):
            handle.set_color(color)

    ax.legend(
        handles=handles_sorted,
        labels=labels_sorted,
        frameon=True,
        framealpha=0.5,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize="small",
    )

    # manually rotate xticklabels instead of using matplotlib's autofmt_xdate
    # because it disables xticklabels for all but the last plot
    if autofmt_xdate:
        for label in ax.get_xticklabels():
            label.set_ha(ha)
            label.set_rotation(rotation)


def sample_colormap(cmap_name, n_samples):
    """
    Sample a colormap from matplotlib
    """
    colors = []
    colormap = cm.cmap_d[cmap_name]
    for i in np.linspace(0, 1, n_samples):
        colors.append(colormap(i))

    return colors
