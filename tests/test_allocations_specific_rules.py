#
# Copyright 2021 InferStat Ltd
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
#
# Created by: Thomas Oliver
# Created date: 08/09/2021

"""
Unit tests that apply to specific allocation strategies.

Tests universal to all rules should go into test_allocations.py
"""

# Standard Python packages
import numpy as np
import pandas as pd
import pytest

# InferStat packages
from infertrade.PandasEnum import PandasEnum
from infertrade.algos.community import allocations, signals as signals
from infertrade.data.simulate_data import simulated_market_data_4_years_gen
from tests.utilities.independent_rule_implementations import (
    simple_moving_average,
    weighted_moving_average,
    exponentially_weighted_moving_average,
    moving_average_convergence_divergence,
    relative_strength_index,
    stochastic_relative_strength_index,
    bollinger_band,
    detrended_price_oscillator,
    percentage_series_oscillator,
    triple_exponential_average,
    true_strength_index,
    schaff_trend_cycle,
    kama_indicator,
    aroon,
    rate_of_change,
    vortex_indicator,
)


# Variables for tests.
np.random.seed(1)
num_simulated_market_data = 10
dataframes = [simulated_market_data_4_years_gen() for i in range(num_simulated_market_data)]
max_investment = 0.2


@pytest.mark.parametrize("df", dataframes)
def test_sma_strategy(df):
    """Checks SMA strategy calculates correctly."""
    window = 50
    df_with_allocations = allocations.SMA_strategy(df, window, max_investment)
    df_with_signals = simple_moving_average(df, window)

    price_above_signal = df_with_signals["close"] > df_with_signals["signal"]
    price_below_signal = df_with_signals["close"] <= df_with_signals["signal"]

    df_with_signals.loc[price_above_signal, "allocation"] = max_investment
    df_with_signals.loc[price_below_signal, "allocation"] = -max_investment
    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_wma_strategy(df):
    """Checks WMA strategy calculates correctly."""
    window = 50
    df_with_allocations = allocations.WMA_strategy(df, window, max_investment)
    df_with_signals = weighted_moving_average(df, window)

    price_above_signal = df_with_signals["close"] > df_with_signals["signal"]
    price_below_signal = df_with_signals["close"] <= df_with_signals["signal"]

    df_with_signals.loc[price_above_signal, "allocation"] = max_investment
    df_with_signals.loc[price_below_signal, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_ema_strategy(df):
    """Checks EMA strategy calculates correctly."""
    window = 50
    df_with_allocations = allocations.EMA_strategy(df, window, max_investment)
    df_with_signals = exponentially_weighted_moving_average(df, window)

    price_above_signal = df_with_signals["close"] > df_with_signals["signal"]
    price_below_signal = df_with_signals["close"] <= df_with_signals["signal"]

    df_with_signals.loc[price_above_signal, "allocation"] = max_investment
    df_with_signals.loc[price_below_signal, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_macd_strategy(df):
    """Checks MACD strategy calculates correctly."""
    df_with_allocations = allocations.MACD_strategy(df, 12, 26, 9, max_investment)
    df_with_signals = moving_average_convergence_divergence(df, 12, 26, 9)

    above_zero_line = df_with_signals["signal"] > 0
    below_zero_line = df_with_signals["signal"] <= 0

    df_with_signals.loc[above_zero_line, "allocation"] = max_investment
    df_with_signals.loc[below_zero_line, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_rsi_strategy(df):
    """Checks RSI strategy calculates correctly."""
    df_with_allocations = allocations.RSI_strategy(df, 14, max_investment)
    df_with_signals = relative_strength_index(df, 14)

    over_valued = df_with_signals["signal"] >= 70
    under_valued = df_with_signals["signal"] <= 30
    hold = df_with_signals["signal"].between(30, 70)

    df_with_signals.loc[over_valued, "allocation"] = -max_investment
    df_with_signals.loc[under_valued, "allocation"] = max_investment
    df_with_signals.loc[hold, "allocation"] = 0.0

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_stochastic_rsi_strategy(df):
    """Checks stochastic RSI strategy calculates correctly."""
    df_with_allocations = allocations.stochastic_RSI_strategy(df, 14, max_investment)
    df_with_signals = stochastic_relative_strength_index(df, 14)

    over_valued = df_with_signals["signal"] >= 0.8
    under_valued = df_with_signals["signal"] <= 0.2
    hold = df_with_signals["signal"].between(0.2, 0.8)

    df_with_signals.loc[over_valued, "allocation"] = -max_investment
    df_with_signals.loc[under_valued, "allocation"] = max_investment
    df_with_signals.loc[hold, "allocation"] = 0.0

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_bollinger_band_strategy(df):
    """Checks bollinger band strategy calculates correctly."""
    # Window_dev is kept lower to make sure prices breaks the band
    window = 20
    window_dev = 2
    df_with_allocations = allocations.bollinger_band_strategy(df, window, window_dev, max_investment)
    df_with_signals = bollinger_band(df, window, window_dev)
    short_position = False
    long_position = False

    for index, row in df_with_signals.iterrows():
        # check if price breaks the bollinger bands
        if row["typical_price"] >= row["BOLU"]:
            short_position = True

        if row["typical_price"] <= row["BOLD"]:
            long_position = True

        # check if position needs to be closed
        if short_position == True and row["typical_price"] <= row["BOLA"]:
            short_position = False

        if long_position == True and row["typical_price"] >= row["BOLA"]:
            long_position = False

        assert not (short_position == True and long_position == True)

        # allocation rules
        if short_position:
            df_with_signals.loc[index, "allocation"] = max_investment

        elif long_position:
            df_with_signals.loc[index, "allocation"] = -max_investment

        else:
            df_with_signals.loc[index, "allocation"] = 0.0

    assert pd.Series.equals(df_with_allocations["allocation"], df_with_signals["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_dpo_strategy(df):
    """Checks DPO strategy calculates correctly."""
    df_with_allocations = allocations.DPO_strategy(df, 20, max_investment)
    df_with_signals = detrended_price_oscillator(df, 20)

    above_zero = df_with_signals["signal"] > 0
    below_zero = df_with_signals["signal"] <= 0

    df_with_signals.loc[above_zero, "allocation"] = max_investment
    df_with_signals.loc[below_zero, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_ppo_strategy(df):
    """Checks PPO strategy calculates correctly."""
    series_name = "close"
    df_with_allocations = allocations.PPO_strategy(df, 26, 12, 9, max_investment)
    df_with_signals = percentage_series_oscillator(df, 26, 12, 9, series_name)

    above_zero = df_with_signals["signal"] > 0
    below_zero = df_with_signals["signal"] < 0

    df_with_signals.loc[above_zero, "allocation"] = max_investment
    df_with_signals.loc[below_zero, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_pvo_strategy(df):
    """Checks PVO strategy calculates correctly."""
    series_name = "volume"
    df_with_allocations = allocations.PVO_strategy(df, 26, 12, 9, max_investment)
    df_with_signals = percentage_series_oscillator(df, 26, 12, 9, series_name)

    above_zero = df_with_signals["signal"] > 0
    below_zero = df_with_signals["signal"] <= 0

    df_with_signals.loc[above_zero, "allocation"] = max_investment
    df_with_signals.loc[below_zero, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_trix_strategy(df):
    """Checks TRIX strategy calculates correctly."""
    df_with_allocations = allocations.TRIX_strategy(df, 14, max_investment)
    df_with_signals = triple_exponential_average(df, 14)

    above_zero = df_with_signals["signal"] > 0
    below_zero = df_with_signals["signal"] <= 0

    df_with_signals.loc[above_zero, "allocation"] = max_investment
    df_with_signals.loc[below_zero, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_tsi_strategy(df):
    """Checks TSI strategy calculates correctly."""
    df_with_allocations = allocations.TSI_strategy(df, 25, 13, 13, max_investment=max_investment)
    df_with_signals = true_strength_index(df, 25, 13, 13)

    above_signal = df_with_signals["TSI"] > df_with_signals["signal"]
    below_signal = df_with_signals["TSI"] <= df_with_signals["signal"]

    df_with_signals.loc[above_signal, "allocation"] = max_investment
    df_with_signals.loc[below_signal, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_stc_strategy(df):
    """Checks STC strategy calculates correctly."""
    df_with_allocations = allocations.STC_strategy(df, 50, 23, 10, 3, 3, max_investment=max_investment)
    df_with_signal = schaff_trend_cycle(df)
    oversold = df_with_signal["signal"] <= 25
    overbought = df_with_signal["signal"] >= 75
    hold = df_with_signal["signal"].between(25, 75)

    df_with_signal.loc[oversold, "allocation"] = max_investment
    df_with_signal.loc[overbought, "allocation"] = -max_investment
    df_with_signal.loc[hold, "allocation"] = 0

    assert pd.Series.equals(df_with_signal["allocation"], df_with_allocations["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_kama_strategy(df):
    """Checks KAMA strategy calculates correctly."""
    df_with_signals = kama_indicator(df, 10, 2, 30)
    df_with_allocations = allocations.KAMA_strategy(df, 10, 2, 30, max_investment)

    downtrend = df_with_signals["signal"] <= df_with_signals["close"]
    uptrend = df_with_signals["signal"] > df_with_signals["close"]

    df_with_signals.loc[uptrend, "allocation"] = max_investment
    df_with_signals.loc[downtrend, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_allocations["allocation"], df_with_signals["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_aroon_strategy(df):
    """Checks the Aroon Indicator strategy calculates correctly."""
    df_with_signals = aroon(df, window=25)
    df_with_allocations = allocations.aroon_strategy(df, 25, max_investment)
    df_with_signals_ta = signals.aroon(df, 25)

    bullish = df_with_signals["aroon_up"] >= df_with_signals["aroon_down"]
    bearish = df_with_signals["aroon_down"] < df_with_signals["aroon_up"]

    df_with_signals.loc[bullish, PandasEnum.ALLOCATION.value] = max_investment
    df_with_signals.loc[bearish, PandasEnum.ALLOCATION.value] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_roc_strategy(df):
    """Checks the Rate of Change strategy calculates correctly."""
    df_with_signals = rate_of_change(df, window=25)
    df_with_allocations = allocations.ROC_strategy(df, 25, max_investment)

    bullish = df_with_signals["signal"] >= 0
    bearish = df_with_signals["signal"] < 0

    df_with_signals.loc[bearish, "allocation"] = -max_investment
    df_with_signals.loc[bullish, "allocation"] = max_investment

    assert pd.Series.equals(df_with_allocations["allocation"], df_with_signals["allocation"])


@pytest.mark.parametrize("df", dataframes)
def test_vortex_strategy(df):
    """Checks Vortex strategy calculates correctly."""
    df_with_signals = vortex_indicator(df, window=25)
    df_with_allocations = allocations.vortex_strategy(df, 25, max_investment)

    bullish = df_with_signals["signal"] >= 0
    bearish = df_with_signals["signal"] < 0

    df_with_signals.loc[bearish, "allocation"] = -max_investment
    df_with_signals.loc[bullish, "allocation"] = max_investment

    assert pd.Series.equals(df_with_allocations["allocation"], df_with_signals["allocation"])
