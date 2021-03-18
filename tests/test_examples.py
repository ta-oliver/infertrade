"""
initial testing

Created by: Joshua Mason
Created date: 11/03/2021
"""

import pandas as pd
from sklearn.pipeline import make_pipeline
from ta.momentum import AwesomeOscillatorIndicator
from ta.trend import AroonIndicator

from infertrade.algos import ta_adaptor
from infertrade.algos import talib_adapter
from infertrade.algos.community import normalised_close
from infertrade.algos.community import scikit_signal_factory
from infertrade.algos.community.positions import constant_allocation_size, scikit_position_factory
from infertrade.base import get_signal_calc
from infertrade.data.simulate_data import simulated_market_data_4_years_gen
from infertrade.utilities.operations import PositionsFromPricePrediction, \
    PricePredictionFromSignalRegression
from infertrade.utilities.operations import PricePredictionFromPositions


def test_run_talib_indicator(test_market_data_4_years):
    """Test implementation of ta-lib technical indicators."""
    adapted_stoch = talib_adapter("STOCH", "slowk")
    get_signal = get_signal_calc(adapted_stoch)
    df = get_signal(test_market_data_4_years)
    assert isinstance(df, pd.DataFrame)

    adapted_stoch = talib_adapter("STOCH", "slowk", fastk_period=7)
    get_signal = get_signal_calc(adapted_stoch)
    df = get_signal(test_market_data_4_years)
    assert isinstance(df, pd.DataFrame)

    params = {"fastk_period": 10}

    adapted_stoch = talib_adapter("STOCH", "slowk", **params)
    get_signal = get_signal_calc(adapted_stoch)
    df = get_signal(test_market_data_4_years)
    assert isinstance(df, pd.DataFrame)

    adapted_stoch = talib_adapter("ATR", "real")
    get_signal = get_signal_calc(adapted_stoch)
    df = get_signal(test_market_data_4_years)
    assert isinstance(df, pd.DataFrame)


def test_run_aroon_indicator(test_market_data_4_years):
    """Test implementation of TA technical indicators."""
    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_up")
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    assert isinstance(df, pd.DataFrame)

    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", window=1)
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    assert isinstance(df, pd.DataFrame)

    params = {"window": 100}

    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", **params)
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    assert isinstance(df, pd.DataFrame)

    adapted_aroon = ta_adaptor(AwesomeOscillatorIndicator, "awesome_oscillator")
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    assert isinstance(df, pd.DataFrame)


def test_transformers():
    """Verify transformers work."""
    pos_from_price = PositionsFromPricePrediction()
    df = simulated_market_data_4_years_gen()
    df["forecast_price_change"] = df["close"] * 0.000_1
    df_with_positions = pos_from_price.fit_transform(df)
    predictions_from_positions = PricePredictionFromPositions()
    df0 = predictions_from_positions.fit_transform(df_with_positions)
    df0 = df0.round()
    df = df_with_positions.round()
    assert list(df["forecast_price_change"]) == list(df0["forecast_price_change"])


def test_regression():
    """Check regression"""
    simulated_market_data = simulated_market_data_4_years_gen()
    simulated_market_data["signal"] = simulated_market_data["close"].shift(-1)
    price_prediction_from_signal = PricePredictionFromSignalRegression()
    out = price_prediction_from_signal.fit_transform(simulated_market_data)
    assert isinstance(out, pd.DataFrame)


def test_pipeline_signal_to_position():
    """Checks we can use a signal in conjunction with a rule to calculate a position."""
    signal_to_positions = make_pipeline(
        scikit_signal_factory(normalised_close, ),
        PricePredictionFromSignalRegression(),
        PositionsFromPricePrediction()
    )
    df = signal_to_positions.fit_transform(simulated_market_data_4_years_gen())
    assert isinstance(df, pd.DataFrame)


def test_readme_example_one():
    """Example of signal generation from time series via simple function"""
    signal_transformer = scikit_signal_factory(normalised_close)
    signal_transformer.fit_transform(simulated_market_data_4_years_gen())


def test_readme_example_one_external():
    """Example of signal generation from time series via simple function"""
    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", window=1)
    signal_transformer = scikit_signal_factory(adapted_aroon)
    signal_transformer.fit_transform(simulated_market_data_4_years_gen())


def test_readme_example_two():
    """Example of position calculation from simple position function"""
    position_transformer = scikit_position_factory(constant_allocation_size)
    position_transformer.fit_transform(simulated_market_data_4_years_gen())


def test_readme_example_three():
    """Get price prediction and positions from a signal transformer"""
    pipeline = make_pipeline(scikit_signal_factory(normalised_close),
                             PricePredictionFromSignalRegression(),
                             PositionsFromPricePrediction()
                             )

    pipeline.fit_transform(simulated_market_data_4_years_gen())


def test_readme_example_four():
    """Get price prediction and positions from an external signal transformer"""
    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", window=1)
    pipeline = make_pipeline(scikit_signal_factory(adapted_aroon),
                             PricePredictionFromSignalRegression(),
                             PositionsFromPricePrediction()
                             )
    pipeline.fit_transform(simulated_market_data_4_years_gen())
