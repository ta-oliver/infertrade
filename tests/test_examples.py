"""
initial testing

Created by: Joshua Mason
Created date: 11/03/2021
"""
from sklearn.pipeline import make_pipeline
from ta.momentum import AwesomeOscillatorIndicator

from infertrade.data.simulate_data import simulated_market_data_4_years_gen
from infertrade.algos.community import normalised_close, scikit_signal_factory
from infertrade.utilities.operations import PositionsFromPricePrediction, \
    PricePredictionFromPositions, PricePredictionFromSignalRegression
# from infertrade.algos.community import
from infertrade.algos import ta_adaptor
from infertrade.algos import finmarketpy_adapter
from infertrade.base import get_signal_calc
from ta.trend import AroonIndicator


def test_run_aroon_indicator(test_market_data_4_years):
    """Test implementation of TA technical indicators."""
    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_up")
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    print(df)

    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", window=1)
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    print(df)

    params = {"window": 100}

    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", **params)
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    print(df)

    adapted_aroon = ta_adaptor(AwesomeOscillatorIndicator, "awesome_oscillator")
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)


def test_run_atr_indicator(test_market_data_4_years):
    """Test implementation of finmarketpy technical indicators."""
    adapted_ATR = finmarketpy_adapter("ATR", **{"atr_period": 10})
    get_signal = get_signal_calc(adapted_ATR)
    df = get_signal(test_market_data_4_years)
    print(df)


def test_transformers():
    pos_from_price = PositionsFromPricePrediction()
    df = simulated_market_data_4_years_gen()
    df["forecast_price_change"] = df["close"] * 0.000_1
    df_with_positions = pos_from_price.fit_transform(df)
    predictions_from_positions = PricePredictionFromPositions()
    df0 = predictions_from_positions.fit_transform(df[["position"]])
    df0 = df0.round()
    df = df_with_positions.round()
    assert list(df["forecast_price_change"]) == list(df0["forecast_price_change"])


def test_regression():
    simulated_market_data_4_years_gen["signal"] = simulated_market_data_4_years_gen["close"].shift(-1)
    print(simulated_market_data_4_years_gen)
    price_predictin_from_signal = PricePredictionFromSignalRegression()

    out = price_predictin_from_signal.fit_transform(simulated_market_data_4_years_gen)
    print(out)


def test_pipeline_signal_to_position():
    signal_to_positions = make_pipeline(
        scikit_signal_factory(normalised_close, ),
        PricePredictionFromSignalRegression(),
        PositionsFromPricePrediction()
    )

    df = signal_to_positions.fit_transform(simulated_market_data_4_years_gen)

    print(df)


def test_readme_example_one():
    """Example of signal generation from time series via simple function"""
    from infertrade.algos.community import normalised_close, scikit_signal_factory
    from infertrade.data import simulated_market_data_4_years_gen
    signal_transformer = scikit_signal_factory(normalised_close)
    signal_transformer.fit_transform(simulated_market_data_4_years_gen)


def test_readme_example_one_external():
    """Example of signal generation from time series via simple function"""
    from infertrade.algos.community import scikit_signal_factory
    from infertrade.data import simulated_market_data_4_years_gen
    from infertrade.algos import ta_adaptor
    from ta.trend import AroonIndicator
    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", window=1)
    signal_transformer = scikit_signal_factory(adapted_aroon)
    signal_transformer.fit_transform(simulated_market_data_4_years_gen)


def test_readme_example_two():
    """Example of position calculation from simple position function"""
    from infertrade.algos.community.positions import constant_allocation_size, scikit_position_factory
    position_transformer = scikit_position_factory(constant_allocation_size)
    position_transformer.fit_transform(simulated_market_data_4_years_gen)


def test_readme_example_three():
    """Get price prediction and positions from a signal transformer"""
    from infertrade.algos.community import normalised_close, scikit_signal_factory
    from infertrade.utilities.operations import PositionsFromPricePrediction, \
        PricePredictionFromSignalRegression
    from sklearn.pipeline import make_pipeline

    pipeline = make_pipeline(scikit_signal_factory(normalised_close),
                             PricePredictionFromSignalRegression(),
                             PositionsFromPricePrediction()
                             )

    pipeline.fit_transform(simulated_market_data_4_years_gen)

def test_readme_example_four():
    """Get price prediction and positions from an external signal transformer"""
    from infertrade.algos.community import scikit_signal_factory
    from infertrade.utilities.operations import PositionsFromPricePrediction, \
        PricePredictionFromSignalRegression
    from sklearn.pipeline import make_pipeline
    from infertrade.algos import ta_adaptor
    from ta.trend import AroonIndicator

    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", window=1)


    pipeline = make_pipeline(scikit_signal_factory(adapted_aroon),
                             PricePredictionFromSignalRegression(),
                             PositionsFromPricePrediction()
                             )

    pipeline.fit_transform(simulated_market_data_4_years_gen)
