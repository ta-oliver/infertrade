"""
Utility code for operations such as converting positions to price predictions and vice versa.

Copyright 2021 InferStat Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created by: Joshua Mason
Created date: 11/03/2021
"""

from copy import deepcopy
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, Binarizer
from infertrade.utilities.performance import calculate_portfolio_performance_python
from infertrade.PandasEnum import PandasEnum, create_price_column_from_synonym


def pct_chg(x: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Percentage change between the current and a prior element."""
    x = x.astype("float64")

    if isinstance(x, pd.DataFrame):
        pc = x.pct_change().values.reshape(-1, 1)
    else:
        x = np.reshape(x, (-1,))
        x_df = pd.Series(x, name="x")
        pc = x_df.pct_change().values.reshape(-1, 1)

    return pc


def diff_log(x: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Differencing and log transformation between the current and a prior element."""
    x = x.astype("float64")
    dl = np.diff(np.log(x), n=1, prepend=np.nan, axis=0)
    return dl


def lag(x: Union[np.ndarray, pd.Series], shift: int = 1) -> np.ndarray:
    """Lag (shift) series by desired number of periods."""
    x = x.astype("float64")
    lagged_array = np.roll(x, shift=shift, axis=0)
    lagged_array[:shift, :] = np.nan
    return lagged_array


def dl_lag(x: Union[np.ndarray, pd.Series], shift: int = 1) -> np.ndarray:
    """Differencing and log transformation of lagged series."""
    x = x.astype("float64")

    dl_trans = FunctionTransformer(diff_log)
    lag_trans = FunctionTransformer(lag, kw_args={"shift": shift})

    dl_lag_pipe = make_pipeline(dl_trans, lag_trans)

    dll = dl_lag_pipe.fit_transform(x)
    return dll


def zero_one_dl(x: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Returns ones for positive values of "diff-log" series, and zeros for negative values."""
    x = x.astype("float64")

    dl_trans = FunctionTransformer(diff_log)

    zero_one_pipe = make_pipeline(
        dl_trans, SimpleImputer(strategy="constant", fill_value=0.0), Binarizer(threshold=0.0)
    )
    zero_one = zero_one_pipe.fit_transform(x)
    return zero_one


def moving_average(x: Union[np.ndarray, pd.Series], window: int) -> np.ndarray:
    """Calculate moving average of series for desired number of periods (window)."""
    x = np.array(x)
    x = x.astype("float64")

    x_pd = pd.DataFrame(x, columns=["x"])
    ma = x_pd["x"].rolling(window=window).mean()
    ma_np = np.reshape(ma.values, (-1, 1))
    return ma_np


def log_price_minus_log_research(x: Union[np.ndarray, pd.Series], shift: int) -> np.ndarray:
    """Difference of two lagged log series."""
    x = np.array(x)
    x = x.astype("float64")

    if x.shape[1] != 2:
        raise IndexError(f"Number of columns must be 2.")

    lag_trans = FunctionTransformer(lag, kw_args={"shift": shift})

    pmr_pipe = make_pipeline(lag_trans)
    lagged = pmr_pipe.fit_transform(x)

    pmr = np.log(lagged[:, [0]]) - np.log(lagged[:, [1]])
    return pmr


def research_over_price_minus_one(x: Union[np.ndarray, pd.Series], shift: int) -> np.ndarray:
    """Difference of two lagged log series."""
    x = np.array(x)
    x = x.astype("float64")

    if x.shape[1] != 2:
        raise IndexError(f"Number of columns must be 2.")

    lag_trans = FunctionTransformer(lag, kw_args={"shift": shift})

    pmr_pipe = make_pipeline(lag_trans)
    lagged = pmr_pipe.fit_transform(x)

    pmr = lagged[:, [1]] / lagged[:, [0]] - 1
    return pmr


class PricePredictionFromSignalRegression(TransformerMixin, BaseEstimator):

    """Class for creating price predictions from signal values."""

    def __init__(self, market_to_trade: str = None):
        """We create by determining one input column as being the price to target."""
        if not market_to_trade:
            # We default to "price" as the target.
            market_to_trade = PandasEnum.MID.value
        self.market_to_trade = market_to_trade

    def fit(self, X: np.array, y=None):
        self.fitted_features_and_target_ = None
        return self

    def transform(self, X, y=None):
        """We transform a signal input to a price prediction."""
        X_ = deepcopy(X)

        create_price_column_from_synonym(X_)

        regression_period = 120
        forecast_period = min(regression_period, len(X_))
        prediction_indices = self._get_model_prediction_indices(len(X_), regression_period, forecast_period)
        self._fit_features_matrix_target_array(X_)
        historical_signal_levels, historical_price_moves = self._get_features_matrix_target_array(X_)

        for ii_day in range(len(prediction_indices)):
            model_idx = prediction_indices[ii_day]["model_idx"]
            prediction_idx = prediction_indices[ii_day]["prediction_idx"]

            # Fit model
            regression_period_signal = historical_signal_levels[model_idx, :]
            regression_period_price_change = historical_price_moves[model_idx]

            rolling_regression_model = LinearRegression().fit(regression_period_signal, regression_period_price_change)
            # Predictions
            current_research = historical_signal_levels[prediction_idx, :]
            forecast = rolling_regression_model.predict(current_research)

            # Apply the calculated allocation to the dataframe.
            X_.loc[prediction_idx, PandasEnum.FORECAST_PRICE_CHANGE.value] = forecast

        if len(prediction_indices) == 0:
            X_[PandasEnum.FORECAST_PRICE_CHANGE.value] = 0
        else:
            X_[PandasEnum.FORECAST_PRICE_CHANGE.value].shift(-1)
        return X_

    def _get_features_matrix_transformer(self):
        """
        1. Percent change of research series as predictor.
        2. Research series level as predictor.
        """
        percent_change_trans = FunctionTransformer(pct_chg)
        lag_1 = FunctionTransformer(lag, kw_args={"shift": 1})

        lag_pct = make_pipeline(lag_1, percent_change_trans)

        lp_m_lr_l1 = FunctionTransformer(research_over_price_minus_one, kw_args={"shift": 1})

        features = ColumnTransformer(
            [
                ("signal", lag_1, ["signal"]),
                ("signal_changes", lag_pct, ["signal"]),
                ("signal_differences", lp_m_lr_l1, [self.market_to_trade, "signal"]),
            ]
        )
        self.feature_names = ["signal", "signal_changes", "signal_differences"]
        return features

    def _get_features_matrix_target_array(
        self, input_time_series: pd.DataFrame
    ) -> [pd.Series, pd.Series]:  # TODO - argument hints please.
        """Returns the target array features."""
        feat_tar_arr = self.fitted_features_and_target_.transform(input_time_series)
        feat_tar_arr = np.nan_to_num(feat_tar_arr, nan=0.0, posinf=0.0, neginf=0.0)

        features = np.delete(feat_tar_arr, -1, axis=1)
        target = feat_tar_arr[:, -1]

        return features, target

    def _fit_features_matrix_target_array(self, X: pd.DataFrame):
        """Get features matrix and target array. TODO -  more description helpful."""
        features = self._get_features_matrix_transformer()
        target = self._get_target_array_transformer()
        feat_tar = FeatureUnion(transformer_list=[("features", features), ("target", target)])
        self.fitted_features_and_target_ = feat_tar.fit(X)

    def _get_target_array_transformer(self):
        """Use level of price series as target (dependant) variable."""
        percent_change_trans = FunctionTransformer(pct_chg)
        target = ColumnTransformer([("historical_price_moves", percent_change_trans, [self.market_to_trade])])
        self.target_name = ["historical_price_moves"]
        return target

    @staticmethod
    def _get_model_prediction_indices(series_length: int, reg_period: int, forecast_period: int) -> List[dict]:
        """
        Create list of ranges for rolling regression.

        Parameters
        ----------
        series_length - total length of series
        reg_period - regression period
        forecast_period - forecast period

        Returns
        -------
        - model_idx are ranges for model fitting
        - prediction_idx are ranges for forecasting

        Examples
        --------
        {'model_idx': range(0, 50), 'prediction_idx': range(50, 60)}
        {'model_idx': range(10, 60), 'prediction_idx': range(60, 70)}
        {'model_idx': range(20, 70), 'prediction_idx': range(70, 80)}
        {'model_idx': range(30, 80), 'prediction_idx': range(80, 90)}
        {'model_idx': range(40, 90), 'prediction_idx': range(90, 100)}
        """

        indices_for_prediction = []

        for i in range(0, series_length - reg_period, forecast_period):
            # idx for model
            ind_start = i
            ind_end = i + reg_period

            # indices_for_prediction
            ind_pred_start = ind_end
            ind_pred_end = ind_pred_start + forecast_period
            if ind_pred_end > series_length:
                ind_pred_end = series_length

            indices_for_prediction.append(
                {"model_idx": range(ind_start, ind_end), "prediction_idx": range(ind_pred_start, ind_pred_end),}
            )

        return indices_for_prediction


class PositionsFromPricePrediction(TransformerMixin, BaseEstimator):
    """This class calculates the positions to take assuming Kelly Criterion."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = deepcopy(X)
        volatility = 0.1
        kelly_fraction = 1.0
        kelly_recommended_optimum = X[PandasEnum.FORECAST_PRICE_CHANGE.value] / volatility ** 2
        rule_recommended_allocation = kelly_fraction * kelly_recommended_optimum
        X_[PandasEnum.ALLOCATION.value] = rule_recommended_allocation
        return X_


class PricePredictionFromPositions(TransformerMixin, BaseEstimator):
    """
    This converts positions into implicit price predictions based on the Kelly Criterion and an assumed volatility.
    """

    def __init__(self):
        """Trivial creation method."""
        pass

    def fit(self, X, y=None):
        """Not used."""
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """Converts allocations into the forecast one-day price changes."""
        X_ = deepcopy(X)
        volatility = 0.1
        kelly_fraction = 1.0

        kelly_recommended_optimum = X_[PandasEnum.ALLOCATION.value] / kelly_fraction
        X_["PandasEnum.FORECAST_PRICE_CHANGE.value"] = kelly_recommended_optimum * volatility ** 2
        return X_


class ReturnsFromPositions(TransformerMixin, BaseEstimator):
    """This calculate returns from positions."""

    def __init__(self):
        """Trivial creation method."""
        pass

    def fit(self, X, y=None):
        """Not used."""
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """Converts positions into the cumulative portfolio return."""
        X_1 = deepcopy(X)
        X_2 = deepcopy(X)
        X_1[PandasEnum.VALUATION.value] = calculate_portfolio_performance_python(X_2)[PandasEnum.VALUATION.value]
        return X_1
