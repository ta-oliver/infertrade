# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created by: Joshua Mason
# Created date: 11/03/2021
# Copyright 2021 InferStat Ltd

"""
This submodule includes facilities for operations such as converting positions to price predictions and vice versa.
"""


from copy import deepcopy
from typing import List, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from infertrade.utilities.performance import calculate_portfolio_performance_python
from infertrade.PandasEnum import PandasEnum, create_price_column_from_synonym


def pct_chg(x: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Percentage change between the current and a prior element.

    Args:
        x: A numpy.ndarray or pandas.Series object

    Returns:
        A numpy.ndarray with the results

    """
    x = x.astype("float64")

    if isinstance(x, pd.DataFrame):
        pc = x.pct_change().values.reshape(-1, 1)
    else:
        x = np.reshape(x, (-1,))
        x_df = pd.Series(x, name="x")
        pc = x_df.pct_change().values.reshape(-1, 1)

    return pc


def lag(x: Union[np.ndarray, pd.Series], shift: int = 1) -> np.ndarray:
    """Lag (shift) series by desired number of periods.

    Args:
        x: A numpy.ndarray or pandas.Series object
        shift: The number of periods by which to shift the input time series

    Returns:
        A numpy.ndarray with the results

    """
    x = x.astype("float64")
    lagged_array = np.roll(x, shift=shift, axis=0)
    lagged_array[:shift, :] = np.nan
    return lagged_array


def research_over_price_minus_one(x: Union[np.ndarray, pd.Series], shift: int) -> np.ndarray:
    """Difference of two lagged log series.

    Args:
        x: A numpy.ndarray or pandas.Series object with exactly two columns
        shift: The number of periods by which the lag both series

    Returns:
        A numpy.ndarray with the results

    """
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

    """This class creates price predictions from signal values.

    Attributes:
        market_to_trade: The name of the column which contains the historical prices.

    """

    def __init__(self, market_to_trade: str = None):
        """Construction method for class PricePredictionFromPositions.

        Args:
            market_to_trade: The name of the column which contains the historical prices.

        Returns:
            None

        """
        if not market_to_trade:
            # We default to "price" as the target.
            market_to_trade = PandasEnum.MID.value
        self.market_to_trade = market_to_trade

    def fit(self, X: np.array, y=None):
        self.fitted_features_and_target_ = None
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """This method transforms a signal input to a price prediction.

        Args:
            X: A pandas.DataFrame object

        Returns:
            A pandas.DataFrame object

        """
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

    def _get_features_matrix_transformer(self) -> ColumnTransformer:
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

    def _get_features_matrix_target_array(self, input_time_series: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
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

    def _get_target_array_transformer(self) -> ColumnTransformer:
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
                {"model_idx": range(ind_start, ind_end), "prediction_idx": range(ind_pred_start, ind_pred_end)}
            )

        return indices_for_prediction


class PositionsFromPricePrediction(TransformerMixin, BaseEstimator):

    """This class calculates the positions to take assuming Kelly Criterion."""

    def __init__(self):
        """Construction method for class PositionsFromPricePrediction.

        Args:
            None

        Returns:
            None

        """
        pass

    def fit(self, X, y=None):
        """This method is not used."""
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """This method calculates the positions to be taken based on the forecast price, assuming the Kelly Criterion.

        Args:
            X: A pandas.DataFrame object

        Returns:
            A pandas.DataFrame object

        """
        X_ = deepcopy(X)
        volatility = 0.1
        kelly_fraction = 1.0
        kelly_recommended_optimum = X[PandasEnum.FORECAST_PRICE_CHANGE.value] / volatility ** 2
        rule_recommended_allocation = kelly_fraction * kelly_recommended_optimum
        X_[PandasEnum.ALLOCATION.value] = rule_recommended_allocation
        return X_


class PricePredictionFromPositions(TransformerMixin, BaseEstimator):
    """
    This class converts positions into implicit price predictions based on the Kelly Criterion and an assumed
     volatility.
     """

    def __init__(self):
        """Construction method for class PricePredictionFromPositions.

        Args:
            None

        Returns:
            None

        """

        pass

    def fit(self, X, y=None):
        """This method is not used."""
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """This method converts allocations into the forecast one-day price changes.

        Args:
            X: A pandas.DataFrame object

        Returns:
            A pandas.DataFrame object

        """

        X_ = deepcopy(X)
        volatility = 0.1
        kelly_fraction = 1.0

        kelly_recommended_optimum = X_[PandasEnum.ALLOCATION.value] / kelly_fraction
        X_["PandasEnum.FORECAST_PRICE_CHANGE.value"] = kelly_recommended_optimum * volatility ** 2
        return X_


class ReturnsFromPositions(TransformerMixin, BaseEstimator):
    """This class calculates returns from positions."""

    def __init__(self):
        """Construction method for class ReturnsFromPositions.

        Args:
            None

        Returns:
            None

        """
        pass

    def fit(self, X, y=None):
        """This method is not used."""
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """This method converts positions into the cumulative portfolio return.

        Args:
            X: A pandas.DataFrame object

        Returns:
            A pandas.DataFrame object

        """
        X_1 = deepcopy(X)
        X_2 = deepcopy(X)
        X_1[PandasEnum.VALUATION.value] = calculate_portfolio_performance_python(X_2)[PandasEnum.VALUATION.value]
        return X_1


def limit_allocation(
    dataframe: pd.DataFrame, allocation_lower_limit: Union[int, float], allocation_upper_limit: Union[int, float]
) -> pd.DataFrame:
    """
    This function limits the ranges by limiting upper and lower limit of allocated values

    params:
    allocated_dataframe
    allocation_lower_limit: the lower limit for allocation values.
    allocation_upper_limit: the upper limit for allocation values.

    returns:
    allocation limited dataframe
    """
    if allocation_lower_limit > allocation_upper_limit:
        raise ValueError(
            "The lower limit for allocation values should not be greater than the upper limit for" " allocation values."
        )
    dataframe.loc[
        dataframe[PandasEnum.ALLOCATION.value] > allocation_upper_limit, PandasEnum.ALLOCATION.value
    ] = allocation_upper_limit
    dataframe.loc[
        dataframe[PandasEnum.ALLOCATION.value] < allocation_lower_limit, PandasEnum.ALLOCATION.value
    ] = allocation_lower_limit
    return dataframe


def daily_stop_loss(dataframe: pd.DataFrame, loss_limit: float) -> pd.DataFrame:
    """
    This function calculates loss and limit the allocation accordingly.
    It restricts allocation to 0 if loss>loss_limit

    params:
    allocated_dataframe
    loss_limit

    returns:
    dataframe
    """

    prev_alloc = 0
    prev_price = 0
    stop_loss_has_triggered = False
    for index, row in dataframe.iterrows():
        price_change = row.price - prev_price
        loss = -price_change * prev_alloc
        if loss > loss_limit or stop_loss_has_triggered:
            stop_loss_has_triggered = True
            row.allocation = 0
        prev_alloc = row.allocation
        prev_price = row.price
    return dataframe


def restrict_allocation(allocation_function: callable) -> callable:
    """
    This function is intended to be used as a decorator that may apply one or more restrictions to functions that
    calculate allocation values.
    """

    def restricted_function(*args, **kwargs) -> pd.DataFrame:
        """
        An allocation function that also incorporates portfolio restrictions, such as maximum allocation size and
         stop loss limits.
         """

        # Get allocation dataframe from allocation function
        dataframe = allocation_function(*args, **kwargs)

        # Restriction by limiting allocations between range

        # initialize limits
        allocation_lower_limit = 0
        allocation_upper_limit = 1

        if "allocation_lower_limit" in kwargs:
            allocation_lower_limit = kwargs.get("allocation_lower_limit")
            dataframe = limit_allocation(dataframe, allocation_lower_limit, allocation_upper_limit)

        if "allocation_lower_limit" in kwargs:
            allocation_upper_limit = kwargs.get("allocation_upper_limit")
            dataframe = limit_allocation(dataframe, allocation_lower_limit, allocation_upper_limit)

        # Restriction by daily stop loss
        loss_limit = None
        if "loss_limit" in kwargs:
            loss_limit = kwargs.get("loss_limit")
            dataframe = daily_stop_loss(dataframe, loss_limit)

        return dataframe

    return restricted_function


def add_two_possibly_zero_length_arrays(regression_period_signal_error_end: np.array,
                                        regression_period_signal_error_start: np.array) -> np.array:
    """Adds two arrays which may be zero length."""
    some_start_data = len(regression_period_signal_error_start) > 0
    some_end_data = len(regression_period_signal_error_end) > 0
    if some_start_data and some_end_data:
        regression_period_signal_error = regression_period_signal_error_start + regression_period_signal_error_end
    elif some_start_data:
        regression_period_signal_error = regression_period_signal_error_start
    elif some_end_data:
        regression_period_signal_error = regression_period_signal_error_end
    else:
        raise IndexError("No error data was provided.")
    return regression_period_signal_error


def calculate_regression_with_kelly_optimum(
    df: pd.DataFrame,
    feature_matrix: pd.Series,
    last_feature_row: np.ndarray,
    target_array: pd.Series,
    regression_period: int,
    forecast_period: int,
    kelly_fraction: float = 1.0,
    out_of_sample_error: bool = False,
) -> pd.DataFrame:
    """
    Calculate the optimum portfolio allocation using a forecast of price change based on regression versus historic
    price changes.
    """
    dataframe = df.copy()
    prediction_indices = PricePredictionFromSignalRegression._get_model_prediction_indices(
        series_length=len(feature_matrix), reg_period=regression_period, forecast_period=forecast_period
    )

    bad_inputs = False
    if len(prediction_indices) > 0:
        for ii_day in range(len(prediction_indices)):
            model_idx = prediction_indices[ii_day]["model_idx"]
            prediction_idx = prediction_indices[ii_day]["prediction_idx"]
            regression_period_signal = feature_matrix[model_idx, :]
            regression_period_price_change = target_array[model_idx]
            regression_period_signal_fit = regression_period_signal
            regression_period_signal_error = regression_period_signal
            regression_period_price_change_fit = regression_period_price_change
            regression_period_price_change_error = regression_period_price_change

            if out_of_sample_error:
                # In this mode we calculate the error out of sample rather than using the calibration error.
                start_error_pct = 0.0
                end_error_pct = 0.25  # unused as fits to remaining values in series
                fit_pct = 1.0 - start_error_pct - end_error_pct
                if fit_pct <= 0.0:
                    raise IndexError("No data selected for calibration.")
                elif fit_pct <= 0.1:
                    print("WARNING - very low fit percentage for calibration.")
                elif fit_pct >= 1.0:
                    raise IndexError("No data selected for error calculation.")
                elif fit_pct >= 0.99:
                    print("WARNING - very little data provided for error calculation.")

                (
                    regression_period_signal_error_start,
                    regression_period_signal_fit,
                    regression_period_signal_error_end,
                ) = np.split(
                    regression_period_signal,
                    [
                        int(len(regression_period_signal) * start_error_pct),
                        int(len(regression_period_signal) * (fit_pct + start_error_pct)),
                    ],
                )
                regression_period_signal_fit = regression_period_signal_fit

                regression_period_signal_error = add_two_possibly_zero_length_arrays(
                    regression_period_signal_error_end, regression_period_signal_error_start
                )

                (
                    regression_period_price_change_error_start,
                    regression_period_price_change_fit,
                    regression_period_price_change_error_end,
                ) = np.split(
                    regression_period_price_change,
                    [
                        int(len(regression_period_price_change) * start_error_pct),
                        int(len(regression_period_price_change) * (start_error_pct + fit_pct)),
                    ],
                )
                regression_period_price_change_fit = regression_period_price_change_fit
                regression_period_price_change_error = add_two_possibly_zero_length_arrays(
                    regression_period_price_change_error_start, regression_period_price_change_error_end
                )

            try:
                # We check if either input has zero changes - if so there is no regression relationship.
                std_price = np.std(regression_period_price_change_fit)
                std_signal = np.std(regression_period_signal_fit)

                if not std_price > 0.0:
                    print("WARNING - price had no variation: ", std_price)
                elif not std_signal > 0.0:
                    print(
                        "WARNING - signal had no variation. Usually this means the lookback period was too short"
                        " for the data sample: ",
                        std_signal,
                    )
                    bad_inputs = True
                else:
                    bad_inputs = False

                if bad_inputs:
                    # Assuming no bad inputs we calculate the recommended allocation
                    rule_recommended_allocation = 0.0
                    volatility = 1.0
                else:
                    rolling_regression_model = LinearRegression().fit(
                        regression_period_signal_fit, regression_period_price_change_fit
                    )

                    # Calculate model error
                    predictions = rolling_regression_model.predict(regression_period_signal_error)
                    forecast_horizon_model_error = np.sqrt(
                        mean_squared_error(regression_period_price_change_error, predictions)
                    )

                    # Predictions
                    forecast_distance = 1
                    current_research = feature_matrix[prediction_idx, :]
                    forecast_price_change = rolling_regression_model.predict(current_research)

                    # Calculate drift and volatility
                    volatility = ((1 + forecast_horizon_model_error) * (forecast_distance ** -0.5)) - 1

                    # Kelly recommended optimum
                    if volatility < 0:
                        raise ZeroDivisionError("Volatility needs to be positive value.")
                    if volatility == 0:
                        volatility = 0.01

                    kelly_recommended_optimum = forecast_price_change / volatility ** 2
                    rule_recommended_allocation = kelly_fraction * kelly_recommended_optimum

            except IndentationError:  # KeyError:
                # QUESTION - not clear why there is KeyError catch here? Changing to IndentationError to see cases.
                # np.zeros(len(prediction_idx))
                rule_recommended_allocation = 0.0

            # Apply the calculated allocation to the dataframe.
            dataframe.loc[prediction_idx, PandasEnum.ALLOCATION.value] = rule_recommended_allocation.reshape(
                -1,
            )

        # Shift position series  (QUESTION - does not appear to shift?)
        dataframe[PandasEnum.ALLOCATION.value] = dataframe[PandasEnum.ALLOCATION.value].shift(-1)

        # Calculate price forecast for last research value
        if std_price > 0.0 and std_signal > 0.0:
            # last_research = [[dataframe[PandasEnum.SIGNAL.value].iloc[-1]]]
            last_research = last_feature_row
            last_forecast_price = rolling_regression_model.predict(last_research)[0]
            value_to_update = kelly_fraction * (last_forecast_price / volatility ** 2)
        else:
            value_to_update = 0.0
        dataframe.iloc[-1, dataframe.columns.get_loc(PandasEnum.ALLOCATION.value)] = value_to_update

    else:
        # If length of prediction indices is zero we set all positions to zero.
        dataframe[PandasEnum.ALLOCATION.value] = 0.0

    return dataframe


def scikit_allocation_factory(allocation_function: callable) -> FunctionTransformer:
    """
    This function creates a SciKit Learn compatible Transformer embedding the position calculation.

    Args:
        allocation_function: A function to be turned into a sklearn.preprocessing.FunctionTransformer

    Returns:
        A sklearn.preprocessing.FunctionTransformer
    """
    return FunctionTransformer(allocation_function)
