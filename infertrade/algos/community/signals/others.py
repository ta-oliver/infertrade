from sklearn.preprocessing import FunctionTransformer
import pandas as pd


def normalised_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales the close by the maximum value of the close across the whole price history.

    Note that this signal cannot be determined until the end of the historical period and so is unlikely to be suitable
    as an input feature for a trading strategy.
    """
    df["signal"] = df["close"] / max(df["close"])
    return df


# creates wrapper classes to fit sci-kit learn interface
def scikit_signal_factory(signal_function: callable):
    """A class compatible with Sci-Kit Learn containing the signal function."""
    return FunctionTransformer(signal_function)


github_permalink = "https://github.com/ta-oliver/infertrade/blob/4b094d3d5a6ffef119cc79b68a4e7131b40a2ad7/infertrade/algos/community/signals/others.py"

infertrade_export_other_signals = {
    "normalised_close": {
        "function": normalised_close,
        "parameters": {},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(normalised_close.__code__.co_firstlineno)
        },
    },
}
