import httpx

from infertrade.data.simulate_data import simulated_market_data_4_years_gen
from infertrade.algos.community.allocations import change_regression
from infertrade.utilities.performance import calculate_portfolio_performance_python


if __name__ == "__main__":

    trading_strategy_df = change_regression(simulated_market_data_4_years_gen())
    trading_strategy_returns_df = calculate_portfolio_performance_python(trading_strategy_df)

    json_data = {
    "service": "privateapi",
    "endpoint": "/",
    "session_id": "session_id",
    "payload": {
        "library": "reducerlib",
        "api_method": "algo_calculate",
        "kwargs": {
            "algorithms": [
                {
                    "name": "SharpeRatio"
                },
                {
                    "name": "PriceBasicStatistics"
                }
            ],
            "inputs": [
                {
                    "time_series": [
                        {
                            "portfolio_return": trading_strategy_returns_df['portfolio_return'].to_list(),
                            "allocation": trading_strategy_returns_df['allocation'].to_list(),
                            "price_1": trading_strategy_returns_df['price'].to_list(),
                            "research_1": trading_strategy_returns_df['research'].to_list()
                        }
                    ]
                }
            ]
        }
    }
}

    response = httpx.post(
        'https://prod.api.infertrade.com/',
        headers=
        {
            'content-type': 'application/json',
            'x-api-key': 'your-api-key-here'
        },
        json=json_data
        )

    print(response.content)