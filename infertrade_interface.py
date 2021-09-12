from infertrade.PandasEnum import PandasEnum
import time
import pandas as pd
from infertrade.algos.community.allocations import infertrade_export_allocations
from infertrade.algos.community.signals import infertrade_export_signals
from infertrade.utilities.performance import calculate_portfolio_performance_python

def evaluate_cross_prediction(dfs_of_asset_prices: pd.DataFrame, export_as_csv: bool = True):
    """A function to evaluate any predictive relationships between the supplied asset time series, with rankings exported to CSV."""

    predictive_relationship = pd.DataFrame(columns=['price_assets', 'signal_assets', 'relationship_type', 'relationship_explaination', 'annualised_return'])

    count = 0
    limit = 10

    for index, price in enumerate(dfs_of_asset_prices):

        # Select prices at the end of list
        # rest_of_the_asset_price = list_of_dfs_of_asset_prices[index+1:]
        for allocation in infertrade_export_allocations.keys():
            for signal in infertrade_export_signals.keys():
                try:
                    print("\n\n" + str(count))

                    calculated_signal = infertrade_export_signals[signal]["function"](price)
                    calculated_allocation = infertrade_export_allocations[allocation]["function"](calculated_signal)
                    calculated_allocation = calculate_portfolio_performance_python(calculated_allocation)
                    
                    print(calculated_allocation)
                except Exception as e:
                    print("ERROR: {}".format(e))
                
                count += 1

                if count > limit:
                    break

            if count > limit:
                break

        # for price_2 in rest_of_the_asset_price:

        #     # calculate_portfolio_performance_python()
        #     print("Count: {}\n\n\n".format(str(count)))
        #     print(price_2)


        #     if count > limit:
        #         break
        
        if count > limit:
            break

    # predictive_relationship.to_csv("predictive_relationship_{}.csv".format(str(int(time.time()))))
    
    return predictive_relationship

data = [ pd.read_csv('./dummy_data/dummy_data_{}.csv'.format(i+1), index_col=0) for i in range(5) ]

evaluate_cross_prediction(data)