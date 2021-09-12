from infertrade.data.simulate_data import simulated_market_data_4_years_gen

for i in range(5):
    df = simulated_market_data_4_years_gen()
    df.to_csv('./dummy_data/dummy_data_{}.csv'.format(i+1))