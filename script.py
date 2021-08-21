import pandas as pd

if __name__ == "__main__":
    btc_df = pd.read_csv('./examples/BTC.csv')
    lbma_df = pd.read_csv('./examples/LBMA_Gold.csv')
    print(btc_df.head)
    print(lbma_df.head)