import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

delta_t = 0.25
market = pd.read_csv("C:/Users/manta/OneDrive/ドキュメント/IEEE/data/spot_summary_2022.csv", encoding='shift_jis')
bidding = pd.read_csv("C:/Users/manta/OneDrive/ドキュメント/IEEE_TEMPR/action/episode_bidding_20241208_1319.csv", encoding='shift_jis')
last_bidding = bidding.iloc[1,:]
def extract_market_prices(market_df, start_row, end_row):
    """6列目のデータを抽出し、エピソードごとに使用"""
    market_prices = market_df.iloc[start_row:end_row, 5].astype(float).dropna().values
    prices_per_episode = []
    for i in range(0, len(market_prices) - 1, 2):
        episode_prices = market_prices[i:i+2]
        prices_per_episode.append(episode_prices)
    return np.array(prices_per_episode)

market_prices = extract_market_prices(market, 8690, 8690+24*2*7)
market_prices_mean = np.mean(market_prices, axis=1)

solar_radiation_all = np.load("C:/Users/manta/OneDrive/ドキュメント/IEEE/data/sample_data_pv.npy") 
solar_radiation = solar_radiation_all[4345:4345 + 24*7]

plt.figure(figsize=(10, 3))

plt.plot(market_prices_mean)
plt.xlim(0,170)
plt.ylim(-10,50)

plt.xlabel('Time (h)',fontsize = 16)
plt.ylabel('Price (¥/kWh)',fontsize = 16)
plt.savefig('C:/Users/manta/OneDrive/ドキュメント/IEEE/plot/market_price.pdf',bbox_inches = 'tight')
plt.show()

plt.figure(figsize=(10, 3))
plt.plot(solar_radiation)
plt.plot(last_bidding)
plt.xlim(0,170)
plt.ylim(-0.1,0.7)

plt.xlabel('Time (h)',fontsize = 16)
plt.ylabel('Power (MW)',fontsize = 16)
plt.savefig('C:/Users/manta/OneDrive/ドキュメント/IEEE/plot/solar_radiation.pdf',bbox_inches = 'tight')
plt.show()