import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

tickers=["AAPL","MSFT","AMZN","GOOGL"]
data=yf.download(tickers,start="2018-01-01")["Adj Close"]
returns=np.log(data/data.shift(1)).dropna()

window=60
gaps=[]

for i in range(window,len(returns)):
    R=returns.iloc[i-window:i]
    Sigma=np.cov(R.T)
    eigvals=np.linalg.eigvalsh(Sigma)[::-1]
    gaps.append(np.min(eigvals[:-1]-eigvals[1:]))

plt.plot(gaps)
plt.title("Spectral Gap")
plt.savefig("spectral_gap.png",dpi=300)
