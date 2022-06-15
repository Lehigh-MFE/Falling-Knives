
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ochl
from matplotlib.dates import date2num, YearLocator, DateFormatter
from hmmlearn.hmm import GaussianHMM
import pandas as pd


def readData(path):
    """
    import data
    """
    data = pd.read_csv(path)
    data = data[data["date"] > "2017-01-01"]
    return data


def transFeature(data):
    """
    Obtaining features: 5-day log return spread, 20-day log return spread, 5-day volume growth rate and 20-day volume growth rate
    """
    data[["amount", "close_price"]] = data[["amount", "close_price"]].apply(pd.to_numeric)
    data["a_5"] = np.log(data["amount"]).diff(-5)
    data["a_20"] = np.log(data["amount"]).diff(-20)
    data["r_5"] = np.log(data["close_price"]).diff(-5)
    data["r_20"] = np.log(data["close_price"]).diff(-20)
    data["date2num"] = data["date"].apply(lambda x: date2num(datetime.strptime(x, "%Y-%m-%d")))
    data = data[data["date"] > "2017-06-01"]
    return data


def getHiddenStatus(data):
    """
    Using Gaussian HMM modeling the data and obtain predictions
    """
    cols = ["r_5", "r_20", "a_5", "a_20"]
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000,
        random_state=2010)
    model.fit(data[cols])
    hiddenStatus = model.predict(data[cols])
    return hiddenStatus


def visualize(data, hiddenStates):
    """
    Visualizing the model
    """
    # Create a chart
    fig = plt.figure(figsize=(12, 8), dpi=80)
    ax0 = fig.add_subplot(4, 1, 1)
    drawData(ax0, data)
    for i in range(max(hiddenStates)+1):
        _data = data[hiddenStates == i]
        ax = fig.add_subplot(4, 1, i+2, sharex=ax0, sharey=ax0)
        drawData(ax, _data)
    plt.show()


def drawData(ax, _data):
    """
    Using histogram to draw stock market data
    """
    candlestick_ochl(ax,
        _data[["date2num", "open_price", "close_price", "high_price", "low_price"]].values,
        colorup="r", colordown="g", width=0.5)
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    return ax


def stockHMM(dataPath):
    """
    Program Entry
    """
    data = readData(dataPath)
    data = transFeature(data)
    hiddenStates = getHiddenStatus(data)
    visualize(data, hiddenStates)


if __name__ == "__main__":
    # Mac or Window switch
    if os.name == "nt":
        dataPath = "%s\\data\\stock_sh.txt" % os.path.dirname(os.path.abspath(__file__))
    else:
        dataPath = "%s/data/stock_sh.txt" % os.path.dirname(os.path.abspath(__file__))
    stockHMM(dataPath)
