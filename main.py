from trading import *

NIFTY = YTicker("^NSEI")

print(NIFTY.print_info())

plot = plotting.CandlestickPlot(NIFTY)
plot.indicator(BollingerBands())
plot.indicator(SMAIndicator(window=20))
plot.indicator(SMAIndicator(window=15))
plot.show(period=periods.ytd)

