from trading import *

NIFTY = YTicker("^NSEI")

print(NIFTY.print_info())

plot = plotting.CandlestickPlot(NIFTY)
plot.show(period=periods.ytd)

