from trading import *

zomato = Stock("ZOMATO.NS")
bel = Stock("BEL.NS")
irfc = Stock("IRFC.NS")
ntpc = Stock("NTPC.NS")
rpower = Stock("RPOWER.NS")
tatamotors = Stock("TATAMOTORS.NS")
reliance = Stock("RELIANCE.NS")
avanti = Stock("AVANTIFEED.NS")
rec = Stock("RECLTD.NS")
asian = Stock("ASIANPAINT.NS")
lici = Stock("LICI.NS")
bhel = Stock("BHEL.NS")

bhavye = Portfolio(zomato, bel, irfc, ntpc, rpower, bhel)
bhavye.weights = [4081, 4520, 3589, 4398, 1619, 3925]

mohit = Portfolio(tatamotors, reliance, avanti, rec, asian, lici)
mohit.weights = [190.26, 215.42, 121.455, 167.37, 142.59, 128.99]

model = capm.EfficientFrontier(mohit, period="max")
model.plot(10000)
model.plot_position()
# model.compute_desired_returns(Stock("^NSEI"))
model.show()
