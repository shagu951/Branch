import scipy as sp

import matplotlib.pyplot as plt

data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")
#print(data[:10])
x = data[:,0]#: full x data , 0 th column
y = data[:,1]
#print(y)
x= x[~sp.isnan(y)] #only valid data
y = y[~sp.isnan(y)]
plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")

inflection = int(3.5*7*24) # calculate the inflection point in hours
xa = x[:inflection] # data before the inflection point
ya = y[:inflection]
xb = x[inflection:] # data after
yb = y[inflection:]
fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))


plt.xticks([w*7*24 for w in range(10)],#list 
 ['week %i'%w for w in range(10)])#names in the axis

#polyfit returns several co-eff if full=True otherwise it will return only fp1

#once we get the optimal co-eff , evaluate  those co-eff in the poly1d 
f1 = sp.poly1d(fa)
f2 = sp.poly1d(fb)
fx = sp.linspace(0,x[inflection], 1000) # generate X-values for plotting
fx1 = sp.linspace(x[inflection],x[-1],1000)
plt.plot(fx, f1(fx), linewidth=4,color='green')
plt.plot(fx1, f2(fx1), linewidth=4,color='yellow')
plt.legend(["d=%i" % f1.order], loc="upper left")
plt.autoscale(tight=True)
plt.grid()
plt.show()
