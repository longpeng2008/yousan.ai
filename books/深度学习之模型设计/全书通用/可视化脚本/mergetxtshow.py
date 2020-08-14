import os
import matplotlib.pyplot as plt
import sys
import numpy as np

params = sys.argv
params = params[1:len(params)]

colors = [(0,0,1),(0,1,0),(1,0,0),(0,1,1),(0,0.5,0.5),(0.5,0.5,1),(0.5,0.5,0)]
#colors = [(0,0,0),(0.1,0.1,0.1),(0.2,0.2,0.2),(0.3,0.3,0.3),(0.4,0.4,0.4),(0.5,0.5,0.5),(0.6,0.6,0.6)]
linestyles = ['-','--','-.',':','-','--','-.']

test_iter = 1000
test_interval=100
minx = 0
maxx = 1000
sample_interval = 1

x_values=list(range(0,test_iter+1))
x_values = [x*test_interval for x in x_values]
sampleindex=(np.array(range(0,maxx,sample_interval))).astype(np.int32)
print "sampleindex=",sampleindex
xs = np.array(x_values)[sampleindex]

count = 0
legendstr=''
for param in params:
   print 'load ',param
   y_values=np.loadtxt(param)
   ys = y_values[sampleindex]
   print 'line styles',linestyles[count]
   plt.plot(xs[minx:maxx],ys[minx:maxx],linestyle=linestyles[count],c=colors[count])
      
   count = count + 1
   if count > 0:
      legendstr = param.replace('.txt','')+','
   legendstr = legendstr
print legendstr    

plt.title("accuracy",size=24)
plt.legend(('allconv5','allconv6','allconv7_1','allconv7_2','allconv8_1','allconv8_2','allconv8_3'))
plt.xlabel("iterations",size=14)
plt.ylabel("accuracy",size=14)
plt.show()



