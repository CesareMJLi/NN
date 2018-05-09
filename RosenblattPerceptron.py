
import numpy as np
import pylab as pl
# <------------logic OR by perceptron------------>

# bias = 0
# a = 0.5 # learning rate

# x = np.array([[0,1,1],[0,1,0],[0,0,0],[0,0,1]])
# y = np.array([1,1,0,1])

# w = np.array([bias,0,0])

# def sgn(v):
#     if v>0:
#         return 1
#     else:
#         return 0

# def compute_outputs(weight,inputs):
#     return sgn(np.dot(weight.T, inputs))

# def newweights(oldweight, outputs, inputs, a):
#     return oldweight+a*(outputs - compute_outputs(oldweight, inputs))* inputs


# for i,xn in enumerate(x):
#     w = newweights(w, y[i], xn, a)

# for xn in x:
#     print("%d or %d => %d" %(xn[1],xn[2],compute_outputs(w, xn)))

# <------------classification------------>
# points on the 2x+1=y marked as positive, on the 7x+1=y marked as negative
bias = 1
a = 0.3 # learning rate

x = np.array([[1,1,3],[1,2,3],[1,1,8],[1,2,15],[1,3,7],[1,4,29]])
y = np.array([1,1,-1,-1,1,-1])

w = np.array([bias,0,0])

def sgn(v):
    if v>0:
        return 1
    else:
        return -1

def compute_outputs(weight,inputs):
    return sgn(np.dot(weight.T, inputs))

def newweights(oldweight, outputs, inputs, a):
    return oldweight+a*(outputs - compute_outputs(oldweight, inputs))* inputs


for i,xn in enumerate(x):
    w = newweights(w, y[i], xn, a)

# plot

myx = x[:,1]
myy = x[:,2]

pl.subplot(111)

x_max = np.max(myx)+15
x_min = np.min(myx)-5

y_max = np.max(myy)+50
y_min = np.min(myy)-5

pl.xlabel(u"x")
pl.xlim(x_min,x_max)
pl.ylabel(u"y")
pl.ylim(y_min,y_max)

for i in xrange(0, len(y)):
    if y[i]>0:
        pl.plot(myx[i],myy[i],'r*')
    else:
        pl.plot(myx[i],myy[i],'ro')

# test

test = np.array([bias,9,19])
if compute_outputs(w,test)>0:
    pl.plot(test[1],test[2],'b.')
else:
    pl.plot(test[1],test[2],'bx')

test = np.array([bias,9,64])
if compute_outputs(w,test)>0:
    pl.plot(test[1],test[2],'b.')
else:
    pl.plot(test[1],test[2],'bx')

test = np.array([bias,9,16])
if compute_outputs(w,test)>0:
    pl.plot(test[1],test[2],'b.')
else:
    pl.plot(test[1],test[2],'bx')

test = np.array([bias,9,60])
if compute_outputs(w,test)>0:
    pl.plot(test[1],test[2],'b.')
else:
    pl.plot(test[1],test[2],'bx')

testx = np.array(range(0,20))
testy = testx*2+1.68
pl.plot(testx,testy,'g--')

pl.show()