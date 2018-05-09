

import numpy as np
import math

bias = 1
learning_rate0 = 0.1
learning_rate = 0

r = 5.0

x = np.array([[1,1,6],[1,2,12],[1,3,9],[1,8,24]])
# the elements in x, the dived is 6 marked as positive, 3 marked as negative
y = np.array([1,1,-1,-1])
w = np.array([bias,0,0])

expect_error = 0.005
maxtrycount = 20
counter = 0

def sgn(v):
    if v>0:
        return 1
    else:
        return -1

def compute_outputs(weight,inputs):
    return sgn(np.dot(weight.T, inputs))

def newweights(oldweight, outputs, inputs, lr):
    error = get_error(oldweight, outputs,inputs)
    lr = learning_rate0/(1+float(counter)/r)
    return (oldweight+lr*error*inputs,error)

def get_error(oldweight, outputs, inputs):
    return outputs - compute_outputs(oldweight, inputs)

while True:
    my_error = 0
    for i,xn in enumerate(x):
        w,e = newweights(w,y[i],xn,learning_rate)
        my_error+=pow(e,2)

    my_error=math.sqrt(my_error)

    counter+=1
    print("The change after %i iteration"%counter)
    print(w)
    print("Error: %f" %my_error)

    if abs(my_error)<expect_error or counter>maxtrycount:
        break


for xn in x:
    print("%d   %d  =>  %d" %(xn[1],xn[2],compute_outputs(w,xn)))


print("TEST")

test = np.array([1,9,27])
print("%d   %d  =>  %d" %(test[1],test[2],compute_outputs(w,test)))

test = np.array([1,11,66])
print("%d   %d  =>  %d" %(test[1],test[2],compute_outputs(w,test)))