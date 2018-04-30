import numpy as np
# import matplotlib.pyplot as plt

# h means hypothesis space

def sigmoid(x):
    return 1/(1+np.exp(-x))

def s_prime(z):
    # this is the derivative function of sigmoid
    return np.multiply(z, 1.0-z) 

def initial_weights(layers, epsilon):
    weights = []
    # the element is matrix
    for i in range(len(layers)-1):
        # generate random value in given ships
        # >>> np.random.rand(3,2)
        #     array([[ 0.14022471,  0.96360618],  #random
        #     [ 0.37601032,  0.25528411],  #random
        #     [ 0.49313049,  0.94909878]]) #random
        w = np.random.rand(layers[i+1],layers[i]+1)
        w = w*2*epsilon - epsilon
        weights.append(w)
    return weights

# def fit(X,Y,w,):
#     # now we have initial inputs, outputs, and a w
#     w_grad = ([np.mat(np.zeros(np.shape(w[i]))) 
#             for i in range(len(w))])
#     # for each weight matrix create a new gradient matrix
#     # for exactly each element in each matrix (a weight) has an w_grad
#     m,n=X.shape
#     h_total = np.zeros((m,1))

#     for i in range(m):
#         # for each input
#         x = X[i]
#         y = Y[0,i]
#         # print("CURRENT X",x)
#         # print("CURRENT Y",y)

#         # forward propagate
#         a = x
#         a_s =[]
#         for j in range(len(w)):
#             # each iteration gets the output of the previouse as 
#             # its own input a
#             a = np.mat(np.append(1,a)).T
#             # append 1 
#             # because we want to put the bias in the weight to optimize
#             # make it a matrix and add a T to get the transpose matrix
#             # https://blog.csdn.net/vincentlipan/article/details/20717163
            
#             a_s.append(a)
#             # store the previous a value
            
#             z = w[j] * a
#             a = sigmoid(z)

#         # final output of current w
#         h_total[i,0] = a

#         # back propagate
#         delta = a - y.T

#         # print(np.shape(w_grad[-1]))
#         # print(np.shape(a_s[-1]))
#         # print(np.shape(delta))
#         # print(np.shape(delta * a_s[-1].T))

#         w_grad[-1] += delta * a_s[-1].T
#         # the gradient of the last matrix is equal to
#         # its previous value add the error*the last a stored in a_s

#         # j is the cost func
#         for j in reversed(range(1,len(w))):
#             delta = np.multiply(w[j].T*delta, s_prime(a_s[j]))
#             w_grad[j-1] += (delta[1:]*a_s[j-1].T)
    
#     w_grad = [w_grad[i]/m for i in range(len(w))]
#     J = (1.0/m) * np.sum(-Y * np.log(h_total) - (np.array([[1]]) -Y )*
#         np.log(1-h_total))
#     return {'w_grad':w_grad, 'J': J, 'h':h_total}

def fit(X, Y, w, predict=False, x=None):
    w_grad = ([np.mat(np.zeros(np.shape(w[i]))) 
              for i in range(len(w))])
    for i in range(len(X)):
        x = x if predict else X[i]
        y = Y[0,i]
        # forward propagate
        a = x
        a_s = []
        for j in range(len(w)):
            a = np.mat(np.append(1, a)).T
            a_s.append(a)
            z = w[j] * a
            a = sigmoid(z)
        if predict: return a
        # backpropagate
        delta = a - y.T
        w_grad[-1] += delta * a_s[-1].T
        for j in reversed(range(1, len(w))):
            delta = np.multiply(w[j].T*delta, s_prime(a_s[j]))
            w_grad[j-1] += (delta[1:] * a_s[j-1].T)
    return [w_grad[i]/len(X) for i in range(len(w))]

def predict(x):
    return fit(X, Y, w, True, x)


X = np.mat([[0,0],[0,1],[1,0],[1,1]])
Y = np.mat([0,1,1,0])

# print("THE INPUT SHAPE IS")
# print(X.shape)

layers = [2,2,1]

epochs = 10000
alpha = 0.75

w = initial_weights(layers,1)
w_s = {}

result = {'J':[],'h':[]} 
# its a dictionary of 2 keys, each correspond to one array

# for i in range(epochs):
#     fit_result = fit(X,Y,w)
#     # return a dictionary of 3 elements

#     w_grad = fit_result.get('w_grad')
#     J = fit_result.get('J')
#     h_current = fit_result.get('h')

#     result['J'].append(J)
#     result['h'].append(h_current)

#     for j in range(len(w)):
#         # for each weight matrix in w
#         # like matrix_input_to_layer1,...
#         w[j] -= alpha * w_grad[j]
#     if i == 0 or i ==(epochs - 1):
#         w_s['w_'+str(i) ]= w_grad[:]

for i in range(epochs):
    w_grad = fit(X, Y, w)
    # print w_grad
    for j in range(len(w)):
        w[j] -= alpha * w_grad[j]

# plt.plot(result.get('J'))
# plt.show()

for i in range(len(X)):
    x = X[i]
    guess = predict(x)
    print x, ":", guess