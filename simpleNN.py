'''
three layer neuronal network.
inspired by Siraj Raval
xrickliao@gmail.com
'''

import numpy as np

def nonlin(x, deriv = False):
    if(deriv==True):
        return x*(1-x) #partial derivative of (1/(1+(e^-z)))
    return 1/(1+np.exp(-x))

#Define input data
X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,0],
    [0,1,0],
    [1,1,1]
            ])

y = np.array(
    [
        [0],
        [1],
        [1],
        [1],
        [0],
        [1]
    ])
np.random.seed(1)

#weight
weight0 = 2*np.random.random((3, 4))-1
weight1 = 2*np.random.random((4, 1))-1



#train the network
for j in range(100000):
    # feed forward
    layer0 = X
    layer1 = nonlin(np.dot(layer0, weight0))
    layer2 = nonlin(np.dot(layer1, weight1))
    #calculate the error
    layer2_error = y - layer2
    if(j% 10000) == 0:
       print("Error:"+str(np.mean(np.abs(layer2_error))))
    #Back propagation of errors using the chain rule
    layer2_delta = layer2_error * nonlin(layer2, deriv=True)
    layer1_error = layer2_delta.dot(weight1.T)  # weight1.T is transpose of weight1
    layer1_delta = layer1_error * nonlin(layer1,deriv=True)
    '''
    using the deltas , we can use them 
    to update the weights to reduce the 
    error rate with every iteration.
    This algorithm is called gradient descent
    '''
    weight1 += layer1.T.dot(layer2_delta) *.1  #.1 is our learning rate
    weight0 += layer0.T.dot(layer1_delta) *.1

#if __name__ == '__main__': print(nonlin(15,False))

