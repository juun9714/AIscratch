import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def softmax(x):
    c=np.max(x) #overflow 문제를 해결하기 위해 데이터 중 최대값을 뽑아서, 모든 요소에서 빼주기
    exp_a = np.exp(x-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['B1'] = np.array([0.1, 0.2, 0.3])
    network['B2'] = np.array([0.1, 0.2])
    network['B3'] = np.array([0.1, 0.2])
    print(network)
    return network


def forward(network, X):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['B1'], network['B2'], network['B3']
    A1 = np.dot(X, W1)+b1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, W2)+b2
    Z2 = sigmoid(A2)
    A3 = np.dot(Z2, W3)+b3

    return A3


a = np.array([0.3, 2.9, 4.0])
softmax(a)
X = np.array([1.0, 0.5])
net = init_network()
y = forward(net, X)
print(y)
