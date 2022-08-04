import numpy as np
import pickle
from mnist import load_mnist
import sys
import os
sys.path.append(os.pardir)


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    # sample_weight.pkl은 이미 학습되어있는 네트워크의 가중치를 담고 있는 파일
    return network


def sigmoid(x):
    return 1/(1+np.exp(-x))


def softmax(x):
    c = np.max(x)  # overflow 문제를 해결하기 위해 데이터 중 최대값을 뽑아서, 모든 요소에서 빼주기
    exp_a = np.exp(x-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y


def predict(network, X):
    #이미 학습된 네트워크의 가중치를 가져와서 순방향으로 전파하기 
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    A1 = np.dot(X, W1)+b1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, W2)+b2
    Z2 = sigmoid(A2)
    A3 = np.dot(Z2, W3)+b3
    y = softmax(A3)
    # 출력층 노드의 모든 출력값을 담은 numpy array 반환
    return y


x, t = get_data()
network = init_network()

accuracy_cnt = 0
batch_size=100
for i in range(0, len(x), batch_size):
    y_batch = predict(network, x[i:i+batch_size])
    p = np.argmax(y_batch, axis=1)  # 출력층 노드의 모든 출력값을 담은 numpy array인 y에서 가장 큰 값을 갖는 요소의 index를 반환
    accuracy_cnt +=np.sum(p==t[i:i+batch_size])
    # if t[i]==p:# 최대값을 갖는 요소의 index(추측값)와 t (테스트 데이터의 i번째 요소의 진짜 정답 라벨이 동일하면, 답을 맞춘 것임)
    #     accuracy_cnt += 1


print("Accuracy: {}".format(float(accuracy_cnt)/len(x)))
