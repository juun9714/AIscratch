import numpy as np
import matplotlib.pylab as plt


def step(x):
    y=x>0 # 요소별로 조건에 부합하는지 확인해서 Bool 요소를 갖는 배열을 반환
    return y.astype(np.int32) # np 배열의 요소를 int로 변환해주는 함수


def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.arange(-5.0,5.0,0.1) # -5.0부터 0.1씩 키워가며 5.0 전까지의 np 배열을 만듦

plt.plot(x,sigmoid(x))
plt.ylim(-0.1,1.1)
plt.show()



