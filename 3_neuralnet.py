# y = h(b + w1*x1 + w2*x2...)
# h() : 활성화 함수 (시그모이드, ReLU 등..)

# 시그모이드 함수
# h(x) =  1 / 1 + exp(-x)

import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)

y1 = step_function(x)
y2 = sigmoid(x)
y3 = relu(x)

plt.plot(x, y1, label='Step')
plt.plot(x, y2, linestyle=':', label = 'Sigmoid')
plt.plot(x, y3, linestyle='--', label = 'ReLU')
plt.ylim(-0.1, 1.1) # y축의 범위 지정
plt.show()


# 간단한 신경망 예제

# I    1    2   O
# x1---@ -+ @---y1
#   \ /    / \ /
#    *-@--*   *
#   / \    \ / \
# x2---@ -+ @---y2

# ==== Layer 1 ==== #
X = np.array([1.0, 5.0])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print('A1:', A1)
print('Z1:', Z1, end='\n\n')

# ==== Layer 2 ==== #
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([[0.1, 0.2]])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print('A2:', A2)
print('Z2:', Z2, end='\n\n')

# Output Layer
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print('A3:', A2)
print('Y:', Y, end='\n\n')


# 항등함수, Softmax
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

softmax_result = softmax(np.array([0.3, 2.9, 4.0]))
print(softmax_result)
print(np.sum(softmax_result))


# MNIST
# import sys, os
# sys.path.append (os.pardir)
# from dataset.mnist import load_mnist

# # 처음 한 번은 몇 분 정도 걸립니다.
# (x_train, t_train), (x_test, t_test) = \
#     load_mnist(flatten=True, normalize=False)

# print(x_train.shape) # (60000, 784)
# print(t_train.shape) # (60000,)
# print(x_test.shape) # (10000, 784)
# print(t_test.shape) # (10000,)
