import numpy as np
from common import numerical_gradient, softmax, cross_entropy_error


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

net = SimpleNet()

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

print(np.argmax(p))

t = np.array([0, 0, 1])
print(net.loss(x, t))


def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
