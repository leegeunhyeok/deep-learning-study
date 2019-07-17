#%%

# 간단한 논리회로를 퍼셉트론으로 나타내기

# y =
# 0 (w1 * x1 + w2 + x2 <= theta)
# 1 (w1 * x1 + w2 + x2 > theta)

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


def OR(x1, x2):
    w1, w2, theta = 0.6, 0.6, 0.5
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


def NAND(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


# def XOR ?

#%%
print(AND(0, 0)) # 0
print(AND(1, 0)) # 0
print(AND(0, 1)) # 0
print(AND(1, 1)) # 1

#%%
print(OR(0, 0)) # 0
print(OR(1, 0)) # 1
print(OR(0, 1)) # 1
print(OR(1, 1)) # 1

#%%
print(NAND(0, 0)) # 1
print(NAND(1, 0)) # 1
print(NAND(0, 1)) # 1
print(NAND(1, 1)) # 0

#%%

# 가중치와 편향 도입하기
# 세타값을 -bias로 치환

# y =
# b + w1 * x1 + w2 + x2 <= 0
# b + w1 * x1 + w2 + x2 > 0
import numpy as np

def AND_(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR_(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# NAND는 AND의 w, b만 다름
def NAND_(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#%%
print(AND_(0, 0)) # 0
print(AND_(1, 0)) # 0
print(AND_(0, 1)) # 0
print(AND_(1, 1)) # 1

print(OR_(0, 0)) # 0
print(OR_(1, 0)) # 1
print(OR_(0, 1)) # 1
print(OR_(1, 1)) # 1

print(NAND_(0, 0)) # 1
print(NAND_(1, 0)) # 1
print(NAND_(0, 1)) # 1
print(NAND_(1, 1)) # 0

#%%

# XOR를 그래프로 나타내면 선형으로 나눌 수 없다.
# 비선형으로 나눠야 함
# 단일 퍼셉트론으론 한계가 있으며 다층 퍼셉트론으로 XOR를 구현할 수 있다.


# x1 +---+ s1 -+
#     \ /       \ 
#      *         *-- y
#     / \       /
# x2 +---+ s2 -+

def XOR (x1, x2):
    s1 = NAND_(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

#%%
print(XOR(0, 0)) # 0
print(XOR(1, 0)) # 1
print(XOR(0, 1)) # 1
print(XOR(1, 1)) # 0