#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

#%%
z1 = np.array([[1, 2], [3, 4]])
z2 = np.array([[1, 4], [9, 16]])

#%%
z1 + z2

# array([[ 2,  6],
#        [12, 20]])


#%%
z1 * 10

# array([[10, 20],
#       [30, 40]])

#%%
z1[z1 >= 2]

#array([2, 3, 4])

#%%
x1 = np.arange(0, 6, 0.1) # 0에서 6까지 0.1 간격으로 생성
y1 = np.sin(x1)

#%%
plt.plot(x1, y1)
plt.show()

#%%
x_ = np.arange(0, 6, 0.1) # 0에서 6까지 0.1 간격으로 생성
y1 = np.sin(x_)
y2 = np.cos(x_)

# 그래프 그리기
plt.plot(x_, y1, label ="sin")
plt.plot(x_, y2, linestyle=":", label = "cos") # cos 함수는 점선으로 그리기
plt.xlabel("x") # x축 이름
plt.ylabel("y") # y축 이름
plt.title('sin & cos') # 제목
plt.legend()
plt.show()
