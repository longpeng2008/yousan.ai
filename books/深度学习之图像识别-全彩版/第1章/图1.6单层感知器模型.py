# Copyright 2021 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
# Author: longpeng
# Email: longpeng2008to2012@gmail.com

#coding:utf8
import numpy as np
import matplotlib.pyplot as plt

n = 0           #迭代次数
lr = 0.10        #学习速率

# 输入数据，三个维度分别表示偏置，x坐标，y坐标
X = np.array([[1,2,3],
              [1,4,5],
              [1,1,1],
              [1,5,3],
              [1,0,1]])
# 标签
Y = np.array([1,1,-1,1,-1])

# 要学习的模型，f(WX+B)，对于正样本，f(WX+b)>0，对于负样本，f(WX+b)<0

# 权重W初始化，取值范围-1到1
W = (np.random.random(X.shape[1])-0.5)*2

# 可视化函数
def get_show():
    # 正样本
    all_positive_x = [2,4,5]
    all_positive_y = [3,5,3]
    # 负样本
    all_negative_x = [1,0]
    all_negative_y = [1,1]
    
    # 计算分界线斜率与截距，W[0]+W[1]*x+W[2]*y=0
    k = -W[1] / W[2]
    b = -(W[0])/ W[2]

    # 生成x刻度
    xdata = np.linspace(0, 5)
    plt.figure()
    plt.plot(xdata,xdata*k+b,'r')
    plt.plot(all_positive_x, all_positive_y,'bo')
    plt.plot(all_negative_x, all_negative_y, 'yo')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# 更新权值函数
def get_update():
    #定义所有全局变量
    global X,Y,W,lr,n
    n += 1
    #计算符号函数输出
    new_output = np.sign(np.dot(X,W.T))
    #更新权重
    new_W = W + lr*((Y-new_output.T).dot(X))
    W = new_W

def main():
    for _ in range(100):
        get_update()
        new_output = np.sign(np.dot(X, W.T))
        if (new_output == Y.T).all():
            print("迭代次数：", n)
            break
    get_show()

if __name__ == "__main__":
    main()

