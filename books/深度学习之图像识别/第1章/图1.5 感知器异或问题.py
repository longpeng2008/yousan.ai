#coding:utf8
import numpy as np
import matplotlib.pyplot as plt
n = 0           #迭代次数
lr = 0.11        #学习速率
#输入数据分别:偏置值,x1,x2,x1^2,x1*x2,x2^2
X = np.array([[1,0,0,0,0,0],
            [1,0,1,0,0,1],
            [1,1,0,1,0,0],
            [1,1,1,1,1,1]])
#标签
Y = np.array([-1,1,1,-1])
# 权重初始化，取值范围-1到1
W = (np.random.random(X.shape[1])-0.5)*2
print('初始化权值：',W)
def get_show():
    # 正样本
    x1 = [0, 1]
    y1 = [1, 0]
    # 负样本
    x2 = [0,1]
    y2 = [0,1]
    #生成x刻度
    xdata = np.linspace(-1, 2)
    plt.figure()
    #画出两条分界线
    plt.plot(xdata, get_line(xdata,1), 'r')
    plt.plot(xdata, get_line(xdata,2), 'r')
    plt.plot(x1, y1, 'bo')
    plt.plot(x2, y2, 'yo')
    plt.show()

#获得分界线
def get_line(x,root):
    a = W[5]
    b = W[2] + x*W[4]
    c = W[0] + x*W[1] + x*x*W[3]
    #两条不同的分界线
    if root == 1:
        return (-b+np.sqrt(b*b-4*a*c))/(2*a)
    if root == 2:
        return (-b-np.sqrt(b*b-4*a*c))/(2*a)

#更新权值函数
def get_update():
    global X,Y,W,lr,n
    n += 1
    #新输出：X与W的转置相乘，得到的结果再由阶跃函数处理，得到新输出
    new_output = np.dot(X,W.T)
    #调整权重: 新权重 = 旧权重 + 改变权重
    new_W = W + lr*((Y-new_output.T).dot(X))/int(X.shape[0])
    W = new_W
def main():
    for _ in range(100):
        get_update()
        get_show()
    last_output = np.dot(X,W.T)
    print('最后逼近值：',last_output)
if __name__ == "__main__":
  main()

