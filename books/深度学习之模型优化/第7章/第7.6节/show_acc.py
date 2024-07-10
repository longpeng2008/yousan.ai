#coding:utf8
import numpy as np
import matplotlib.pyplot as plt 

######-----测试train------#####
simple_12_3s = np.loadtxt('simple_12_3/simple_12_3_acc.txt')
simple_32_5s = np.loadtxt('simple_32_5/simple_32_5_acc.txt')
kd_10s = np.loadtxt('kd_10/kd_10_acc.txt')

plt.figure()
plt.plot(range(0,99),simple_12_3s,'r-')
plt.plot(range(0,99),simple_32_5s,'g--')
plt.plot(range(0,99),kd_10s,'b-*')
plt.xlabel('epoch')
plt.ylabel('accs')
plt.legend(['simple_12_3','simple_32_5','kd_T=10'])

plt.savefig('acc_compare.png')
plt.show()
