#coding:utf8
import numpy as np
import matplotlib.pyplot as plt 

######-----测试train------#####
iter_train_interval = 100
statistic_train_interval = 1
start_plot_idx = 0

train_acc_file = open('trainacc.refine', 'r')
train_acc_total = train_acc_file.readlines()
train_acc_num = len(train_acc_total)
train_acc_res = np.zeros(train_acc_num)
train_acc_idx = np.arange(train_acc_num) * iter_train_interval

for idx in range(train_acc_num) :
    train_acc_str = train_acc_total[idx]
    str_start_idx = train_acc_str.find('=')+1
    str_end_idx = len(train_acc_str) - 1
    train_acc_res[idx] = float(train_acc_str[str_start_idx:str_end_idx])
    
statistic_len = (train_acc_num + statistic_train_interval - 1)/statistic_train_interval
statistic_idx = np.arange(statistic_len) * iter_train_interval * statistic_train_interval
statistic_res_mean = np.zeros(statistic_len)
statistic_res_var = np.zeros(statistic_len)

for idx in range(statistic_len) :
    train_acc_start_idx = idx*statistic_train_interval
    train_acc_end_idx = min(train_acc_start_idx + statistic_train_interval, train_acc_num)
    train_acc_part = train_acc_res[train_acc_start_idx : train_acc_end_idx]
    statistic_res_mean[idx] = np.mean(train_acc_part)
    statistic_res_var[idx] = np.var(train_acc_part)
    
plt.plot(statistic_idx[start_plot_idx:], statistic_res_mean[start_plot_idx:],'k-')
######-----测试test------#####
iter_test_interval = 100
statistic_test_interval = 1
start_plot_idx = 0
maxacc = 0
maxindex = -1
test_acc_file = open('testacc.refine', 'r')
test_acc_total = test_acc_file.readlines()
test_acc_num = len(test_acc_total)
test_acc_res = np.zeros(test_acc_num)
test_acc_idx = np.arange(test_acc_num) * iter_test_interval

for idx in range(test_acc_num) :
    test_acc_str = test_acc_total[idx]
    str_start_idx = test_acc_str.find('=')+1
    str_end_idx = len(test_acc_str) - 1
    test_acc_res[idx] = float(test_acc_str[str_start_idx:str_end_idx])
    if test_acc_res[idx] > maxacc:
        maxacc = test_acc_res[idx]
        maxindex = idx
    
statistic_len = (test_acc_num + statistic_test_interval - 1)/statistic_test_interval
statistic_idx = np.arange(statistic_len) * iter_test_interval * statistic_test_interval
statistic_res_mean = np.zeros(statistic_len)
statistic_res_var = np.zeros(statistic_len)

for idx in range(statistic_len) :
    test_acc_start_idx = idx*statistic_test_interval
    test_acc_end_idx = min(test_acc_start_idx + statistic_test_interval, test_acc_num)
    test_acc_part = test_acc_res[test_acc_start_idx : test_acc_end_idx]
    statistic_res_mean[idx] = np.mean(test_acc_part)
    statistic_res_var[idx] = np.var(test_acc_part)

#plt.errorbar(statistic_idx[start_plot_idx:], statistic_res_mean[start_plot_idx:], statistic_res_var[start_plot_idx:])
plt.plot(statistic_idx[start_plot_idx:], statistic_res_mean[start_plot_idx:],'r-o')
plt.legend(('train','test'))
plt.title('train_val_acc')
plt.savefig('acc.png')
print "max acc=",maxacc,"index=",maxindex*iter_test_interval 
plt.show()
