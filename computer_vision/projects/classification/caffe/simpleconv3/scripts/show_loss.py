import numpy as np
import matplotlib.pyplot as plt 
import argparse

start_plot_idx = 1
def parse_args():
   parser = argparse.ArgumentParser(description='show loss acc')
   parser.add_argument('--file', dest='lossfile', help='the model', default='loss.refine', type=str)
   args = parser.parse_args()
   return args

def show_loss(lossfile,iter_interval,statistic_interval,lineset):
   loss_file = open(lossfile, 'r')
   loss_total = loss_file.readlines()
   loss_num = len(loss_total)
   loss_res = np.zeros(loss_num)
   loss_idx = np.arange(loss_num) * iter_interval

   for idx in range(loss_num) :
       loss_str = loss_total[idx]
       str_start_idx = loss_str.find('= ')+1
       str_end_idx = len(loss_str) - 1
       tmp = loss_str[str_start_idx+1:str_end_idx]
       #print "tmp=",tmp
       loss_res[idx] = float(tmp.split(' ')[0])
       
   statistic_len = (loss_num + statistic_interval - 1)/statistic_interval
   statistic_idx = np.arange(statistic_len) * iter_interval * statistic_interval
   statistic_res_mean = np.zeros(statistic_len)
   statistic_res_var = np.zeros(statistic_len)
   
   for idx in range(statistic_len) :
       loss_start_idx = idx*statistic_interval
       loss_end_idx = min(loss_start_idx + statistic_interval, loss_num)
       loss_part = loss_res[loss_start_idx : loss_end_idx]
       statistic_res_mean[idx] = np.mean(loss_part)
       statistic_res_var[idx] = np.var(loss_part)
       
   plt.plot(statistic_idx[start_plot_idx:], statistic_res_mean[start_plot_idx:],lineset)
if __name__ == '__main__':
    args = parse_args()
    show_loss('trainloss.refine',100,1,'k-')
    show_loss('testloss.refine',100,1,'r-o')
    plt.legend(('train','test'))
    plt.title('train_val_loss')
    plt.savefig('loss.png')
    plt.show()
