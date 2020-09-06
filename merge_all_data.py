import numpy as np

a_18=(np.loadtxt('18_epsilon0.txt')+np.loadtxt('18_epsilon1.txt'))/2.0
a_20=(np.loadtxt('20_epsilon0.txt')+np.loadtxt('20_epsilon1.txt'))/2.0
a_22=(np.loadtxt('22_epsilon0.txt')+np.loadtxt('22_epsilon1.txt'))/2.0
a_24=(np.loadtxt('24_epsilon0.txt')+np.loadtxt('24_epsilon1.txt'))/2.0
a_25=(np.loadtxt('25_epsilon0.txt')+np.loadtxt('25_epsilon1.txt'))/2.0
a_26=(np.loadtxt('26_epsilon0.txt')+np.loadtxt('26_epsilon1.txt'))/2.0
a_27=(np.loadtxt('27_epsilon0.txt')+np.loadtxt('27_epsilon1.txt'))/2.0
a_28=(np.loadtxt('28_epsilon0.txt')+np.loadtxt('28_epsilon1.txt'))/2.0

a_18=a_18[49,:]
a_20=a_20[49,:]
a_22=a_22[49,:]
a_24=a_24[49,:]
a_25=a_25[49,:]
a_26=a_26[49,:]
a_27=a_27[49,:]
a_28=a_28[49,:]


a_merge = np.vstack([a_18,a_20,a_22,a_24,a_25,a_26,a_27,a_28])
svhn_val = a_merge[:,0]
lsun_val = a_merge[:,1]
timg_val = a_merge[:,2]
svhn_max = np.argmax(svhn_val)
lsun_max = np.argmax(lsun_val)
timg_max = np.argmax(timg_val)
print(a_merge[svhn_max,3:8])
print(a_merge[lsun_max,8:13])
print(a_merge[timg_max,13:18])



