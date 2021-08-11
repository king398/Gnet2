import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt

scales = range(1, 2)
x = np.load(r"F:\Pycharm_projects\Gnet2\data\00000e74ad.npy")

print(x.shape)
print(x)
cwtmatr = signal.cwt(x, signal.ricker, scales)
print(cwtmatr.shape)
plt.imshow(cwtmatr)
plt.show()
