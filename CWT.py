import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt

scales = range(1, 2)
x = np.load(r"F:\Pycharm_projects\Gnet2\data\00000e74ad.npy")
waves = np.hstack(x)
waves = waves / np.max(waves)
print(x.shape)

cwtmatr = pywt.cwt(x, scales, "morl", 1)
plt.plot(cwtmatr)
plt.show()
