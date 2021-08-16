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
import numpy as np
import pywt
import torch
import torch.nn as nn
from scipy import signal
import matplotlib.pyplot as plt

t = np.linspace(-1, 1, 200, endpoint=False)
sig = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)

plt.plot(t, sig);