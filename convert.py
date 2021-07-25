import time
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import os
import numpy as np
from multiprocessing import Process

start = time.process_time()
files = tf.io.gfile.glob("F:\Pycharm_projects\Gnet\data/*/*/*/*.npy")
import numpy as np

def f():
	for i in tqdm(files):
		files_save = i.split("\\")
		array = np.load(i).astype(np.float16)
		os.makedirs("E:\\Data\\float16/"+ files_save[4]+"/"+files_save[5]+"/"+files_save[6]+"/",exist_ok=True)
		np.save(file=r"E:\Data\float16"+"/"+ files_save[4]+"/"+files_save[5]+"/"+files_save[6]+"/"+files_save[7],arr=array)
	
f()
    



	
	

