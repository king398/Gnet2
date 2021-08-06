import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore") 


sub1 = pd.read_csv('F:\Pycharm_projects\Gnet2\Sub\submission.csv').sort_values('id') 
sub2 = pd.read_csv('F:\Pycharm_projects\Gnet2\Sub\submission866.csv').sort_values('id') 
preds1 = sub1.target
preds2 = sub2.target
sub = sub1.copy()
sub.loc[:, 'target'] =preds1*0.499+preds2*0.501
sub.to_csv('F:\Pycharm_projects\Gnet2\Sub/submission.csv', index=False)
