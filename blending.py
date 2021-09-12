import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore") 


sub1 = pd.read_csv('F:\Pycharm_projects\Gnet2\Sub\TPU\TEST_EfficientNetB7_512_21.csv').sort_values('id') 
sub2 = pd.read_csv('F:\Pycharm_projects\Gnet2\Sub\TPU\TEST_EfficientNetB7_512_1991_CQT.csv').sort_values('id') 
preds1 = sub1.target
preds2 = sub2.target

sub3 = pd.read_csv('F:\Pycharm_projects\Gnet2\Sub\TPU\TEST_EfficientNetB7_512_1991.csv').sort_values('id') 
sub4 = pd.read_csv('F:\Pycharm_projects\Gnet2\Sub\TPU\TEST_EfficientNetB7_512_2020.csv').sort_values('id') 
preds3 = sub3.target
preds4 = sub4.target
sub = sub1.copy()
sub.loc[:, 'target'] =preds1*0.25+preds2*0.25+preds3*0.25+preds4*0.25
sub.to_csv('F:\Pycharm_projects\Gnet2\Sub/submission.csv', index=False)

