import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

sub1 = pd.read_csv(r"F:\Pycharm_projects\Gnet2\notebooks\blending.py").sort_values('id').reset_index(drop=True)
sub2 = pd.read_csv('F:\Pycharm_projects\Gnet2\Sub\submission (1).csv').sort_values('id').reset_index(drop=True)
preds1 = sub1.target
preds2 = sub2.target

sub = sub2.copy()

sub.loc[:, 'target'] = preds1 * 0.6 + preds2 * 0.4
sub.to_csv(r'F:\Pycharm_projects\Gnet2\Sub/submission.csv', index=False)

print(sub.head())
