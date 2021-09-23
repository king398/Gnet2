import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

sub1 = pd.read_csv(r"F:\Pycharm_projects\Gnet2\Sub/submission.csv").sort_values('id').reset_index(
	drop=True)
sub2 = pd.read_csv('F:\Pycharm_projects\Gnet2\Sub\submission_egor.csv').sort_values('id').reset_index(drop=True)
preds1 = sub1.target
preds2 = sub2.target

sub = sub2.copy()

sub.loc[:, 'target'] = preds1 * 0.5 + preds2 * 0.5
sub.to_csv(r'F:\Pycharm_projects\Gnet2\Sub/submission.csv', index=False)

print(sub.head())
