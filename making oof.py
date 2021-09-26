import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

img_ids_fold1 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\id_fold1.npy")
img_ids_fold2 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\id_fold2.npy")
img_ids_fold3 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\id_fold3.npy")
img_ids = np.concatenate((img_ids_fold1, img_ids_fold2, img_ids_fold3))
y_pred_fold1 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_pred_fold1.npy")
y_pred_fold2 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_pred_fold2.npy")
y_pred_fold3 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_pred_fold3.npy")
y_pred = np.concatenate((y_pred_fold1, y_pred_fold2, y_pred_fold3))
y_true_fold1 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_true_fold1.npy")
y_true_fold2 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_true_fold2.npy")
y_true_fold3 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_true_fold3.npy")
y_true = np.concatenate((y_true_fold1, y_true_fold2, y_true_fold3))
df = pd.DataFrame({
	"y_true": y_true,
	"y_pred": y_pred,
	"id": img_ids
})
df.head()
auc = roc_auc_score(y_true=y_true, y_score=y_pred)
print(f"AUC: {auc:.7f}")
df.to_csv(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788/oof.csv", index=False)
np.average(y_pred)
np.average(y_true)
print(np.average(y_true), np.average(y_pred))
