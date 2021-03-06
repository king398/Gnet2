import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def effficientoof():
	img_ids_fold1 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\id_fold1.npy")
	img_ids_fold2 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\id_fold2.npy")
	img_ids_fold3 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\id_fold3.npy")

	img_ids = np.concatenate((img_ids_fold1, img_ids_fold2, img_ids_fold3, img_ids_fold4))
	y_pred_fold1 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_pred_fold1.npy")
	y_pred_fold2 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_pred_fold2.npy")
	y_pred_fold3 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_pred_fold3.npy")

	y_pred = np.concatenate((y_pred_fold1, y_pred_fold2, y_pred_fold3, y_pred_fold4))
	y_true_fold1 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_true_fold1.npy")
	y_true_fold2 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_true_fold2.npy")
	y_true_fold3 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_true_fold3.npy")

	y_true = np.concatenate((y_true_fold1, y_true_fold2, y_true_fold3, y_true_fold4))
	df = pd.DataFrame({
		"y_true": y_true,
		"y_pred": y_pred,
		"id": img_ids
	})
	df.head()
	auc = roc_auc_score(y_true=y_true, y_score=y_pred)
	print(f"AUC: {auc:.20f}")
	df.to_csv(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788/oof.csv", index=False)
	np.save("F:\Pycharm_projects\Gnet2\OOF\Full preds/y_pred_Efficinet", y_pred)
	print(np.average(y_pred))


def SEresnetoof():
	img_ids_fold1 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\id_fold1.npy")
	img_ids_fold2 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\id_fold2.npy")
	img_ids_fold3 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\id_fold3.npy")
	img_ids_fold4 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\id_fold0.npy")

	img_ids = np.concatenate((img_ids_fold1, img_ids_fold2, img_ids_fold3, img_ids_fold4))
	y_pred_fold1 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\y_pred_fold1.npy")
	y_pred_fold2 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\y_pred_fold2.npy")
	y_pred_fold3 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\y_pred_fold3.npy")
	y_pred_fold4 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_pred_fold0.npy")

	y_pred = np.concatenate((y_pred_fold1, y_pred_fold2, y_pred_fold3, y_pred_fold4))
	y_true_fold1 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\y_true_fold1.npy")
	y_true_fold2 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\y_true_fold2.npy")
	y_true_fold3 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\y_true_fold3.npy")
	y_true_fold4 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\ronit 8788\y_true_fold0.npy")

	y_true = np.concatenate((y_true_fold1, y_true_fold2, y_true_fold3, y_true_fold4))
	df = pd.DataFrame({
		"y_true": y_true,
		"y_pred": y_pred,
		"id": img_ids
	})
	df.head()
	auc = roc_auc_score(y_true=y_true, y_score=y_pred)
	print(f"AUC: {auc:.20f}")
	df.to_csv(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34/oof.csv", index=False)
	np.save("F:\Pycharm_projects\Gnet2\OOF\Full preds/y_pred_SEresnet", y_pred)
	return auc
SEresnetoof()

def ensemble():
	img_ids_fold1 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\id_fold1.npy")
	img_ids_fold2 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\id_fold2.npy")
	img_ids_fold3 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\id_fold3.npy")
	img_ids = np.concatenate((img_ids_fold1, img_ids_fold2, img_ids_fold3))

	y_true_fold1 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\y_true_fold1.npy")
	y_true_fold2 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\y_true_fold2.npy")
	y_true_fold3 = np.load(r"F:\Pycharm_projects\Gnet2\OOF\SERESNET34\y_true_fold3.npy")
	y_true = np.concatenate((y_true_fold1, y_true_fold2, y_true_fold3))
	y_pred = (np.load("F:\Pycharm_projects\Gnet2\OOF\Full preds\y_pred_Efficinet.npy") + np.load(
		"F:\Pycharm_projects\Gnet2\OOF\Full preds\y_pred_SEresnet.npy") / 2)
	df = pd.DataFrame({
		"y_true": y_true,
		"y_pred": y_pred,
		"id": img_ids
	})
	df.head()
	auc = roc_auc_score(y_true=y_true, y_score=y_pred)
	print(f"AUC: {auc:.4f}")
	df.to_csv(r"F:\Pycharm_projects\Gnet2\OOF\Full oof/oofSEresnet+EfficinetNet.csv", index=False)
	return auc
