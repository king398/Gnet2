import pandas as pd
import numpy as np

id = np.load(r"F:\Pycharm_projects\Gnet2\Sub\id.npy")
test_pred = np.load(r"F:\Pycharm_projects\Gnet2\Sub\testpred.npy")
print(test_pred[0], id[0])

test_df = pd.DataFrame({
	"id": id,
	"target": test_pred
})
test_df.head()
test_df.to_csv("F:\Pycharm_projects\Gnet2\Sub\TPU/submissioneff4fold.csv", index=False)
