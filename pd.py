import pandas as pd

csv = pd.read_csv(r"C:\Users\pc\Desktop\STOCK.csv")
print(csv.columns.values.tolist())
