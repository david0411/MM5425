import pandas as pd

df = pd.read_csv(r"supermarket_sales.csv")

d1 = df[0:3]
d2 = df[5:9]
d3 = pd.concat([d1,d2], ignore_index=True, axis=0)

print(d3)
