import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn

df = pd.read_csv('candidates.csv')
#new_df  = pd.Dataframe('candidates.csv')

abc = df.head(50)
#c = df[0:10000,4]
#c.boxplot
c = df['class'].value_counts()
print(c)
temp = df['class'].mean()
print(temp)

print(abc)
