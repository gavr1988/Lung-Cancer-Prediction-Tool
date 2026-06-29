import pandas as pd

#loading the data
df=pd.read_csv('survey lung cancer.csv')

#printing column headers
print(df.columns)

#identifying duplicates
duplicates = df[df.duplicated()]
print ('Number of duplicate rows: ', duplicates.shape[0])

#Whilst there may be duplicate rows, we will not remove them as they may represent valid data points. Instead, we will keep them for analysis.

