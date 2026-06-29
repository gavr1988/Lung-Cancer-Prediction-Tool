import pandas as pd

#loading the data
df=pd.read_csv('survey lung cancer.csv')

#printing column headers
print(df.columns)

print (df.head)

#checking column data types
print (df.dtypes)

#identifying duplicates
duplicates = df[df.duplicated()]
print ('Number of duplicate rows: ', duplicates.shape[0])

#Whilst there may be duplicate rows, we will not remove them as they may represent valid data points. Instead, we will keep them for analysis.

# checking for missing  values
print("Missing Values:")
print (df.isnull().sum())

#Cleaning Columns titles
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace (' ', '_')
#checking columns
print (df.columns)

#Changing M to 1 and F to 0 in the Gender Column
#This is to convert the Gender column into a binary format for analysis

df ['GENDER'] = df['GENDER'].map({
        'M': 0,
        'F': 0
})

#exporting to a new 'cleaned' csv file
df.to_csv ('cleaned_survey_lung_cancer.csv', index=False)

