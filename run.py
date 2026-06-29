#installing the required libraries
import pandas as pd


def main():
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

	#Save the cleaned dataframe to a new CSV file so the updated data can be reused later
	#exporting to a new 'cleaned' csv file
	df.to_csv ('cleaned_survey_lung_cancer.csv', index=False)
	print('Cleaned CSV saved to cleaned_survey_lung_cancer.csv')

# if_name__ == '__main__': this is to ensure that the main function is only executed when the script is run directly, and not when it is imported as a module in another script.
if __name__ == '__main__':
	main()


