#installing the required libraries
import pandas as pd
#To help reduce overfitting, we will use train_test_split to split the dataset into training and testing sets, allowing us to evaluate the model's performance on unseen data.
from sklearn.datasets import fetch_california_housing
#To help reduce overfitting, we will use GridSearchCV to perform hyperparameter tuning and find the best combination of hyperparameters for the model, which can help improve its performance and reduce overfitting.
from sklearn.model_selection import train_test_split, GridSearchCV
#To help reduce overfitting, we will use Pipeline to streamline the process of applying multiple transformations and modeling steps in a single workflow, which can help prevent data leakage and improve model performance.
from sklearn.pipeline import Pipeline
# To help reduce overfitting, we will use StandardScaler to standardise the features by removing the mean and scaling to unit variance.
from sklearn.preprocessing import StandardScaler
#To Help reduce overfitting, we will use Random Forest Regressor, which is an ensemble learning method that combines multiple decision trees to improve predictive performance and reduce overfitting.
from sklearn.ensemble import RandomForestRegressor 


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

# Creating the ML pipeline

#Loading the data from the cleaned CSV file

df = pd.read_csv('cleaned_survey_lung_cancer.csv')

#Splitting the data into features (X) and target variable (Y)
#X variable will be removed from the data frame to help calculate Y (In this case is lung_Cancer)
X = df.drop('LUNG_CANCER', axis=1)  
#Y variable will be the target variable we want to predict (In this case is LUNG_CANCER)
Y = df['LUNG_CANCER']

print ()
#Splitting for training and testing sets, with 20% of the data reserved for testing. The random_state parameter is set to ensure reproducibility of the results.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 101)

