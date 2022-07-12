#setup
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *
print("Setup Complete")

#Specify Prediction Target
# print the list of columns in the dataset to find the name of the prediction target
home_data.columns
y = home_data.SalePrice #the prediction taget

#Create X
feature_names =['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = home_data[feature_names]
print(X.describe()) # print description or statistics from X
print(X.head()) # print the top few lines

#Specify and Fit Model
from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X,y)

#Make predictions with the model's predict command using X as the data. Save the results to a variable called predictions
predictions = iowa_model.predict(X)
print(predictions)
