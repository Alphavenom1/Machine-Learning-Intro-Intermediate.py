#setup
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]
iowa_model = DecisionTreeRegressor()
iowa_model.fit(X, y)
print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex4 import *
print("Setup Complete")

#Use the train_test_split function to split up your data
from sklearn.model_selection import train_test_split
train_X,Val_X,train_Y,Val_y=train_test_split(X,y,random_state=1)

#Create a DecisionTreeRegressor model and fit it to the relevant data
iowa_model =DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X,train_y)

#Make Predictions with Validation data
val_predictions =iowa_model.predict(val_X)

#Calculate the Mean Absolute Error in Validation Data
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y,val_predictions)
print(val_mae)

