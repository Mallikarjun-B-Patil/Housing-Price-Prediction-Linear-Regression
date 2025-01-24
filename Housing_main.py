# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from joblib import dump

# Load the dataset
df = pd.read_csv("Housing.csv")

# Data preprocessing
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.drop(columns=["furnishingstatus"], inplace=True)

# Converting the categorical columns to numerical columns using One Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# Splitting the data into Features(x) and Target(y)
x = df.drop("price", axis=1)
y = df["price"]

# Split the data into Training and Testing using train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Choose Linear Regression Model and fit it for training data
model = LinearRegression()
model.fit(x_train, y_train)

# Save the trained model to a file
dump(model, 'model.joblib')