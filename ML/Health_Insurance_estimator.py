# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Read in the insurance dataset
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

# Inspect first 5 entries of dataset
print(insurance.head())


# ML models only read numbers, so we need to convert string values to numbers.
# the pandas get_dummies() function does this in a one hot encoding fashion

# Turn all categories into numbers
insurance_one_hot = pd.get_dummies(insurance).astype(dtype=float)
print(insurance_one_hot.head()) # view the converted columns

'''
From this health insurance data set we want to predict how much would one spend in insurance ("charges" column) based on the combination of all other factors
'''
# Create X & y values
X = insurance_one_hot.drop("charges", axis=1) # All features minus the charges are our independent values
y = insurance_one_hot["charges"] # Insurance cost is what we want to predict

# Create training and test sets
from sklearn.model_selection import train_test_split

# train_test_split is a very cool function that allow us to create "train" and "test" batches of data from our data set
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # test batch size is 20% so train batch size is 80%
                                                    random_state=42) # set random state for reproducible splits. This ensure the same random shuffling and splitting of data

# Set random seed
tf.random.set_seed(42)

# Create a new model (same as model_2)
insurance_model = tf.keras.Sequential([
  tf.keras.layers.Dense(1), # 1 input layer
  tf.keras.layers.Dense(1)  # 1 output layer
])

# Compile the model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=['mae'])

# Fit the model
insurance_model.fit(X_train, y_train, epochs=100)

# Check the results of the insurance model
print("Insurance model: ", insurance_model.evaluate(X_test, y_test))
'''
This model won't perform well based on the mae.

'''


# Set random seed
tf.random.set_seed(42)

# Add an extra layer and increase number of units
insurance_model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(100), # 100 units
  tf.keras.layers.Dense(10), # 10 units
  tf.keras.layers.Dense(1) # 1 unit (important for output layer)
])

# Compile the model
insurance_model_2.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(), # Adam works but SGD doesn't 
                          metrics=['mae'])



# Fit the model and save the history (we can plot this). History will contain the loss and the list of parameters defined in "metrics" upon compilation of the model
history = insurance_model_2.fit(X_train, y_train, epochs=100, verbose=0)
history_2 = insurance_model_2.fit(X_train, y_train, epochs=200, verbose=0)

# Evaulate 3rd model
insurance_model_2_loss, insurance_model_2_mae = insurance_model_2.evaluate(X_test, y_test)
# Check the results of the insurance model
print("Insurance model 2: ", insurance_model_2.evaluate(X_test, y_test))

# Plot history (also known as a loss curve)
print(pd.DataFrame(history.history).head())
ax = pd.DataFrame(history.history).plot()
pd.DataFrame(history_2.history).plot(ax = ax)
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()

'''
How to improve the model:
- Add more data
- Modify number of layers
- Modify number of nodes in layers
- Change optimizer (and related parameters, e.g. learning rate)
- Fit for longer epochs
- Normalization and Standardization of data (covered below)
'''

# Standardization: Rescale set of numbers from 0 to 1
# Standardization: Subtract the mean of the data set (per column) to each value, so the standardized data are centered on 0. We enforce variace to be one by deviding everything by the standard deviation
#
# Both procedure can benefit to the learning of a NN. If two features have numbers on a very different scale, the NN might have troubles finding and understanding patterns.

# sklearn has several function to preprocess data in these ways. These functions are used collectively within a transformer

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder



# Create column transformer (this will help us normalize/preprocess our data)
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]), # get all values between 0 and 1 (Normalization)
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"]) # handle_unknown tells the transformer how to behave when it doesn't recognize colums specified in the list, otherwise
    # apply One _hot _encoding, i.e. get_dummies function seen before
)

# Create X & y
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

# Build our train and test sets (use random state to ensure same split as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit column transformer on the training data only (doing so on test data would result in data leakage)
ct.fit(X_train)

# Transform training and test data with normalization (MinMaxScalar) and one hot encoding (OneHotEncoder. They Must be transformed separately
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

print(X_train_normal[:5])

# Set random seed
tf.random.set_seed(42)

# Build the model (3 layers, 100, 10, 1 units)
insurance_model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])

# Compile the model
insurance_model_3.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=['mae'])

# Fit the model for 200 epochs (same as insurance_model_2)
insurance_model_3.fit(X_train_normal, y_train, epochs=200, verbose=0) 

# Evaulate 3rd model
insurance_model_3_loss, insurance_model_3_mae = insurance_model_3.evaluate(X_test_normal, y_test)

# Compare modelling results from non-normalized data and normalized data
print(insurance_model_2_mae, insurance_model_3_mae)

# One of the main advantages of Normalization: Faster convergence.