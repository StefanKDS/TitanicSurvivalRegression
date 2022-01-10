# Import required libraries
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Read in the insurance dataset
train_data = pd.read_csv("Data/train.csv")
#print(train_data.head())

# Missing data ?
#print(train_data.isnull().sum())
#print(train_data.shape)

# Fill missing values
train_data["Age"] = train_data["Age"].replace(np.NaN, train_data["Age"].mean())
train_data["Cabin"] = train_data["Cabin"].fillna('U')
train_data["Embarked"] = train_data["Embarked"].fillna('U')

# Missing data check
#print(train_data.isnull().sum())
#print(train_data.shape)

ct = make_column_transformer(
    (MinMaxScaler(), ["Pclass", "Age", "SibSp", "Parch", "Fare"]), # get all values between 0 and 1
    (OneHotEncoder(handle_unknown="ignore"), ["Sex", "Ticket", "Embarked"]))

# Create X & y
train_data = train_data.drop(["Name", "PassengerId"], axis=1)

# Create X & y values
X = train_data.drop("Survived", axis=1)
y = train_data["Survived"]

# Build our train and test sets (use random state to ensure same split as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit column transformer on the training data only (doing so on test data would result in data leakage)
ct.fit(X_train)

# Transform training and test data with normalization (MinMaxScalar) and one hot encoding (OneHotEncoder)
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', min_lr=0.001, patience=5, mode='min', verbose=1)
early_stopping = EarlyStopping(patience=10, monitor='val_accuracy')
callbacks = [early_stopping,reduce_lr]

# Set random seed
tf.random.set_seed(42)

# Build the model (3 layers, 100, 10, 1 units)
model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(100, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model_1.compile(loss = 'binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=['accuracy'])

X_train_normal = X_train_normal.toarray()
X_test_normal = X_test_normal.toarray()

# Fit the model
history = model_1.fit(X_train_normal, y_train, epochs=100, callbacks=callbacks, validation_data=(X_test_normal, y_test))

# Save model
np.save('Auswertung/history.npy', history.history)
model_1.save('Auswertung/model1')

import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()

from helper_functions import plot_saved_loss_curves
plot_saved_loss_curves('Auswertung/history.npy')
