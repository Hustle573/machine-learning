import pandas as pd 
import numpy as np

#Loading data
data=pd.read_csv("C:/Users/Allen/OneDrive - CHRIST UNIVERSITY/Documents/Internship/predictive analysis/train_data.csv")
data=data.drop(['Name','State','City','Date'],axis='columns')
yes_no_columns = ['Status']
for col in yes_no_columns:
    data[col].replace({'Interested': 1,'Not interested': 0},inplace=True)
data = pd.get_dummies(data=data, columns=['Gender','Festival','Offer','city tier','Area','Source','Month','Day'])
# test_train split
X = data.drop('Status', axis='columns')
y = data['Status']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


#model building
import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(46, input_shape=(46,), activation='relu'),
    keras.layers.Dense(36, activation='gelu'),
    keras.layers.Dense(26, activation='relu'),
    keras.layers.Dense(16, activation='gelu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=500)

y_pred=[]

pred=model.predict(X_test)
for i in range(len(pred)):
  if i>0.5:
    y_pred.append(1)
  else:
    y_pred.append(0)
model.save("status_prediction_model.json")

