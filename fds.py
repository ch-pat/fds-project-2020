import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

headers = ["satisfaction","Gender","Customer Type","Age","Type of Travel","Class","Flight Distance","Seat comfort","Departure/Arrival time convenient","Food and drink","Gate location","Inflight wifi service","Inflight entertainment","Online support","Ease of Online booking","On-board service","Leg room service","Baggage handling","Checkin service","Cleanliness","Online boarding","Departure Delay in Minutes","Arrival Delay in Minutes"]
data = pd.read_csv("./Invistico_Airline.csv", names=headers, header=0, engine='python')

# Cleanup the dataset
data.dropna(inplace=True)
data.drop(['Flight Distance','Departure/Arrival time convenient','Gate location','Departure Delay in Minutes','Arrival Delay in Minutes'], axis=1, inplace=True)

# Encode qualitative columns
Y = data.pop("satisfaction")

for i, v in Y.iteritems():
    if Y[i] == "satisfied":
        Y[i] = 1
    else:
        Y[i] = 0

ct = ColumnTransformer([('encoder', OneHotEncoder(), [0, 1, 3, 4])], remainder='passthrough')
data = np.array(ct.fit_transform(data), dtype=np.float)

X_train, X_test, Y_train, Y_test = train_test_split(data, Y, train_size=0.7, random_state=1)


print(X_train)
print(Y_train)
print(X_train.shape)

