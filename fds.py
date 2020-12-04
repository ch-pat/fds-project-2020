import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import fdsfunctions as fds
import seaborn as sns
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

data.insert(0, "Ones", [1 for i in range(data.shape[0])], True) 

ct = ColumnTransformer([('encoder', OneHotEncoder(), [1, 2, 4, 5])], remainder='passthrough')
data = np.array(ct.fit_transform(data), dtype=np.float)
# TODO: Ones column does not contain only ones after encoding

X_train, X_test, Y_train, Y_test = train_test_split(data, Y, train_size=0.5, random_state=1)
Y_train = np.array(Y_train, dtype=np.float32)
Y_test = np.array(Y_test, dtype=np.float32)

theta0 = np.zeros(X_train.shape[1])

# Run Gradient Ascent method
n_iter=50000
theta_final, log_l_history, theta_history = fds.gradient_ascent(theta0, X_train, Y_train, fds.grad_l, alpha=0.002, iterations=n_iter)

fig, ax = plt.subplots(num=2)

ax.set_ylabel('l(Theta)')
ax.set_xlabel('Iterations')
_=ax.plot(range(len(log_l_history)),log_l_history,'b.')
plt.show()

print(X_test.shape, theta_final.shape, Y_test.shape)
correct_predictions = 0
for i in range(X_test.shape[0]):
    prediction = theta_final.T.dot(X_test[i, :])
    if prediction < 0.5:
        prediction = 0
    else:
        prediction = 1
    if prediction == Y_test[i]:
        correct_predictions += 1
    
print(f"\nNumber of correct predictions: {correct_predictions}, total predictions: {X_test.shape[0]}, accuracy: {correct_predictions / X_test.shape[0]}")