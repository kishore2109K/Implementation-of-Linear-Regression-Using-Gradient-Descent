# Ex.No-03:Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the dataset containing city populations feature

2.Normalize the features (mean normalization) to improve gradient descent convergence.

3.Compute predicted values.

4.Compute gradient of loss function.

5.Update weights using gradient descent.

## Program:

## Developed by: Kishore K

## RegisterNumber:212223040101 

~~~
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions - y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
~~~



~~~
X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X1_scaled)
print(Y1_scaled)
~~~




~~~
theta=linear_regression(X1_scaled,Y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
~~~


## Output:
![image](https://github.com/user-attachments/assets/2d59c6b6-9008-44ab-bfe2-b52f125d0706)

## Scaled value:
![image](https://github.com/user-attachments/assets/5ef52654-8859-415c-9237-d502f0bebb5a)

## Predicted value:
![image](https://github.com/user-attachments/assets/ada06a6b-704b-4326-ab26-d8283a68500b)






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
