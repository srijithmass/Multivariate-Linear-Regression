'''
Program to implement multivariate linear regression and predict the output.
Developed by: SRIJITH R
RegisterNumber: 212221240054
'''
import pandas as pd
from sklearn import linear_model
df=pd.read_csv("cars.csv")
x=df[['Weight', 'Volume']]
y=df['CO2']
regr=linear_model.LinearRegression()
regr.fit(x,y)
print('Coefficients: ', regr.coef_)
print('Intercept:', regr.intercept_)
predictedCO2=regr.predict([[3300, 1300]])
print('Predicted CO2 for the corresponding weight and volume:',predictedCO2)