import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
train_data = pd.read_csv('kc_house_train_data.csv')
test_data = pd.read_csv('kc_house_test_data.csv')

#create new features
train_data['bedrooms_squared'] = train_data['bedrooms'] * train_data['bedrooms']
train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms']
train_data['log_sqft_living'] = np.log(train_data['sqft_living'])
train_data['lat_plus_long'] = train_data['lat']+ train_data['long']
#print(train_data.loc[:,['lat']])
#print(train_data.loc[:,['long']])
test_data['bedrooms_squared'] = test_data['bedrooms'] * test_data['bedrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms'] * test_data['bathrooms']
test_data['log_sqft_living'] = np.log(test_data['sqft_living'])
test_data['lat_plus_long'] = test_data['lat']+ train_data['long']

# Quiz Question: what are the mean (arithmetic average) values of your 4 new variables on TEST data?

new_variables_mean = np.mean(test_data[['bedrooms_squared','bed_bath_rooms','log_sqft_living','lat_plus_long']])
print(new_variables_mean)

#Model 1: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’, and ‘long’
x_1 = train_data[['sqft_living','bedrooms','bathrooms','lat','long']]
y_1 = train_data[['price']]
y_test = test_data[['price']]
model_1 = LinearRegression().fit(x_1,y_1)
y_predict_1=model_1.predict(x_1)
print(mean_squared_error(y_1, y_predict_1))
#print(model_1.coef_)

#Model 2: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’,‘long’, and ‘bed_bath_rooms’
x_2 = train_data[['sqft_living','bedrooms','bathrooms','lat','long','bed_bath_rooms']]
y_2 = train_data[['price']]
model_2 = LinearRegression().fit(x_2,y_2)
y_predict_2 = model_2.predict(x_2)
print(mean_squared_error(y_2, y_predict_2))
#print(model_2.coef_)

#Model 3: ‘sqft_living’, ‘bedrooms’, ‘bathrooms’, ‘lat’,‘long’, ‘bed_bath_rooms’, ‘bedrooms_squared’, ‘log_sqft_living’,
# and ‘lat_plus_long’

x_3 = train_data[['sqft_living','bedrooms','bathrooms','lat','long','bed_bath_rooms','bedrooms_squared','log_sqft_living','lat_plus_long']]
y_3 = train_data[['price']]
model_3 = LinearRegression().fit(x_3,y_3)
y_predict_3 = model_3.predict(x_3)
print(mean_squared_error(y_3, y_predict_3))
#print(model_3.coef_)
