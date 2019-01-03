import pandas as pd
import numpy as np

train_data = pd.read_csv('kc_house_train_data.csv')
test_data = pd.read_csv('kc_house_test_data.csv')

input_feature = train_data['sqft_living']
sqft_living_lst = [i for i in train_data['sqft_living']] # seriers converts into lst
sqft_living_array = np.array(sqft_living_lst) #lst converts into array

output = train_data['price']
price_lst = [i for i in train_data['price']]
price_array = np.array(price_lst)
#print(price_array)

def simple_linear_regression(input_feature,output):
    numerator = (input_feature * output).mean(axis =0) - (input_feature.mean(axis =0))*(output.mean(axis=0)) #axis = 0  work on the row
    denominator =(input_feature**2).mean(axis=0) - (input_feature.mean(axis=0))*(input_feature.mean(axis=0))
    slope = numerator/ denominator
    intercept = output.mean(axis= 0) - slope *(input_feature.mean(axis=0))
    return slope, intercept
slope_train,intercept_train = simple_linear_regression(sqft_living_array, price_array)

def get_regression_predictions(input_feature,slope, intercept):
    predicted_output = intercept + input_feature * slope
    return predicted_output

'''Quiz Question: Using your Slope and Intercept from (4), What is the predicted price for a house with 2650 sqft?'''
print(get_regression_predictions(2650,slope_train,intercept_train))
'''What is the RSS for the simple linear regression using squarefeet to predict prices on TRAINING data'''
def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    RSS =(((intercept + input_feature * slope)- output)**2).sum(axis=0)
    return RSS

print(get_residual_sum_of_squares(sqft_living_array,price_array,intercept_train,slope_train))

'''Quiz Question: According to this function and the slope and intercept from 
(4) What is the RSS for the simple linear regression using squarefeet to predict prices on TRAINING data?'''

def inverse_regression_predictions(output, intercept, slope):
    estimated_input = (output - intercept) / slope
    return(estimated_input)

'''According to this function and the regression slope and intercept from (3) what is the estimated square-feet for a house costing $800,000?'''

print(inverse_regression_predictions(800000,intercept_train,slope_train))

'''Which model (square feet or bedrooms) has lowest RSS on TEST data? Think about why this might be the case.'''
sqft_living_array_test = np.array([ i for i in test_data['sqft_living']])
bedrooms_array_test = np.array([i for i in test_data['bedrooms']])
price_array_test = np.array([i for i in test_data['price']])

rss_sqft =get_residual_sum_of_squares(sqft_living_array_test,price_array_test,intercept_train,slope_train)
print('rss_sqft:', rss_sqft)

bed_rss = get_residual_sum_of_squares(bedrooms_array_test,price_array_test,intercept_train,slope_train)
print('bed_rss:',bed_rss)

print (rss_sqft - bed_rss)