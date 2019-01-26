import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.NaN) #list all numbers
train_data = pd.read_csv('kc_house_train_data.csv')
test_data = pd.read_csv('kc_house_test_data.csv')
#print(train_data.shape[0]) #show the amount of rows %  shape[1] shows the colunm

def get_numpy_data(data,features,output):
    feature_matrix = np.mat(data[features])
    feature_matrix = np.column_stack((np.ones(feature_matrix.shape[0]),feature_matrix)) #add constant column
    output_matrix = np.mat(data[output])
    return feature_matrix,output_matrix

def predict_outcome(feature_matrix,weights):
    prediction = np.dot(feature_matrix,weights)
    return prediction

def feature_derivative(errors,feature):
    derivative = 2 * np.dot(feature.T,errors)
    return derivative

def regress_gradient_descent(feature_matirx,output,initial_weights,step_size,tolerance):
    converged = False
    weights = np.array(initial_weights, dtype = np.float64)
    output  = np.array(output)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions = predict_outcome(feature_matirx,weights)
        # compute the errors as predictions - output:
        errors = predictions - output
        gradient_sum_squares = 0
        for i in range(len(weights)):
            # compute the derivative for weitht[i]
            derivative = feature_derivative(errors ,feature_matirx[:, i]) #[17384,2]
            #add the squared derivative to the gradient magnitude
            gradient_sum_squares += derivative ** 2
            #update the weight[i] based on step size and derivative:
            weights[i] = weights[i]- step_size * derivative
        gradient_magnitude = np.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return weights


simple_features=['sqft_living']
my_output=['price']
(simple_feature_matrix,output)=get_numpy_data(train_data,simple_features,my_output)
initial_weights= np.mat([-47000.,1.]).T
step_size = 7e-12
tolerance = 2.5e7

#Use these parameters to estimate the slope and intercept for predicting prices based only on ‘sqft_living’.
sqft_living_weights = regress_gradient_descent(simple_feature_matrix,output,initial_weights,step_size,tolerance)
print(sqft_living_weights[1])


#Now build a corresponding ‘test_simple_feature_matrix’ and ‘test_output’ using test_data. Using ‘test_simple_feature_matrix’ and
# ‘simple_weights’ compute the predicted house prices on all the test data.
test_feature = ['sqft_living']
test_output = ['price']
(test_feature,test_output) = get_numpy_data(test_data,test_feature,test_output)
model1_weights = regress_gradient_descent(test_feature,test_output,initial_weights,step_size,tolerance)


#11. Quiz Question: What is the predicted price for the 1st house in the Test data set for model 1 (round to nearest dollar)?
predcition_test = predict_outcome(test_feature,model1_weights)
print(predcition_test[0])

#12. Now compute RSS on all test data for this model. Record the value and store it for later
errors_test = predcition_test - test_output
rss_test = (errors_test * errors_test.T)

#Now we will use the gradient descent to fit a model with more than 1 predictor variable (and an intercept). Use the following parameters:
model_features = ['sqft_living','sqft_living15']
my_output = ['price']
(feature_matrix_1,output_1) = get_numpy_data(train_data,model_features,my_output)
initial_weights_1 = np.mat([-100000.,1.,1.]).T
step_size_1 = 4e-12
tolerance_1 = 1e9

model2_weights = regress_gradient_descent(feature_matrix_1,output_1,initial_weights_1,step_size_1,tolerance_1)

#Use the regression weights from this second model (using sqft_living and sqft_living_15) and predict the outcome of all the house prices on the TEST data.
(test_feature_2,test_output_2) = get_numpy_data(test_data,model_features,my_output)
model2_test = predict_outcome(test_feature_2,model2_weights)
print(model2_test[0])

#Quiz Question: Which estimate was closer to the true price for the 1st house on the TEST data set, model 1 or model 2?
#Now compute RSS on all test data for the second model. Record the value and store it for later.
errors_test2 = model2_test - test_output_2
rss_2 = (errors_test2 * errors_test2.T)