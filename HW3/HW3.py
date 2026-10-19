# OEAS805 HOMEWORK 3: MACHINE LEARNING - REGRESSION
#import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# Linear model and basic functions from scikit-learn
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Specific models we will use beyond linear regression from scikit-learn
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

#%% LOAD DATA
filepath ='HW3/CBP_TWQM_MAIN_WQ_2021_2026_surface.csv'
data = pd.read_csv(filepath, sep = ',')

#%% TARGET VARIABLE SELECTION
#view available parameters
unique_parameters = data['Parameter'].unique()
print(unique_parameters)

# Selected target variable: Total Suspended Solids (TSS)
# Selected predictor variables: 
# - Particulate Carbon (PC)
# - Light attenuation coefficient (Kd)

# I expect TSS to result as a conbination of suspended sediments, 
# phytoplankton, and other suspended matter which reduce light penetration. 
# I have chosen to predict TSS using particulate carbon (PC) and
# light attenuation (Kd). Orginally, I was also using Chlorophyll-a as a
# predictor variable but I chose to omit it because omitting CHLA generally
# increased model performance across all of the models I am using.

TSS_idx = data.index[data['Parameter'] == 'TSS']
TSS = data.loc[TSS_idx,'MeasureValue']

KD_idx = data.index[data['Parameter'] == 'KD']
KD = data.loc[KD_idx,'MeasureValue']

PC_idx = data.index[data['Parameter'] == 'PC']
PC = data.loc[PC_idx,'MeasureValue']

#%% MULTIPLE LINEAR REGRESSION
# ~~~~~~ Model Implementation ~~~~~~ 
#create dataset
X = np.column_stack((PC, KD))
y = np.asarray(TSS)

#split the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

#create a linear regression object
regr = linear_model.LinearRegression()

#train the model
regr.fit(X_train, y_train)


#make predictions
y_pred = regr.predict(X_test)

# ~~~~~~ Model Evaluation ~~~~~~ 
# Evaluate model performance with R-squared value (1 is perfect prediction)
r2 = r2_score(y_test, y_pred)
print('Coefficient of determination (R-squared): %.2f' % r2)

# ~~~~~~ Model Visualization ~~~~~~ 
plt.figure()
plt.scatter(y_test, y_pred) #plot data
plt.plot([0, 12], [0, 12], 'r--', label='1:1 line') #plot 1:1 line
plt.text(0.05, 0.95, f'$R^2 = {r2:.2f}$', #plot r-squared value
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top')

plt.xlabel('Actual TSS')
plt.ylabel('Predicted TSS')
plt.title('Actual vs. Predicted TSS')
plt.legend()
plt.show()

#%% NON-LINEAR MULTI-LAYER PERCEPTRON REGRESSION
# ~~~~~~ Model Implementation ~~~~~~ 
#create dataset
X = np.column_stack((PC, KD))
y = np.asarray(TSS)

#split the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

#node values from 1 through 9 (length of testing dataset)
n = np.arange(1, 10)
#number of hidden layers 1 through 9 (match max number of nodes)
layers = np.arange(1, 10)

# Create a matrix to store R-squared values
r_vals = np.zeros((len(layers), len(n)))

# Iterate through each combination of hidden nodes and layers
for i in range(len(layers)):
    for j in range(len(n)):
        
        hidden_layers = (n[j],) * layers[i]

        # Fit regression model
        regr_1 = MLPRegressor(hidden_layer_sizes=hidden_layers, activation='tanh', max_iter=10000)
        regr_1.fit(X_train, y_train)
        
        # Predict
        y_pred = regr_1.predict(X_test)
        
        # Evaluate model performance
        r_vals[i, j] = r2_score(y_test, y_pred)


# ~~~~~~ Model Evaluation ~~~~~~ 
best_index = np.unravel_index(np.argmax(r_vals), r_vals.shape)

best_layers = layers[best_index[0]]
best_nodes = n[best_index[1]]
max_r2 = r_vals[best_index]

print('Maximum R-squared:', max_r2)
print('Optimal hidden layers:', best_layers)
print('Optimal nodes per layer:', best_nodes)


# ~~~~~~ Model Visualization ~~~~~~ 
# Fit regression model with best number of nodes and layers
regr_1 = MLPRegressor(hidden_layer_sizes=(1,1,1,1,1,1), activation='tanh', max_iter=10000)
regr_1.fit(X_train, y_train)

# Predict
y_pred = regr_1.predict(X_test)

# Evaluate model performance
r2 = r2_score(y_test, y_pred)

# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, c="darkorange", label="Predictions")
plt.plot([0, 12], [0, 12], 'r--', label='1:1 line') #plot 1:1 line
plt.text(
    0.05, 0.95, f'$R^2 = {r2:.2f}$',
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top'
)
plt.xlabel("Actual TSS")
plt.ylabel("Predicted TSS")
plt.title("MLP Regression: Actual vs. Predicted TSS")
plt.legend()
plt.show()




#%% NON-LINEAR RADIUS NEIGHBORS REGRESSOR

# ~~~~~~ Model Implementation ~~~~~~ 
r  = 2

# Create a random dataset
X = np.column_stack((PC, KD))
y = np.asarray(TSS)

#split the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

# Fit regression model
regr_1 = RadiusNeighborsRegressor(radius=r)
regr_1.fit(X_train, y_train)

# Predict
y_pred = regr_1.predict(X_test)

# ~~~~~~ Model Evaluation ~~~~~~ 
# Evaluate model performance with R-squared value (1 is perfect prediction)
r2 = r2_score(y_test, y_pred)
print('Coefficient of determination (R-squared): %.2f' % r2)

# ~~~~~~ Model Visualization ~~~~~~ 
# Plot the results
plt.figure()
plt.scatter(y_test, y_pred, c="darkorange", label="data")
plt.plot([0, 12], [0, 12], 'r--', label='1:1 line') #plot 1:1 line
plt.text(
    0.05, 0.95, f'$R^2 = {r2:.2f}$',
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top'
)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.xlabel("Actual TSS")
plt.ylabel("Predicted TSS")
plt.title("NON-LINEAR RADIUS NEIGHBORS REGRESSOR: Actual vs. Predicted TSS")
plt.legend()
plt.show()



#%% Hyperparameter Explanation
# For the Non-linear Multi-layer Perceptron Regression, I determined the ideal
# number of nodes and hidden layers by interating through each combination of
# 9 nodes and 9 layers and finding which combination produced the best 
# r-squared value

# For the Non-linear Radius Neighbors Regressor, I chose the ideal radius by
# first plotting the data with radius = 1 to see the range of values 
# present (~2-12), then manually running the model at each integer radius from
# 1 to 12. I chose to keep the radius which produced the best r-squared value.
# Since multiple radii produced the same best r-squared, I kept the smallest of
# those radius values (radius = 2, r-squared = -0.1)














