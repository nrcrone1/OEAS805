# OEAS805 HOMEWORK 4: CLASSIFICATION MODEL
#import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

#%% PART 1 - MODEL CONSTRUCTION
wine = load_wine()

print(wine.data)
print(wine.target_names)
print(wine.target)
print(wine.feature_names)

# Feature matrix in a object named X
X = wine.data
# response vector in a object named y
y = wine.target

print(np.shape(X))
print(np.shape(y))


# Split wine data in train and test data (80:20)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

# scale the training data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#%% PART 2 - EXPLORE HYPERPARAMETER SPACE

#make a loop to test all combinations of N layers with N neurons each
max_layers = 10
max_neurons = 10

scores = np.zeros((max_layers, max_neurons)) #rows = layers, cols = neurons

results = [] #store confusion matrix, precision, recall, and f1-scores

for i in range(1, max_layers + 1):
    for j in range(1, max_neurons + 1):
        layeri = np.tile(j, i)
        print(layeri)
        mlp = MLPClassifier(hidden_layer_sizes=(layeri), max_iter=4000, random_state=2)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        accuracy =  metrics.accuracy_score(y_test,y_pred)
        scores[i-1, j-1] = accuracy
        
        #save results for high performing layer/neurons combinations
        if accuracy > 0.9: 
            results.append({
            'Layers': i,
            'Neurons': j,
            'Accuracy': accuracy,
            'Confusion Matrix': confusion_matrix(y_test, y_pred),
            'Classification Report': classification_report(
                y_test, y_pred, output_dict=True
            )
        })

#%% PLOT SCORES AS A HEATMAP
plt.figure(figsize=(9, 6))
sns.heatmap(
    scores,
    annot=True,
    cmap='viridis',
    vmin=0,
    vmax=1,
    xticklabels=range(1, max_neurons + 1),
    yticklabels=range(1, max_layers + 1)
)

plt.xlabel('Number of Neurons per Layer')
plt.ylabel('Number of Hidden Layers')
plt.title('MLP Classification Scores')

plt.show()

#I tested 100 combinations of 1-10 layers with 1-10 neurons each. Model 
#accuracy socres ranged from 0.36 to 1.0. When the model had one hidden
#layer, it performed well with any number of neurons. As the number of hidden
#layers increased, the number of neurons per layer needed to perform
#accurately (score > 0.9) increased. 

#Although many configurations of hidden layers and neurons performed well,
#I would use 1 hidden layer with 3 neurons because it performed the best while
#requiring the least number of layers and neurons. 


#%% TRAIN THE MODEL WITH THE CHOSEN NUMBER OF HIDDEN LAYERS AND NEURONS
mlp = MLPClassifier(hidden_layer_sizes=(3), max_iter=4000, random_state=2)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)
plt.plot(y_test, predictions, 'ro')
plt.plot([0, 2], [0, 2],'k')

#print confusion matrix and classification report for 1 layer with 3 neurons
print(confusion_matrix(y_test,predictions)) # Confusion Matrix
print(classification_report(y_test,predictions)) # Classification Report


#%% PART 3 - APPLY THE MODEL
from sklearn.utils import Bunch

filepath = 'unknown_wine.csv'

df = pd.read_csv(filepath)

unknown_wine = Bunch(
    data=df.iloc[:, :-1].values,
    target=df.iloc[:, -1].values,
    feature_names=df.columns[:-1].tolist(),
    target_names=df.iloc[:, -1].unique()
)

# Feature matrix in a object named X
X = unknown_wine.data
# response vector in a object named y
y = unknown_wine.target

print(np.shape(X))
print(np.shape(y))


X_test = scaler.transform(X)
y_test = y

predictions = mlp.predict(X_test)
plt.plot(y_test, predictions, 'ro')
plt.plot([0, 2], [0, 2],'k')

print(confusion_matrix(y_test,predictions)) # Confusion Matrix
print(classification_report(y_test,predictions)) # Classification Report

probabilities = mlp.predict_proba(X_test)

print(probabilities)

#convert probabilities to percentages with 2 decimal places
probabilities_percent = probabilities * 100
np.set_printoptions(suppress=True, precision=2)
print(probabilities_percent)

#The model classified the unknown winedataset with 100% accuracy. For any given
#wine sample, the probability for the correct classification was generally
#quite high (>70%) with the exception of a few outliner (as low as ~40%). 
#Overall, I think the model performed very well. 




















