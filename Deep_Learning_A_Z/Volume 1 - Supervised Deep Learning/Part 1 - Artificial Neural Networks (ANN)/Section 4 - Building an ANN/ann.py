# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])

labelencoder_X_gender = LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)    

# Implement ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# K fold validation
from keras.wrappers.scikit_learn import KerasClassifier

def build_classifier(optimizer_in):
    # Init ANN
    classifier = Sequential()
    # Add input layer and 1st hidden layer with dropout
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="glorot_uniform"))
    classifier.add(Dropout(rate=0.1))
    # 2nd hidden layer with dropout
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="glorot_uniform"))
    classifier.add(Dropout(rate=0.1))
    # Output layer
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="glorot_uniform"))
    # Complie ANN
    classifier.compile(optimizer=optimizer_in,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

# Tuning hyper-parameter
from sklearn.model_selection import GridSearchCV
classifier = KerasClassifier(build_fn = build_classifier)
hParameters = {'batch_size':[4,16,32,64,128],
                'epochs':[50,100,250,360,480,520],
                'optimizer_in':['adam','rmsprop']
                }
gridSearch = GridSearchCV(estimator = classifier,
                          param_grid = hParameters,
                          scoring='accuracy',
                          cv=10)

# Fit ANN to the training set with optimized hyper-parameters
gridSearch = gridSearch.fit(X_train, y_train)

# Get best parameters and accuracy
bestParams = gridSearch.best_params_
bestAccuracy = gridSearch.best_score_

###
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=10, epochs=50)

## Visual ANN
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(classifier,show_shapes=True, show_layer_names=True, rankdir='TB').create(prog='dot', format='svg'))
#
#from keras.utils import plot_model
#plot_model(classifier, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='LR')

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Transform probablity to true/false
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show() 

nX = sc.transform(np.asarray([0,0,600,1,40,3,60000,2,1,1,50000]).reshape(1, -1))
nX_pred = classifier.predict(nX)
