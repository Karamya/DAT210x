# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:24:01 2016

@author: Karthick Perumal
"""

import pandas as pd
import os
import numpy as np

os.chdir('D:/Data analysis/data/DAT210x/Module6/Datasets/')

#Load up the /Module6/Datasets/parkinsons.data data set into a variable X, 
#being sure to drop the name column.

X = pd.read_csv('parkinsons.data')
X.drop('name', axis = 1, inplace = True)
#print(X.head())

#Splice out the status column into a variable y and delete it from X.
y = X.status
X.drop('status', axis = 1, inplace = True)
#print(y.head())
#print(X.head())

#Wait a second. Pull open the dataset's label file from: 
#https://archive.ics.uci.edu/ml/datasets/Parkinsons
#Look at the units on those columns: Hz, %, Abs, dB, etc. What happened
# to transforming your data? With all of those units interacting with one
# another, some pre-processing is surely in order.
#
#Right after you splice out the status column, but before you process 
#the train/test split, inject SciKit-Learn pre-processing code. Unless
# you have a good idea which one is going to work best, you're going to
# have to try the various pre-processors one at a time, checking to see 
# if they improve your predictive accuracy.
#
#Experiment with Normalizer(), MaxAbsScaler(), MinMaxScaler(), 
#and StandardScaler().
#
#After trying all of these scalers, what is the 
#new highest accuracy score you're able to achieve?

from sklearn import preprocessing
#X = preprocessing.Normalizer().fit_transform(X)
#X = preprocessing.MaxAbsScaler().fit_transform(X)
#X = preprocessing.MinMaxScaler().fit_transform(X)
X = preprocessing.StandardScaler().fit_transform(X)
#X = X #No change



#The accuracy score keeps creeping upwards. Let's have one more 
#go at it. Remember how in a previous lab we discovered that SVM's
#are a bit sensitive to outliers and that just throwing all of our
#unfiltered, dirty or noisy data at it, particularly in high-dimensionality
#space, can actually cause the accuracy score to suffer?
#
#Well, let's try to get rid of some useless features. Immediately after
#you do the pre-processing, run ISO on your dataset. The original dataset
#has 22 columns and 1 label column. So try experimenting with PCA n_component
#values between 4 and 14. Are you able to get a better accuracy?
#
#If you are not, then forget about PCA entirely, unless you want to
#visualize your data. However if you are able to get a higher score, 
#then be *sure* keep that figure in mind, and comment out all the PCA code.

#from sklearn.decomposition import PCA
#n_components = 14
#print('For ', n_components, 'number of PCA components')
#pca = PCA(n_components = n_components)
#X = pca.fit_transform(X)

## Getting the same accuracy as without PCA

#In the same spot, run Isomap on the data, before sending it to the 
#train / test split. Manually experiment with every inclusive combination
# of n_neighbors between 2 and 5, and n_components between 4 and 6.
# Are you able to get a better accuracy?
#
#If you are not, then forget about isomap entirely, unless you want
# to visualize your data. However if you are able to get a higher score,
# then be *sure* keep that figure in mind.
#
#If either PCA or Isomap helped you out, then uncomment out the appropriate
# transformation code so that you have the highest accuracy possible.
#
#What is your highest accuracy score on this assignment to date?

from sklearn.manifold import Isomap
n_neighbors =  5  # 2 to 5
n_components = 6   #4 to 6
iso = Isomap(n_neighbors = n_neighbors, n_components = n_components)
X = iso.fit_transform(X)
###########################################################
###              Results                                ###
###########################################################
##n_neighbors = 2,  n_components = 4, best_score = 0.966101694915
##                  n_components = 5, best_score = 0.966101694915
##                  n_components = 6, best_score = 0.966101694915
###########################################################
##n_neighbors = 3,  n_components = 4, best_score = 0.932203389831
##                  n_components = 5, best_score = 0.915254237288
##                  n_components = 6, best_score = 0.932203389831
###########################################################
##n_neighbors = 4,  n_components = 4, best_score = 0.949152542373
##                  n_components = 5, best_score = 0.966101694915
##                  n_components = 6, best_score = 0.949152542373
###########################################################
##n_neighbors = 5,  n_components = 4, best_score = 0.915254237288
##                  n_components = 5, best_score = 0.932203389831
##                  n_components = 6, best_score = 0.932203389831
###########################################################

#Perform a train/test split. 30% test group size, with a random_state equal to 7.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7)



#Create a SVC classifier. Don't specify any parameters, just leave everything as default. 
#Fit it against your training data and then score your testing data.

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print('Score is ', score)

#That accuracy was just too low to be useful. We need to get it up. 
#Once way you could go about doing that would be to manually try a bunch
# of combinations of C, and gamma values for your rbf kernel. But that 
# could literally take forever. Also, you might unknowingly skip a pair 
# of values that would have resulted in a very good accuracy.
#
#Instead, let us allow computers to do what computers do best. Program 
#a naive, best-parameter searcher by creating a nested for-loops. The 
#outer for-loop should iterate a variable C from 0.05 to 2, using 0.05 
#unit increments. The inner for-loop should increment a variable gamma 
#from 0.001 to 0.1, using 0.001 unit increments. As you know, Python 
#ranges won't allow for float intervals, so you'll have to do some
# research on NumPy ARanges, if you don't already know how to use them.
#
#Since the goal is to find the parameters that result in the model 
#having the best score, you'll need a best_score = 0 variable that you
# initialize outside of the for-loops. Inside the for-loop, create a
# model and pass in the C and gamma parameters into the class constructor.
# Train and score the model appropriately. If the current best_score is 
# better than the model's score, then update the best_score, being sure 
# to print it out, along with the C and gamma values that resulted in it.
#
#After running your assignment again, what is the highest accuracy score you are able to get?

best_score = 0
for c in np.arange(start = 0.05, stop = 2.001, step = 0.05):
    for gamma in np.arange(start = 0.001, stop = 0.1001, step= 0.001):
        model =SVC(C = c, gamma = gamma)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_C = c
            best_gamma = gamma
            
print('best_score is :', best_score, ' for C = ', best_C , ' and gamma = ', best_gamma)
        
    
