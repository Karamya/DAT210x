import pandas as pd
import time
import os
# Grab the DLA HAR dataset from:
# http://groupware.les.inf.puc-rio.br/har
# http://groupware.les.inf.puc-rio.br/static/har/dataset-har-PUC-Rio-ugulino.zip


#
# TODO: Load up the dataset into dataframe 'X'
#
# .. your code here ..
os.chdir('D:/Data analysis/data/DAT210x/Module6/Datasets/')

X = pd.read_csv('dataset-har-PUC-Rio-ugulino.csv', sep = ';', decimal = ',')
#print(X.z4.unique())
print(X.head())
#
# TODO: Encode the gender column, 0 as male, 1 as female
#
# .. your code here ..
#X.gender = X.gender.map({'Man':1, 'Woman':0})
#X.gender = pd.get_dummies(X.gender)
X.drop(labels = ['gender'], axis =1, inplace = True)
#
# TODO: Clean up any column with commas in it
# so that they're properly represented as decimals instead
#
# .. your code here ..


#
# INFO: Check data types
print (X.dtypes)



#
# TODO: Convert any column that needs to be converted into numeric
# use errors='raise'. This will alert you if something ends up being
# problematic
#
# .. your code here ..
X.z4 = pd.to_numeric(X.z4, errors = 'coerce')
#print(X.z4.unique())

#
# INFO: If you find any problematic records, drop them before calling the
# to_numeric methods above...
print(X.isnull().sum())
X.dropna(axis = 0, how = 'any', inplace = 'True')

#
# TODO: Encode your 'y' value as a dummies version of your dataset's "class" column
#
# .. your code here ..
y = X[['class']]
y = pd.get_dummies(y)

#
# TODO: Get rid of the user and class columns
#
# .. your code here ..
X.drop(labels = ['user', 'class'], axis =1, inplace = True)

print (X.describe())


#
# INFO: An easy way to show which rows have nans in them
print( X[pd.isnull(X).any(axis=1)])


#
# TODO: Create an RForest classifier 'model' and set n_estimators=30,
# the max_depth to 10, and oob_score=True, and random_state=0
#
# .. your code here ..
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 30, max_depth = 10, oob_score = True, random_state = 0)


# 
# TODO: Split your data into test / train sets
# Your test size can be 30% with random_state 7
# Use variable names: X_train, X_test, y_train, y_test
#
# .. your code here ..
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7)




print ("Fitting...")
s = time.time()
#
# TODO: train your model on your training set
#
# .. your code here ..
model.fit(X_train, y_train)
print ("Fitting completed in: ", time.time() - s)


#
# INFO: Display the OOB Score of your data
score = model.oob_score_
print ("OOB Score: ", round((score*100), 3))




print ("Scoring...")
s = time.time()
#
# TODO: score your model on your test set
#
# .. your code here ..
score = model.score(X_test, y_test)
print ("Score: ", round((score*100), 3))
print ("Scoring completed in: ", time.time() - s)


#
# TODO: Answer the lab questions, then come back to experiment more

#Fitting completed in:  10.754999876022339
#OOB Score:  98.744
#Scoring...
#Score:  95.687
#Scoring completed in:  0.8149998188018799

#
# TODO: Try playing around with the gender column
# Encode it as Male:1, Female:0

#Fitting completed in:  10.881999969482422
#OOB Score:  98.771
#Scoring...
#Score:  95.939
#Scoring completed in:  0.7430000305175781

# Try encoding it to pandas dummies

#Fitting completed in:  10.748999834060669
#OOB Score:  98.771
#Scoring...
#Score:  95.939
#Scoring completed in:  0.7350001335144043

# Also try dropping it. See how it affects the score
# This will be a key on how features affect your overall scoring
# and why it's important to choose good ones.

#Fitting completed in:  10.971999883651733
#OOB Score:  98.887
#Scoring...
#Score:  96.158
#Scoring completed in:  0.7559998035430908

#
# TODO: After that, try messing with 'y'. Right now its encoded with
# dummies try other encoding methods to experiment with the effect.

