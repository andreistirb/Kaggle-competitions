# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:18:11 2017

@author: Andrei
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model,neighbors
#from sklearn import svm

# Initialize output files
logreg_file = open('output/logreg_output.csv', 'w')
knn_file = open('output/knn_output.csv', 'w')
svr_file = open('output/svr_output.csv', 'w')

logreg_file.write('ImageId,Label\n')
knn_file.write('ImageId,Label\n')
svr_file.write('ImageId,Label\n')

# Read data from csv files
train_dataset = pd.read_csv('input/train.csv')
test_dataset = pd.read_csv('input/test.csv')

# Convert the test dataset
test_pixels = np.array(test_dataset)

# Divide train dataset into train and validation sets
msk = np.random.rand(len(train_dataset)) < 0.8

train = train_dataset[msk]
validation = train_dataset[~msk]

# Isolate only the pixels of the image
train_pixels = np.array(train.iloc[:,1:])
validation_pixels = np.array(validation.iloc[:,1:])

# Isolate the labels
train_labels = np.array(train.iloc[:,0:1])
validation_labels = np.array(validation.iloc[:,0:1])

# Normalize the images
mean = train_pixels.mean(0)

#for i in range(0, train_pixels.shape[0]):
#    train_pixels[i,:] = train_pixels[i,:] - mean
                
# Trying first a multi-class logistic regression

logreg = linear_model.LogisticRegression(solver='sag',n_jobs=-1)

logreg.fit(train_pixels, np.ravel(train_labels))

a = logreg.predict(validation_pixels)

count = 0

for i in range(0, validation_labels.shape[0]):
    if a[i] != validation_labels[i]:
        count += 1
        
print("logistic regression error percent: ")
print(count/validation_labels.shape[0])
final = logreg.predict(test_pixels)

for i in range(0, test_pixels.shape[0]):
    q = str(i+1) + ',' + str(final[i]) + '\n'
    logreg_file.write(q)


# Trying k-nearest neighbors

knn = neighbors.KNeighborsClassifier(n_jobs=-1)
knn.fit(train_pixels, np.ravel(train_labels))

b = knn.predict(validation_pixels)

count = 0

for i in range(0, validation_labels.shape[0]):
    if b[i] != validation_labels[i]:
        count += 1
        
print("knn error percent: ")
print(count/validation_labels.shape[0])
knn_final = knn.predict(test_pixels)

for i in range(0, test_pixels.shape[0]):
    q = str(i+1) + ',' + str(knn_final[i]) + '\n'
    knn_file.write(q)
    
# Trying Support Vector Regression

#svr = svm.SVR(kernel='linear')
#svr.fit(train_pixels, np.ravel(train_labels))

#c = svr.predict(validation_pixels)

#count = 0
#for i in range(0, validation_labels.shape[0]):
#    if c[i] != validation_labels[i]:
#        count += 1

#print("svm error percent: ")
#print(count/validation_labels.shape[0])
#svm_final = svr.predict(test_pixels)

#for i in range(0, test_pixels.shape[0]):
#    q = str(i+1) + ',' + str(svm_final[i]) + '\n'
#    svr_file.write(q)

logreg_file.close()
knn_file.close()
#svr_file.close()

# Now trying a CNN