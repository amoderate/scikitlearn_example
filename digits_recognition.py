
# coding: utf-8

# In[1]:

#Import library

import sklearn


# In[4]:

import numpy


# In[5]:

import scipy

                
# In[8]:

from sklearn import datasets


# Load data from example sets

digits = datasets.load_digits()


# View the data set

print(digits.data)


# 

digits.target


# In[14]:

digits.images[0]


# Import Support Vector Machines Classifier:

from sklearn import svm


''' Set some initial paramaters When training an SVM with the Radial Basis Function (RBF) kernel, 
    two parameters must be considered: C and gamma. The parameter C, common to all SVM kernels, 
    trades off misclassification of training examples against simplicity of the decision surface. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly. 
    gamma defines how much influence a single training example has. 
    The larger gamma is, the closer other examples must be to be affected. :
'''

clf = svm.SVC(gamma=0.001, C=100)

# set up some visialization (this only works on a local client:

images_and_labels = list(zip(digits.images, digits.target))


# In[41]:

from sklearn import metrics
import matplotlib.pyplot as plt 

#for index, (image, label) in enumerate(images_and_labels[:4]):
#   plt.subplot(2,4, index + 1)
#    plt.axis('off')
#    plt.title('Training: %i' % label)
    
# In[42]:

#plt.show()

# In[43]:

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# In[44]:

clf.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# In[45]:

expected = digits.target[n_samples / 2:]
predicted = clf.predict(data[n_samples / 2:])

# In[46]:

print("Classification report for classifire %s:\n%s\n"
      %(clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))




