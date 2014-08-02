
# coding: utf-8

# In[1]:

print("Hello World!")

import sklearn


# In[4]:

import numpy


# In[5]:

import scipy

                
# In[8]:

from sklearn import datasets


# In[11]:

digits = datasets.load_digits()


# In[12]:

print(digits.data)


# In[13]:

digits.target


# In[14]:

digits.images[0]


# In[36]:

from sklearn import svm


# In[37]:

clf = svm.SVC(gamma=0.001, C=100)


# In[38]:

clf.fit(digits.data[:-1],digits.target[:-1])


# In[39]:

clf.predict(digits.data[-1])


# In[40]:

images_and_labels = list(zip(digits.images, digits.target))


# In[41]:

from sklearn import metrics
import matplotlib.pyplot as plt 

for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2,4, index + 1)
    plt.axis('off')
    plt.title('Training: %i' % label)
    
# In[42]:

plt.show()

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




