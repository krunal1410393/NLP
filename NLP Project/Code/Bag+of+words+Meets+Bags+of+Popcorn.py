
# coding: utf-8

# # Project : Bag of words Meets Bags of Popcorn
# 

# ## Importing Packages and Data

# In[27]:

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import time
import seaborn as sns
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
get_ipython().magic('matplotlib inline')


# In[28]:

your_local_path="Mcintosh HD/Users/rk/Desktop/UPXTECH/PROJECT/NLP/Bag of world"


# In[29]:

cd /Users/rk/Desktop/UPXTECH/PROJECT/NLP/Bag of world


# # Train Data

# In[30]:

train = pd.read_csv("labeledTrainData.tsv", delimiter='\t')


# In[31]:

train.head()


# # Test data

# In[32]:

test = pd.read_csv("testData.tsv", delimiter='\t')


# In[33]:

test.head()


# In[34]:

train.shape


# ## Data Cleaning and Text Preprocessing 

# In[35]:

from bs4 import BeautifulSoup
import re     # to remove Punctuation and numbers


# In[36]:

from nltk.corpus import stopwords 
stopset = set(stopwords.words('english'))


# ## Import the stop word list

# In[13]:

from nltk.corpus import stopwords 
print 
stopwords.words("english")


# In[37]:

def review_to_words( raw_review ):
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))


# In[ ]:

num_reviews = train["review"].size
clean_train_reviews = []
for i in range( 0, num_reviews ):
   
    clean_train_reviews.append( review_to_words( train["review"][i] ) )


# In[ ]:

from sklearn.feature_extraction.text import CountVectorizer 

vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 5000) 
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()


# In[48]:

vocab = vectorizer.get_feature_names()
print(vocab)


# ## Training the Random forest

# In[83]:

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 500) 
forest = forest.fit( train_data_features, train["sentiment"] )


# In[51]:

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = []


# In[53]:

print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )


# ## Get a bag of words for the test set, and convert to a numpy array

# In[69]:


test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()


# In[84]:

result = forest.predict(test_data_features)
print (result)


# In[67]:

test.shape


# In[85]:

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
print (output)


# In[ ]:



