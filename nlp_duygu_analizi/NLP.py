

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

df = pd.read_csv('NLPlabeledData.tsv',  delimiter="\t", quoting=3)

nltk.download('stopwords')



def process(review):
    # review without HTML tags
    review = BeautifulSoup(review).get_text()
    # review without punctuation and numbers
    review = re.sub("[^a-zA-Z]",' ',review)
    # converting into lowercase and splitting to eliminate stopwords
    review = review.lower()
    review = review.split()
    # review without stopwords
    swords = set(stopwords.words("english"))                      # conversion into set for fast searching
    review = [w for w in review if w not in swords]               
   
    return(" ".join(review))


train_x_tum = []
for r in range(len(df["review"])):        
    if (r+1)%1000 == 0:        
        print("No of reviews processed =", r+1)
    train_x_tum.append(process(df["review"][r]))



x = train_x_tum
y = np.array(df["sentiment"])

# train test split
train_x, test_x, y_train, y_test = train_test_split(x,y, test_size = 0.1)

vectorizer = CountVectorizer( max_features = 5000 )

train_x = vectorizer.fit_transform(train_x)

train_x = train_x.toarray()
train_y = y_train



model = RandomForestClassifier(n_estimators = 100, random_state=42)
model.fit(train_x, train_y)

test_xx = vectorizer.transform(test_x)

test_xx = test_xx.toarray()

test_predict = model.predict(test_xx)
dogruluk = roc_auc_score(y_test, test_predict)



print("Doğruluk oranı : % ", dogruluk * 100)

