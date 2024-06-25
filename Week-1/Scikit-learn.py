# Loading and Splitting the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size = 0.3
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# There are lot of Machine learning models in Scikit-learn
import pandas as pd
import numpy as np
df = pd.read_csv("spam.csv")
df.head()
df['spam'] = df.Category.apply(lambda x: 1 if x == 'spam' else 0)
df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_cv = v.fit_transform(X_train.values)
X_train_cv
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_cv, y_train)
# Classification Report
from sklearn.metrics import classification_report
y_pred = model.predict(X_test_cv)
print(classification_report(y_test, y_pred))
