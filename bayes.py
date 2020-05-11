import pandas
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB


data = pandas.read_csv('Fruits.csv')
print(data)

y = data["Type"]
X = data[["Long", "Sucre", "Jaune"]]
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape)
print(X_test.shape)

classifier = BernoulliNB()
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))
print(classifier.score(X_train, y_train))
