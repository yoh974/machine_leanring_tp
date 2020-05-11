from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X_train.shape)
print(X_test.shape)
classifier = DecisionTreeClassifier(random_state=0, max_depth=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classifier.score(X_test, y_test))
print(classifier.score(X_train, y_train))
classifier = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classifier.score(X_test, y_test))
print(classifier.score(X_train, y_train))
