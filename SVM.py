from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)
print(X_train.shape)
print(X_test.shape)
classifier = SVC(gamma='auto')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classifier.score(X_test, y_test))
print(classifier.score(X_train, y_train))
