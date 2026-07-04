from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

hard_problem = datasets.make_classification(
    n_samples=100,
    n_features=100,
    n_informative=50,
    n_classes=3,
    n_redundant=50,
    n_clusters_per_class=1,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    *hard_problem,
    test_size=0.3,
    random_state=1,
)

clf = KNeighborsClassifier(n_neighbors=8)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy_score(y_test, predictions)

k = np.arange(1, 30)
ac = []
for i in k:
  clf = KNeighborsClassifier(n_neighbors=i)
  clf.fit(X_train, y_train)
  predictions = clf.predict(X_test)
  a1 = accuracy_score(y_test, predictions)
  ac.append(a1)

plt.figure(figsize=(15,15))
plt.plot(k, ac)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.grid()
plt.show()

k_optim = 16
clf = KNeighborsClassifier(n_neighbors=k_optim)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Test accuracy: %.5f" % accuracy)
assert accuracy > 0.8, "попробуйте изменить следующие параметры: penalty, solver"

print('Хорошая работа!')

skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
print(classification_report(y_test, y_pred))