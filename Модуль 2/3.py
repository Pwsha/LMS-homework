from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import GridSearchCV

iris = load_iris()
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
param_grid = {'criterion':['gini', 'entropy'],
              'max_depth':[1, 2],
              'min_samples_split':[3, 5, 7, 9]}
tree_clf = tree.DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=tree_clf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

best = grid_search.best_params_best

pred_test = grid_search.predict(X_test)
pred_train = grid_search.predict(X_train)
a_test = accuracy_score(y_test, pred_test)
a_train = accuracy_score(y_train, pred_train)
print(a_test, a_train)