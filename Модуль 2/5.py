import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('/content/users_behavior.csv')
df.drop('Unnamed: 0', axis= 1 , inplace= True )
df.head(5)
df_st = (df - df.mean(axis=0) / df.std(axis=0))

X = df.drop('is_ultra', axis=1)
y = df['is_ultra']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model1 = KNeighborsClassifier(n_neighbors=15)
model1.fit(X_train, y_train)
pred_knn = model1.predict(X_test)

model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
pred_rand_forest = model2.predict(X_test)

model3 = LogisticRegression(max_iter=2500, multi_class='ovr')
model3.fit(X_train, y_train)
pred_log = model3.predict(X_test)

acc_knn = accuracy_score(y_test, pred_knn)
acc_rand_forest = accuracy_score(y_test, pred_rand_forest)
acc_log = accuracy_score(y_test, pred_log)
print(f'KNN accuracy = {acc_knn}')
print(f'RandomForest accuracy = {acc_rand_forest}')
print(f'Logistic accuracy = {acc_log}')

if (acc_knn > acc_log) and (acc_knn > acc_rand_forest):
  print('Метод KNN лучше')
elif (acc_rand_forest > acc_log) and (acc_rand_forest > acc_knn):
  print('Метод RandomForest лучше')
else:
  print('Метод LogisticRegression лучше')