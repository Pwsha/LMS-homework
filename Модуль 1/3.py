import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(35)
X = 2 * np.random.rand(150, 1) ** 2
y =  2 * X * 3 + np.random.randn(150, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
x = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.scatter(X_train, y_train, color='blue', label='Тренировочные', alpha=0.2)
plt.scatter(X_test, y_test, color='green', label='Тестовые', alpha=0.2)
plt.plot(X_test, y_pred, color='red', label='Линейная регрессия')
plt.grid()
plt.xlabel('Признак')
plt.ylabel('Целевая переменная')
plt.legend()
plt.show()

plt.scatter(X_train, y_train, color='blue', label='Тренировочные', alpha=0.2)
plt.scatter(X_test, y_test, color='green', label='Тестовые', alpha=0.2)
# plt.plot()
plt.grid()
plt.xlabel('Признак')
plt.ylabel('Целевая переменная')
plt.legend()
plt.show()

mse = mean_squared_error(y_test, y_pred)
print(mse)

w1 = 2.95
w0 = 0.56
m = [1, 3, 4, 7, 8]
F = [3, 8 ,13, 19, 24]
plt.scatter(m, F, color='red')
plt.plot([0,8], [w0, w1 * 8 + w0])
plt.xlabel('Масса')
plt.ylabel('Сила')
plt.grid()
plt.show()

np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # один признак
y = 4 + 3 * X + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
x = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.scatter(X_train, y_train, color='blue', label='Тренировочные', alpha=0.2)
plt.scatter(X_test, y_test, color='green', label='Тестовые', alpha=0.2)
# plt.plot()
plt.grid()
plt.xlabel('Признак')
plt.ylabel('Целевая переменная')
plt.legend()
plt.show()