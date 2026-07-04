
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

df_train = pd.read_csv("/content/sample_data/california_housing_train.csv")
df_test = pd.read_csv("/content/sample_data/california_housing_test.csv")
df_train
X_train = df_train[['longitude', 'latitude', 'housing_median_age', 'total_rooms','total_bedrooms', 'population', 'households', 'median_income']]
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
y_train = df_train['median_house_value'] / 1000
X_test = df_test[['longitude', 'latitude', 'housing_median_age', 'total_rooms','total_bedrooms', 'population', 'households', 'median_income']]
X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
y_test = df_test['median_house_value'] / 1000

def make_fit(X_train, y_train, num_of_models):
  models = []
  y_train1 = y_train.copy()
  X_train1 = X_train.copy()
  for i in range(1, num_of_models + 1):
    X_train_shuffled, y_train_shuffled = shuffle(X_train1, y_train1)
    model = LinearRegression()
    model.fit(X_train_shuffled, y_train_shuffled)
    models.append(model)
    preds = model.predict(X_train_shuffled)
    y_train1 = (y_train_shuffled - preds).copy()
  return models

def predc(X_test, y_test, models):
  preds = np.zeros(len(y_test))
  for i in range(len(models)):
    preds += models[i].predict(X_test)
  return preds

MSEs = []
numbers = np.arange(1, 21)
for num in numbers:
  models = make_fit(X_train, y_train, num)
  preds = predc(X_test, y_test, models)
  MSE = ((y_test - preds) ** 2).mean()
  MSEs.append(MSE)
def mse_plot(numbers, MSEs):
  plt.plot(numbers, MSEs)
  plt.xlabel('Кол-во моделей')
  plt.ylabel('MSE')
  plt.title('Зависимость ошибки')
  plt.show()
mse_plot(numbers, MSEs)
def f(x):
    return np.tan(x)
def f_1(x):
    return np.arctan(x)
def make_and_fit(X_train, y_train, num_of_models):
    models = []
    y_train1 = y_train.copy()
    X_train1 = X_train.copy()
    for i in range(1, num_of_models + 1):
      X_train_shuffled, y_train_shuffled = shuffle(X_train1, y_train1)
      model = LinearRegression()
      model.fit(X_train_shuffled, y_train_shuffled)
      models.append(model)
      preds = model.predict(X_train_shuffled)
      y_train1 = (y_train_shuffled - preds).copy()
    return models
def predr(X_test, y_test, models):
  preds = np.zeros(len(y_test))
  for i in range(len(models)):
    preds += f(y - f_1(preds[i]))
  return preds
MSEs = []
numbers = np.arange(1, 21)
for num in numbers:
  models = make_fit(X_train, y_train, num)
  preds = predr(X_test, y_test, models)
  MSE = ((y_test - preds) ** 2).mean()
  MSEs.append(MSE)
def mse_plot(numbers, MSEs):
    plt.plot(numbers, MSEs)
    plt.xlabel('Кол-во моделей')
    plt.ylabel('MSE')
    plt.title('Зависимость ошибки')
    plt.show()
mse_plot(numbers, MSEs)

def make_and_fit(X_train, y_train, num_of_models):
    models = []
    y_train1 = y_train.copy()
    X_train1 = X_train.copy()
    for i in range(1, num_of_models + 1):
      X_train_shuffled, y_train_shuffled = shuffle(X_train1, y_train1)
      model = LinearRegression()
      model.fit(X_train_shuffled, y_train_shuffled)
      models.append(model)
      preds = model.predict(X_train_shuffled)
      y_train1 = (y_train_shuffled - preds).copy()
    return models
def predt(X_test, y_test, models):
  preds = np.zeros(len(y_test))
  for i in range(len(models)):
    preds += models[i].predict(X_test)
  return preds
MSEs = []
numbers = np.arange(1, 21)
for _ in range(1, 51):
  for num in numbers:
    models = make_fit(X_train, y_train, num)
    preds = predt(X_test, y_test, models)
    MSE = ((y_test - preds) ** 2).mean()
    a = np.mean(MSE)
    MSEs.append(a)
def mse_plot(numbers, MSEs):
  plt.plot(numbers, MSEs)
  plt.xlabel('Кол-во моделей')
  plt.ylabel('MSE')
  plt.title('Зависимость ошибки')
  plt.show()
mse_plot(numbers, MSEs)