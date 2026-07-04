import numpy as np

x1 = np.random.randint(100, size = 100)
x2 = np.random.randint(200, size = 100)
y = 3 * x1 + 2 * x2 - 1
x1 = x1 + np.random.randint(5, size = 100) / 10
x2 = x2 + np.random.randint(5, size = 100) / 10
y = y + np.random.randint(5, size = 100) / 10
w2 = 0
w1 = 0
w0 = 0
lr = 0.00001
ep = 100
for epoh in range(1, ep + 1):
  for i in range(len(x1)):
    per = w1 * x1[i] + w2 * x2[i] + w0
    w2 += 2 * lr * x2[i] * (y[i] - per)
    w1 += 2 * lr * x1[i] * (y[i] - per)
    w0 += 2 * lr * (y[i] - per)
print(f'w1 = {w1}, w2 = {w2}, w0 = {w0}')

x1 = np.random.randint(100, size = 100)
x2 = np.random.randint(200, size = 100)
y = 3 * x1 + 2 * x2 - 1
x1 = x1 + np.random.randint(5, size = 100) / 10
x2 = x2 + np.random.randint(5, size = 100) / 10
y = y + np.random.randint(5, size = 100) / 10
w2 = 0
w1 = 0
w0 = 0
lr = 0.00001
ep = 100
for epoh in range(1, ep + 1):
  for i in range(len(x1)):
    per = w1 * x1[i] + w2 * x2[i] + w0
    w2 += 2 * lr * x2[i] * (y[i] - per)
    w1 += 2 * lr * x1[i] * (y[i] - per)
    w0 += 2 * lr * (y[i] - per)
print(f'w1 = {w1}, w2 = {w2}, w0 = {w0}')

x1 = x1 + np.random.randint(5, size = 100) / 10
x2 = x2 + np.random.randint(5, size = 100) / 10
y = y + np.random.randint(5, size = 100) / 10
w2 = 0
w1 = 0
w0 = 0
lr = 0.00001
ep = 150
for epoh in range(1, ep + 1):
  for i in range(len(x1)):
    pred = w1 * x1[i] + w2 * x2[i] + w0
    w1 += 2 * lr * x1[i] * (y[i] - pred)
    w2 += 2 * lr * x2[i] * (y[i] - pred)
    w0 += 2 * lr * (y[i] - pred)
print(f'w1 = {w1}, w2 = {w2}, w0 = {w0}')