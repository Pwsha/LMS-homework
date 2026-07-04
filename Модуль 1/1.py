import matplotlib.pyplot as plt

x = [1, 3, 7]
y = [2, 6, 14]
w0 = 0
w1 = 0
lr = 0.01
for i in range(len(x)):
  pred = w1 * x[i] + w0
  w1 += 2 * lr * x[i] * (y[i] - pred)
  w0 += 2 * lr * (y[i] - pred)
print(f'w0 = {w0}, w1 = {w1}')
plt.scatter(x, y, color='red')
plt.plot([0,7], [w0, w1 * 7 + w0])
plt.grid()
plt.show()

x = [1, 3, 7]
y = [2, 6, 14]
w0 = 0.379008
w1 = 1.945856
plt.scatter(x, y, color='red')
plt.plot([0,8], [w0, w1 * 7 + w0])
plt.grid()
plt.show()

