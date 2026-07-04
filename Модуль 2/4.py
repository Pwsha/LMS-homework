import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import  LabelEncoder, OneHotEncoder

df = pd.read_csv('/content/Mall_Customers.csv')
df.head(5)
plt.scatter(df['Age'], df['Spending Score (1-100)'], edgecolors='k', alpha=0.6, s=100)

X = df.iloc[:, [3,4]].values
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)

dendr = sch.dendrogram(sch.linkage(X, method='ward'))
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X_scaler)

plt.figure(figsize=(12,7))
colors=['red', 'green', 'orange', 'blue', 'purple', 'skyblue', 'black']
for i in range(3):
  plt.scatter(X_scaler[y_hc == i, 0], X_scaler[y_hc==i, 1], c=colors[i], s=100, alpha=0.6)

wcss = []
for i in range(1, 16):
  kmeans = KMeans(n_clusters=i, init='k-means++')
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)
plt.plot(range(1,16), wcss)
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++')
labels = kmeans.fit_predict(X_scaler)

s1 = silhouette_score(X_scaler, y_hc)
s2 = silhouette_score(X_scaler, labels)

df = pd.read_csv('/content/healthcare_dataset.csv')
df.head(5)

def encode_features(df):
  le = LabelEncoder()
  for col in ['Gender', 'Test Results']:
    if col in df.columns:
      df[col] = le.fit_transform(df[col])
  return df

df = encode_features(df)
en = OneHotEncoder()
df['Blood Type'] = en.fit_transform(df['Blood Type'])