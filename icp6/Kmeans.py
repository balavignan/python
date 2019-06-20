from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)

dataset = pd.read_csv('CC.csv')
data = dataset.select_dtypes(include=[np.number]).interpolate().fillna(dataset.select_dtypes(include=[np.number]).interpolate().mean(axis=0))

x = data.iloc[:,[3,13,14,15]]
y = data.iloc[:-1]

#print(dataset["TENURE"].value_counts())
#sns.FacetGrid(dataset, hue="TENURE", size=4).map(plt.scatter, "PURCHASES", "CREDIT_LIMIT")
# do same for petals
#sns.FacetGrid(dataset, hue="TENURE", size=4).map(plt.scatter, "PAYMENTS", "MINIMUM_PAYMENTS")
#plt.show()

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
from sklearn.decomposition import PCA
pca = PCA(2)
x_pca = pca.fit_transform(X_scaled_array)


X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)
from sklearn.cluster import KMeans

nclusters = 3 # this is the k in kmeans
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_scaled)
# predict the cluster for each data point
#y_cluster_kmeans = km.predict(X_scaled)
y_cluster_kmeans= km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score)

y_cluster_kmeans= km.predict(X_scaled)
from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print(score)

#from sklearn import metrics
wcss = []
##elbow method to know the number of clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()