from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

# X, y = make_blobs(n_samples=100, centers=3)
X, y = make_moons(n_samples=100, shuffle=True, noise=None, random_state=42)
print(plt.scatter(X[:, 0], X[:, 1]))
kmeans = DBSCAN(eps=0.5, min_samples=5)
# kmeans = KMeans(n_clusters=3).fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print(labels)
print(centroids)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.plot(centroids[:, 0], centroids[:, 1], 'r')
