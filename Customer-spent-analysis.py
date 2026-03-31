# ============================================
# Customer Spend Analysis using Hierarchical Clustering
# ============================================

# ---------- Import Required Libraries ----------
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as clus
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering

# ---------- Load Dataset ----------
# Upload dataset (for Google Colab)
from google.colab import files
uploaded = files.upload()

# Read dataset
dataset = pd.read_csv('dataset.csv')

# ---------- Basic Data Exploration ----------
print("Shape of dataset:", dataset.shape)
print("\nStatistical Summary:\n", dataset.describe())
print("\nFirst 5 rows:\n", dataset.head())

# ---------- Data Preprocessing ----------
# Convert categorical column 'Gender' into numeric
label_encoder = LabelEncoder()
dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])

print("\nAfter Encoding:\n", dataset.head())

# ---------- Dendrogram Visualization ----------
# Helps to find optimal number of clusters

plt.figure(figsize=(16, 8))

# Create dendrogram using Ward method
dendrogram = clus.dendrogram(
    clus.linkage(dataset, method='ward')
)

# Titles and labels
plt.title('Dendrogram (Hierarchical Clustering)')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')

plt.show()

# ---------- Model Training ----------
# Apply Hierarchical Clustering (Agglomerative)

model = AgglomerativeClustering(
    n_clusters=5,          # Number of clusters
    metric='euclidean',    # Distance metric (updated from 'affinity')
    linkage='average'      # Linkage method
)

# Predict cluster labels
y_means = model.fit_predict(dataset)

print("\nCluster Labels:\n", y_means)

# ---------- Visualization of Clusters ----------
# Selecting Income & Spending columns (index 3 and 4)
X = dataset.iloc[:, [3, 4]].values

plt.figure(figsize=(10, 6))

# Plot each cluster with different colors
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=50, c='purple', label='Cluster 1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=50, c='orange', label='Cluster 2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s=50, c='red', label='Cluster 3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s=50, c='green', label='Cluster 4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s=50, c='blue', label='Cluster 5')

# Graph details
plt.title('Customer Segmentation (Hierarchical Clustering)')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()

plt.show()

# ---------- Cluster Interpretation ----------
"""
Cluster 1: Medium Income - Medium Spending
Cluster 2: High Income - High Spending
Cluster 3: Low Income - Low Spending
Cluster 4: High Income - Low Spending
Cluster 5: Low Income - High Spending
"""