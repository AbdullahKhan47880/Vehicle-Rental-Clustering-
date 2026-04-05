import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("vehicle_rental_company_dataset.csv")

# Select numeric features for clustering
features = [
    "AvgRentalDurationHours",
    "AvgTripDistanceKm",
    "RentalsPerMonth",
    "AvgRevenuePerRentalUSD",
    "WeekendRentalRatio",
    "LongTripRatio",
    "AvgIdleDaysPerMonth",
    "MonthlyMaintenanceCostUSD"
]

X = df[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to choose k
inertias = []
K = range(2, 7)
for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(K, inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Vehicle Clustering')
plt.tight_layout()
plt.savefig('elbow_method_output.png', dpi=200)
plt.show()

# Final model using k = 3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
df['ClusterID'] = kmeans.fit_predict(X_scaled)

# Give clusters business-friendly labels
cluster_means = df.groupby('ClusterID')[features].mean()
cluster_order = cluster_means['AvgRentalDurationHours'].sort_values().index.tolist()
label_map = {
    cluster_order[0]: 'Frequent Short-Trip Vehicles',
    cluster_order[1]: 'Balanced Mid-Range Vehicles',
    cluster_order[2]: 'Long-Term Premium Vehicles'
}
df['ClusterName'] = df['ClusterID'].map(label_map)

# Show cluster sizes
print("\nCluster Sizes:\n")
print(df['ClusterName'].value_counts())

# Show cluster characteristics
print("\nCluster Summary:\n")
print(df.groupby('ClusterName')[features].mean().round(2))

# PCA plot for visualization
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df['PC1'] = coords[:, 0]
df['PC2'] = coords[:, 1]

plt.figure(figsize=(8, 5))
for cluster_name, group in df.groupby('ClusterName'):
    plt.scatter(group['PC1'], group['PC2'], label=cluster_name, s=40, alpha=0.8)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clusters of Rental Vehicles')
plt.legend()
plt.tight_layout()
plt.savefig('cluster_scatter_output.png', dpi=200)
plt.show()

# Save final clustered dataset
df.to_csv('vehicle_rental_clustered_output.csv', index=False)
print("\nClustered dataset saved as vehicle_rental_clustered_output.csv")
