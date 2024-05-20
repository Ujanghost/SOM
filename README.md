# Self-Organizing Map (SOM) for Stock Data Clustering

This document provides a comprehensive explanation of the Python code used to implement a Self-Organizing Map (SOM) for clustering stock data. The code preprocesses the data, trains the SOM, visualizes the clusters, and evaluates the clustering quality using common metrics.

## Libraries Required

Ensure the following libraries are installed:

```sh
pip install numpy pandas minisom matplotlib scikit-learn
```

## Code Explanation

### 1. Import Libraries

```python
import numpy as np
import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
```

### 2. Load and Preprocess the Dataset

```python
# Load the dataset
data = pd.read_csv('stock_dataset.csv')

# Remove commas from numerical columns and convert to floats
for column in ['Open', 'High', 'Low', 'LTP', 'Volume (lacs)', 'Turnover (crs.)', '52w H', '52w L', '365 d % chng', '30 d % chng']:
    data[column] = data[column].replace({',': ''}, regex=True).astype(float)
```

- **Data Loading**: The CSV file is read into a pandas DataFrame.
- **Data Cleaning**: Commas are removed from numerical values to ensure they are properly converted to floats.

### 3. Feature Engineering and Normalization

```python
# Feature engineering
features = ['Open', 'High', 'Low', 'LTP', 'Volume (lacs)', 'Turnover (crs.)', '52w H', '52w L', '365 d % chng', '30 d % chng']
data_features = data[features]

# Normalize the data
scaler = MinMaxScaler()
data_features_normalized = scaler.fit_transform(data_features)
```

- **Feature Engineering**: Select relevant features from the dataset for clustering.
- **Normalization**: Normalize the selected features to a range between 0 and 1 using `MinMaxScaler`.

### 4. Initialize and Train the SOM

```python
# Initialize and train the SOM
som_shape = (10, 10)  # 10x10 grid
som = MiniSom(som_shape[0], som_shape[1], data_features_normalized.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(data_features_normalized)
som.train_random(data_features_normalized, 100)  # Train with 100 iterations
```

- **SOM Initialization**: Create a 10x10 SOM grid.
- **Training**: Train the SOM with 100 iterations using the normalized data.

### 5. Visualization

```python
# Plot the SOM
plt.figure(figsize=(10, 10))
for i, x in enumerate(data_features_normalized):
    w = som.winner(x)
    plt.text(w[0] + 0.5, w[1] + 0.5, data['Symbol'][i], fontsize=8, ha='center', va='center')

# Plotting the map with the number of hits (how many times each neuron is the winner)
plt.title('SOM - Stock Data Clustering')
plt.xlim([0, som_shape[0]])
plt.ylim([0, som_shape[1]])
plt.grid()
plt.show()

# Create a U-Matrix to show the distances between the neurons
u_matrix = som.distance_map().T

# Plot the U-Matrix
plt.figure(figsize=(10, 10))
plt.pcolor(u_matrix, cmap='bone_r')
plt.colorbar()
plt.title('U-Matrix - SOM')
plt.show()
```

- **SOM Plot**: Visualize the SOM grid with stock symbols placed at their corresponding cluster positions.
- **U-Matrix**: Plot the U-Matrix to visualize the distances between neurons, indicating the cluster boundaries.

### 6. Cluster Assignment

```python
# Predict cluster for each stock
clusters = np.array([som.winner(x) for x in data_features_normalized])
cluster_labels = np.apply_along_axis(lambda x: f"{x[0]},{x[1]}", 1, clusters)
data['Cluster'] = cluster_labels

# Display cluster assignment
print(data[['Symbol', 'Cluster']])
```

- **Cluster Prediction**: Each stock is assigned a cluster label based on the winning neuron.

### 7. Clustering Evaluation

```python
# Convert cluster labels to integers for silhouette and Davies-Bouldin score calculations
cluster_int_labels = np.array([x[0] * som_shape[1] + x[1] for x in clusters])

# Calculate Silhouette Score
sil_score = silhouette_score(data_features_normalized, cluster_int_labels)
print(f'Silhouette Score: {sil_score}')

# Calculate Davies-Bouldin Index
db_score = davies_bouldin_score(data_features_normalized, cluster_int_labels)
print(f'Davies-Bouldin Index: {db_score}')
```

- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters. A higher score indicates better-defined clusters.
- **Davies-Bouldin Index**: Calculates the average similarity ratio of each cluster with the one that is most similar to it. Lower values indicate better clustering.

## Conclusion

This code effectively implements a Self-Organizing Map (SOM) to cluster stock data. It includes data preprocessing, SOM training, visualization of the clustering results, and evaluation of clustering quality using Silhouette Score and Davies-Bouldin Index. These steps provide a comprehensive approach to understanding and leveraging SOM for stock data analysis.

---
