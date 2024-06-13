import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Generate sample network traffic data (features can be packet size, frequency, etc.)
# For demonstration purposes, let's assume we have 1000 samples with 5 features
np.random.seed(42)
num_samples = 1000
num_features = 5
normal_traffic = np.random.normal(loc=0, scale=1, size=(num_samples, num_features))

# Perform dimensionality reduction using PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
normal_traffic_reduced = pca.fit_transform(normal_traffic)

# Create an Isolation Forest model for anomaly detection
clf = IsolationForest(contamination=0.05)  # Contamination parameter can be adjusted based on the expected rate of anomalies

# Fit the model to the reduced normal traffic data
clf.fit(normal_traffic_reduced)

# Generate some test data (can be real-time network traffic)
test_data = np.random.normal(loc=0, scale=1, size=(10, num_features))  # Generate 10 test samples
test_data_reduced = pca.transform(test_data)  # Reduce test data dimensionality

# Predict anomalies in the reduced test data
predictions = clf.predict(test_data_reduced)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(normal_traffic_reduced[:, 0], normal_traffic_reduced[:, 1], label='Normal Data', alpha=0.5)
plt.scatter(test_data_reduced[:, 0], test_data_reduced[:, 1], c=predictions, cmap='viridis', label='Test Data')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.colorbar(label='Anomaly Prediction')
plt.show()

# Evaluate the model
def evaluate_model(y_true, y_pred):
    # Compute accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f'Accuracy: {accuracy}')

evaluate_model(np.ones_like(predictions), predictions)  # Assuming all test data is normal for simplicity