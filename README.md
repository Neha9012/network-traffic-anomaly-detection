# Network Traffic Anomaly Detection

## Description

This project demonstrates the application of Principal Component Analysis (PCA) for dimensionality reduction and Isolation Forest for anomaly detection in network traffic data. The primary goal is to identify anomalies in network traffic, which can be indicative of malicious activities or unexpected network behavior.

## Key Features

- **Data Generation:** Simulates network traffic data with 5 features, assuming 1000 samples for normal traffic.
- **Dimensionality Reduction:** Utilizes PCA to reduce the dimensionality of the network traffic data to 2 principal components for easier visualization and analysis.
- **Anomaly Detection:** Employs an Isolation Forest model to detect anomalies in the reduced data. The model is trained on the normal traffic data and then used to predict anomalies in test data.
- **Visualization:** Provides a scatter plot to visualize normal data and test data, with colors indicating the anomaly predictions.
- **Evaluation:** Includes a simple evaluation function to compute the accuracy of the anomaly detection, assuming all test data is normal for simplicity.

## Usage

1. **Clone the Repository:**
    ```sh
    git clone https://github.com/yourusername/network-traffic-anomaly-detection.git
    cd network-traffic-anomaly-detection
    ```

2. **Set Up the Environment:**
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. **Install Dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Script:**
    ```sh
    python network_traffic_anomaly_detection.py
    ```

This project is a helpful starting point for anyone interested in applying machine learning techniques for network security and anomaly detection.
