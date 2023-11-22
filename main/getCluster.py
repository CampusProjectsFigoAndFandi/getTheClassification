import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score, accuracy_score
import numpy as np
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
import os

def getCluster(userInput):
    # Load and Prepare the Data
    csv_file_path = os.path.join('main', 'survey2.csv')
    data = pd.read_csv(csv_file_path)
    data = data[['K1.1', 'K1.2', 'F2.1', 'F2.2', 'K1.6', 'K1.8', 'F2.9', 'F2.10', 'P3.2']]

    # Standardize the Data
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    # Apply PCA
    pca = PCA(n_components=4)  # You can adjust the number of components based on your preference
    data_pca = pca.fit_transform(data_standardized)

    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(data_pca)
    data['Cluster'] = cluster_labels

    # Save the Cluster Labels to Use for Classification
    cluster_labels_user = data[['Cluster']].copy()

    # Prepare Data for Decision Tree Training
    X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
        data.drop(['Cluster'], axis=1),
        data['Cluster'],
        test_size=0.2,
        random_state=42
    )

    # Initialize and Train Decision Tree Model
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train_dt, y_train_dt)

    # Predict on the test set
    y_pred_dt = decision_tree.predict(X_test_dt)


    # Calculate accuracy
    accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)


    user_data = pd.DataFrame([userInput], columns=X_train_dt.columns)

    # Using pre-trained PCA object called 'pca'
    # Apply the same PCA transformation to the user data
    user_data_standardized = scaler.transform(user_data)
    user_data_pca = pca.transform(user_data_standardized)


    # Predict the cluster for the user input using the trained Decision Tree model
    user_cluster_decision_tree = decision_tree.predict(user_data)

    return user_cluster_decision_tree[0]