# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 17:46:57 2023

@author: asus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Reading dataset
url =  'https://www.kaggle.com/arjunbhasin2013/ccdata/'
data = pd.read_csv(url, delimiter=";")

# Dropping unnecessary columns
data.drop(['CUST_ID'], axis=1, inplace=True)

# Handling missing values
data.fillna(method='ffill', inplace=True)

# Scaling the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Elbow plot to determine optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

# Adding the predicted clusters to the original dataset
data['Cluster'] = pred

# Visualizing the clusters
plt.figure(figsize=(12,6))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['Cluster'])
plt.title('Clusters of Credit Card Users')
plt.xlabel('Credit Limit')
plt.ylabel('Total Transactions')
plt.show()