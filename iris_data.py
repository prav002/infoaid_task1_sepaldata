# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:01:57 2023

@author: Laptop
"""

import pandas as pd

# Load the dataset from CSV file
csv_path = "D:/DOWNLOADS/IRIS.csv"
iris_df = pd.read_csv(csv_path)
#Pre-process the dataset:
from sklearn.model_selection import train_test_split

X = iris_df.drop('species', axis=1)
y = iris_df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Explore the dataset:
import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(iris_df, hue='species')
plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors

#Train the model:
knn_model.fit(X_train, y_train)
#Evaluate the model:
accuracy = knn_model.score(X_test, y_test)
print("Accuracy:", accuracy)

new_data = [[5.1, 3.5, 1.4, 0.2],  # Replace with your own values
            [6.2, 3.4, 5.4, 2.3]]

predicted_species = knn_model.predict(new_data)
print("Predicted species:", predicted_species)
