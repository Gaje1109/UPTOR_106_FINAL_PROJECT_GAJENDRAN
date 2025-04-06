import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Developer = R.Gajendran

# Load dataset
file_path = "/MediPredict_AI_Health_Analyzer/CSVFiles/Symptom2Disease.csv"
symptoms_df = pd.read_csv(file_path)

print("Dataframe loaded and opened: \n", symptoms_df.head())

# Vectorize symptom descriptions
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(symptoms_df["symptoms"]).toarray()

# Train KMeans
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
symptoms_df["Cluster"] = kmeans.fit_predict(X)

# Optional PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=symptoms_df["Cluster"], cmap="viridis", alpha=0.7)
plt.colorbar(label="Cluster")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Symptoms Cluster Visualization")
plt.savefig("/MediPredict_AI_Health_Analyzer/Images/CDP_Scatter.png")
plt.show()

# Save the model and vectorizer
with open("/MediPredict_AI_Health_Analyzer/Model/kmeans_model.sav", "wb") as f:
    pickle.dump(kmeans, f)

with open("/MediPredict_AI_Health_Analyzer/Model/tfidf_vectorizer.sav", "wb") as f:
    pickle.dump(vectorizer, f)

# Save the DataFrame with clusters
with open("/MediPredict_AI_Health_Analyzer/Model/symptom_data_with_clusters.sav", "wb") as f:
    pickle.dump(symptoms_df, f)

print("\n All models and data saved successfully.")
