import joblib
import numpy as np

model = joblib.load("kmeans_model.pkl")

new_object = np.array([[30, 52000, 70, 16, 8]])

cluster = model.predict(new_object)

print("Cluster:", cluster[0])