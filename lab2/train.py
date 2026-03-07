import joblib
from utils import load_dataset, get_features
from kmeans_model import create_model

DATA_PATH = "data/dataset.csv"

data = load_dataset(DATA_PATH)

X = get_features(data)

model = create_model(n_clusters=3)

model.fit(X)

print("Cluster centers:")
print(model.cluster_centers_)

joblib.dump(model, "kmeans_model.pkl")

print("Model saved!")