import json
from model import create_model
from utils import load_datasets

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "archive",
    "raw-img"
)

print("ABS DATASET PATH:", DATASET_PATH)
print("ABS EXISTS:", os.path.exists(DATASET_PATH))

EPOCHS = 10

train_ds, val_ds, class_names = load_datasets(DATASET_PATH)

print("Classes:", class_names)

model = create_model((64, 64, 3), len(class_names))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

model.save("ffnn_model.keras")


with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print("Model and classes saved!")
