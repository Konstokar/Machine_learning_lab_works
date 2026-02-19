from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


def create_model(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),

        Dense(256, activation='relu'),
        Dense(128, activation='relu'),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
