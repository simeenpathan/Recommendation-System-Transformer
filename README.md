# Recommendation-System-Transformer

Movie Rating Prediction with Transformer Model
This project aims to predict movie ratings using a Transformer-based neural network model. The dataset consists of user IDs, movie IDs, and ratings. The model leverages multi-head attention and dense layers to achieve good prediction accuracy.

Project Structure
.
├── train_data.csv
├── test_data.csv
├── main.py
├── README.md
└── requirements.txt

train_data.csv: Training dataset file.
test_data.csv: Test dataset file.
main.py: Main script for training and evaluating the model.
README.md: Project documentation.
requirements.txt: Python dependencies required for the project.

Requirements
Make sure you have Python 3.7 or higher installed. Install the required packages using the following command:
pip install -r requirements.txt

Dataset
The dataset should be in CSV format with the following columns:

user_id: Integer representing the user ID.
movie_id: Integer representing the movie ID.
rating: Float representing the rating given by the user to the movie.
Ensure that the CSV files train_data.csv and test_data.csv are placed in the project directory.

Usage
Training the Model
To train the model, run the main.py script:
python main.py
The script will load the training data, train the model for 5 epochs, and evaluate it on the test data. The Mean Absolute Error (MAE) will be printed after evaluation.

Main Script (main.py)
Here's a brief overview of what the main.py script does:

Load the data: Reads the training and test datasets.
Create the model: Defines a Transformer-based neural network model.
Train the model: Fits the model on the training data for a specified number of epochs.
Evaluate the model: Evaluates the model on the test dataset and prints the MAE.

import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd

# Placeholder for creating model inputs
def create_model_inputs():
    return {
        "user_id": tf.keras.Input(name="user_id", shape=(), dtype=tf.int32),
        "movie_id": tf.keras.Input(name="movie_id", shape=(), dtype=tf.int32)
    }

# Placeholder for encoding input features
def encode_input_features(inputs):
    user_embedding = layers.Embedding(input_dim=10000, output_dim=64)(inputs["user_id"])
    user_embedding = layers.Reshape((1, -1))(user_embedding)

    movie_embedding = layers.Embedding(input_dim=10000, output_dim=64)(inputs["movie_id"])
    movie_embedding = layers.Reshape((1, -1))(movie_embedding)

    return user_embedding, movie_embedding

hidden_units = [256, 128]
dropout_rate = 0.1
num_heads = 3

# Model creation function
def create_model():
    inputs = create_model_inputs()
    transformer_features, other_features = encode_input_features(inputs)

    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=transformer_features.shape[-1], dropout=dropout_rate
    )(transformer_features, transformer_features)

    attention_output = layers.Dropout(dropout_rate)(attention_output)
    x1 = layers.Add()([transformer_features, attention_output])
    x1 = layers.LayerNormalization()(x1)
    x2 = layers.LeakyReLU()(x1)
    x2 = layers.Dense(units=x2.shape[-1])(x2)
    x2 = layers.Dropout(dropout_rate)(x2)
    transformer_features = layers.Add()([x1, x2])
    transformer_features = layers.LayerNormalization()(transformer_features)
    features = layers.Flatten()(transformer_features)

    features = layers.concatenate([features, layers.Flatten()(other_features)])

    for num_units in hidden_units:
        features = layers.Dense(num_units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.LeakyReLU()(features)
        features = layers.Dropout(dropout_rate)(features)

    outputs = layers.Dense(units=1)(features)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_dataset_from_csv(file_path, shuffle=False, batch_size=32):
    df = pd.read_csv(file_path)
    df['user_id'] = df['user_id'].astype(int)
    df['movie_id'] = df['movie_id'].astype(int)
    dataset = tf.data.Dataset.from_tensor_slices(({"user_id": df['user_id'], "movie_id": df['movie_id']}, df['rating']))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(batch_size)
    return dataset

# Load and preprocess the data
train_dataset = get_dataset_from_csv("train_data.csv", shuffle=True, batch_size=265)
test_dataset = get_dataset_from_csv("test_data.csv", batch_size=265)

# Define the model
model = create_model()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()],
)

# Train the model
model.fit(train_dataset, epochs=5)

# Evaluate the model
_, mae = model.evaluate(test_dataset, verbose=0)
print(f"Test MAE: {round(mae, 3)}")

Expected Output
The expected output after training and evaluation should be similar to:
Epoch 1/5
 1600/1600 [==============================] - 19s 11ms/step - loss: 1.5762 - mean_absolute_error: 0.9892
Epoch 2/5
 1600/1600 [==============================] - 17s 11ms/step - loss: 1.1263 - mean_absolute_error: 0.8502
Epoch 3/5
 1600/1600 [==============================] - 17s 11ms/step - loss: 1.0885 - mean_absolute_error: 0.8361
Epoch 4/5
 1600/1600 [==============================] - 17s 11ms/step - loss: 1.0943 - mean_absolute_error: 0.8388
Epoch 5/5
 1600/1600 [==============================] - 17s 10ms/step - loss: 1.0360 - mean_absolute_error: 0.8142
Test MAE: 0.782


License
This project is licensed under the MIT License - see the LICENSE file for details.


