import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import json

# Load the JSON config file
with open("../Config/config.json", "r") as f:
    config = json.load(f)

# Access configuration values
MODEL_NAME = config["BASIC"]["MODEL_NAME"]
DATA_FOLDER = config["BASIC"]["DATA_FOLDER"]
HIDDEN_LAYERS = config["MODEL"]["HIDDEN_LAYERS"]
LEARNING_RATE = config["MODEL"]["LEARNING_RATE"]
BATCH_SIZE = config["MODEL"]["BATCH_SIZE"]
EPOCHS = config["MODEL"]["EPOCHS"]
VERBOSE = config["MODEL"]["VERBOSE"]

def load_data():
    print("INFO | Loading Encoded x train data")
    x_train = np.load(f'../{DATA_FOLDER}/encoded/{MODEL_NAME}_x_train.npy')
    print("INFO | Loading Encoded y train data")
    y_train = np.load(f'../{DATA_FOLDER}/encoded/{MODEL_NAME}_y_train.npy')

    print("INFO | Shuffling x train data")
    np.random.seed(42) 
    np.random.shuffle(x_train)
    print("INFO | Shuffling y train data")
    np.random.seed(42) 
    np.random.shuffle(y_train)

    return x_train, y_train

def clamp_activation(x):
    return tf.square(tf.clip_by_value(x, 0, 1))

def build_model(input_shape):
    print("INFO | Defining model")
    model = Sequential([
        Dense(HIDDEN_LAYERS, input_shape=(input_shape,)),
        Lambda(clamp_activation),
        Dense(1, activation='sigmoid')
    ])
    print("INFO | Compiling model")
    model.compile(optimizer=AdamW(learning_rate=LEARNING_RATE), loss='mean_squared_error', metrics=['mae'])
    return model

def train_model():
    print("INFO | Start loading data")
    x_train, y_train = load_data()
    
    print("INFO | Building model")
    model = build_model(x_train.shape[1])
    
    print("INFO | Training model")
    history = model.fit(
        x_train, y_train, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        verbose=VERBOSE,
    )
    
    print("INFO | Saving final model")
    # Save the final model
    model.save(f'../{DATA_FOLDER}/models/{MODEL_NAME}.keras')
    
    return history

# Train models for white and black pieces
train_model()
