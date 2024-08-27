import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

# Load the JSON config file
with open("../Config/config.json", "r") as f:
    CONFIG = json.load(f)

# Access configuration values with capitalized variable names
MODEL_NAME = CONFIG["BASIC"]["MODEL_NAME"]
DATA_FOLDER = CONFIG["BASIC"]["DATA_FOLDER"]


@tf.keras.utils.register_keras_serializable()
def clamp_activation(x):
    return tf.square(tf.clip_by_value(x, 0, 1))

# Load the models with the custom activation function
model_path = fr'..\{DATA_FOLDER}\models\{MODEL_NAME}.keras'

model = load_model(model_path, custom_objects={'clamp_activation': clamp_activation})

# Save the weights of the models
model.save_weights(f"{MODEL_NAME}.weights.h5")

print("Model saved")