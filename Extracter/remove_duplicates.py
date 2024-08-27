import pandas as pd
import json

# Load the JSON config file
with open("../Config/config.json", "r") as f:
    CONFIG = json.load(f)

# Access configuration values with capitalized variable names
MODEL_NAME = CONFIG["BASIC"]["MODEL_NAME"]
DATA_FOLDER = CONFIG["BASIC"]["DATA_FOLDER"]

def remove_duplicates(file_path, output_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Remove duplicate rows based on the "FEN" column
    df = df.drop_duplicates(subset='FEN')
    
    # Save the cleaned data to a new CSV file
    df.to_csv(output_path, index=False)
    print(f"INFO | Duplicate rows removed and saved to {output_path}")

# Path to your original CSV file
input_file_path = fr"..\{DATA_FOLDER}\raw\{MODEL_NAME}.csv"

# Path to the new CSV file
output_file_path = fr"..\{DATA_FOLDER}\raw\{MODEL_NAME}.csv"

# Remove duplicates and save as a new file
remove_duplicates(input_file_path, output_file_path)
