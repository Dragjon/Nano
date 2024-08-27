import numpy as np
import pandas as pd
import json

# Load the JSON config file
with open("../Config/config.json", "r") as f:
    CONFIG = json.load(f)

# Access configuration values with capitalized variable names
MODEL_NAME = CONFIG["BASIC"]["MODEL_NAME"]
DATA_FOLDER = CONFIG["BASIC"]["DATA_FOLDER"]
ENCODER_PRINT_FREQUENCY = CONFIG["DATA_ENCODER"]["DATA_ENCODER_PRINT_FREQUENCY"]
ENCODER_SET_ARRAY_SIZE = CONFIG["DATA_ENCODER"]["DATA_ENCODER_SET_ARRAY_SIZE"]

def encode_fen(fen):
    # Define the index mapping for each piece type in the 384-element array
    piece_to_index = {
        'P': 0,  'p': 384,  # Pawns
        'N': 64, 'n': 448, # Knights
        'B': 128, 'b': 512, # Bishops
        'R': 192, 'r': 576, # Rooks
        'Q': 256, 'q': 640, # Queens
        'K': 320, 'k': 704  # Kings
    }
    
    # Initialize the 384-element array
    board_array = np.zeros(768, dtype=np.int8)  
    
    # Split the FEN string to get the board layout
    board, _ = fen.split(' ', 1)
    
    # Split the board part into rows
    rows = board.split('/')
    
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                # Empty squares, advance the column index
                col_idx += int(char)
            else:
                # Piece, determine its position in the 384-element array
                piece_index = piece_to_index[char]
                board_position = row_idx * 8 + col_idx
                board_array[piece_index + board_position] = 1
                col_idx += 1
    
    return board_array

def process_and_split_data(file_path):
    # Read the CSV file
    print("INFO | Loading csv data")
    df = pd.read_csv(file_path)
    
    i = 0

    # Encode each FEN string to a 768-element array
    encoded_data = np.zeros((ENCODER_SET_ARRAY_SIZE, 768), dtype=np.int8)  # Use int8 for sbyte equivalent
    print(encoded_data.shape)
    for fen in df['FEN']:
        i += 1
        if i % ENCODER_PRINT_FREQUENCY == 0:
            print(f"INFO | Encoding fens {i})")
        encoded_data[i] = encode_fen(fen)
    
    encoded_data = encoded_data[:i]
    evaluations = []
    print("INFO | Getting white wdl")
    for i in df['WDL'].values:
        evaluations.append(i)

    evaluations = np.array(evaluations)
    
    print("INFO | Saving numpy files")
    
    # Save the datasets as numpy arrays
    np.save(f'../{DATA_FOLDER}/encoded/{MODEL_NAME}_x_train.npy', encoded_data)
    np.save(f'../{DATA_FOLDER}/encoded/{MODEL_NAME}_y_train.npy', evaluations)

# Process and split data for white and black pieces
process_and_split_data('../Data/raw/bee-net_768x32x1.csv')
