# Nano
Chess Neural Network Trainer for Nectar with Tensorflow
# Data generation
* Generate your own data in the form of a `.pgn` file and place it in `/Data/pgn`
  * Do make sure your pgn is correctly formatted and all pgns within the file have a definite result
  * For datagen, probably should do a fixed nodes data-gen for best results
  * For starting, can do 10m positions first, then increasing
* No data is provided
# Config
* Modify `/Config/config.json` according to your needs
  * You probably want to change `HIDDEN_LAYERS` `EPOCHS` `MODEL_NAME` `DATA_FILE_NAME` `DATA_EXTRACTER_THREADS`
# Parsing data
1) Run `/Extracter/extract_csv.py
2) Run `/Extracter/remove_duplicates.py
# Training model
1) Run `/Trainer/encode_data.py`
2) Run `/Trainer/main.py`
# Getting binary weights of model
1) Run `/Converter/converter.py
2) Download a tool like [this](https://portal.hdfgroup.org/downloads/hdfview/hdfview3_3_1.html) to view the hdf5 file
3) Save all the weights and biases into binary files using the tool
# Using the weights and biases
## Encoding fen
```python
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
```
## Params for quantisation when reading and scaling
```cs
    static readonly int scale = 150;
    static readonly int quantise = 255;
```
## Neural network class
```cs
public class NeuralNetwork
{

    // Sigmoid activation function
    private static float Sigmoid(float x)
    {
        return 1 / (1 + (float)Math.Exp(-x));
    }

    // SCReLU activation function
    private static int SCReLU(int x)
    {
        int clipped = Math.Clamp(x, 0, quantise);
        return clipped * clipped;
    }

    public static int Predict(int[] inputs, int[,] inputWeights, int[] inputBiases, int[] outputWeights, int outputBias)
    {

        // Compute hidden layer activations
        int[] hiddenLayer = new int[hiddenLayerSize];
        for (int j = 0; j < hiddenLayerSize; j++)
        {
            int sum = 0;
            for (int i = 0; i < inputLayerSize; i++)
            {
                sum += inputs[i] * inputWeights[i, j];
            }
            hiddenLayer[j] = SCReLU(sum + inputBiases[j]);
        }

        // Compute output layer activation
        int output = 0;
        for (int j = 0; j < hiddenLayerSize; j++)
        {
            output += hiddenLayer[j] * outputWeights[j];
        }

        return (output / quantise + outputBias) * scale / (quantise * quantise);
    }

}

```
