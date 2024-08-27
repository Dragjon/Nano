import chess.pgn
import csv
import threading
import json

# Load the JSON config file
with open("../Config/config.json", "r") as f:
    CONFIG = json.load(f)

# Access configuration values with capitalized variable names
MODEL_NAME = CONFIG["BASIC"]["MODEL_NAME"]
DATA_FILE_NAME = CONFIG["BASIC"]["DATA_FILE_NAME"]
DATA_FOLDER = CONFIG["BASIC"]["DATA_FOLDER"]
DATA_EXTRACTER_THREADS = CONFIG["DATA_EXTRACTER"]["DATA_EXTRACTER_THREADS"]
NUM_GAMES_PER_THREAD = CONFIG["DATA_EXTRACTER"]["NUM_GAMES_PER_THREAD"]

# Function to convert game result to WDL (Win/Draw/Loss) in terms of white
def result_to_wdl(result):
    if result == "1-0":
        return 1.0  # White wins
    elif result == "0-1":
        return 0.0  # White loses
    else:
        return 0.5  # Draw

def read_games(f,n,mutex_read):
    games_array=[]
    mutex_read.acquire()
    for _ in range(n):
        curr_game=chess.pgn.read_game(f)
        if curr_game is None:
            break
        games_array.append(curr_game)
    mutex_read.release()
    return games_array
    
def process_game(thread_num, games_array,counter_fen):
    rows_array = []
    for game in games_array:
            
            board = game.board()
            result = game.headers["Result"]
            wdl = result_to_wdl(result)

            moves = list(game.mainline_moves()) 
            for _, move in enumerate(moves):
                if not board.is_check() and board.piece_at(move.to_square) is None:
                    fen_tmp = board.fen().split()
                    fen = fen_tmp[0] + " " + fen_tmp[1]
                    rows_array.append([fen, wdl])
                    counter_fen[thread_num] += 1
                board.push(move)
            if len(rows_array) == 0:
                print("ERROR, length of rows array is 0")
                    
    return rows_array
    

def process_cycle(thread_num, counter_fen, f, mutex_read, writer, mutex_write):
    
    while True:
        games_array=read_games(f,NUM_GAMES_PER_THREAD,mutex_read)
        if len(games_array)==0:
            return

        rows_array=process_game(thread_num, games_array,counter_fen)
        
        if len(rows_array)==0:
            return
        
        mutex_write.acquire()
        writer.writerows(rows_array)
        mutex_write.release()

        num_total_fens = 0
        for i in range(DATA_EXTRACTER_THREADS):
            num_total_fens += counter_fen[i]

        print(f"Thread: \033[91m{thread_num}\033[0m Fen: \033[93m{counter_fen[thread_num]}\033[0m Total: \33[92m{num_total_fens}\033[0m")


# Function to play through a game and save FENs and WDLs
def process_pgn(pgn_path, output_file):
    
    with open(pgn_path) as f:
        writer = csv.writer(output_file)
        writer.writerow(["FEN", "WDL"])
    
        counter_fen = [0 for _ in range(DATA_EXTRACTER_THREADS)]
        thread_array = [0 for _ in range(DATA_EXTRACTER_THREADS)]

        mutex_write=threading.Lock()
        mutex_read=threading.Lock()
        for i in range(DATA_EXTRACTER_THREADS):
            thread_array[i] = threading.Thread(target=process_cycle, args=(i, counter_fen, f, mutex_read, writer, mutex_write))
            thread_array[i].start()

        for i in range(DATA_EXTRACTER_THREADS):
            thread_array[i].join()
                        
    

# Main function
def main():
    pgn_path = fr"..\{DATA_FOLDER}\pgn\{DATA_FILE_NAME}.pgn"
    output_file_path = fr"..\{DATA_FOLDER}\raw\{MODEL_NAME}.csv"

    with open(output_file_path, "w", newline='') as output_file:
        process_pgn(pgn_path, output_file)

if __name__ == "__main__":
    main()
