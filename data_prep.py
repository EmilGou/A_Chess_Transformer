#Adapted from https://github.com/sgrvinod/chess-transformers/blob/main/chess_transformers/data/prep.py

import tables as tb
from tqdm import tqdm
from .uci_moves import UCI_MOVES
import os

def prepare_data(
    data_folder,
    h5_file,
    val_split_fraction=None,
    overwrite=False,
):

    # Get names of files/chunks containing moves and FENs
    moves_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".moves")])
    fens_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".fens")])
    assert len(moves_files) == len(fens_files)
    print("\nMoves and FENs are stored in %d chunk(s).\n" % len(moves_files))

    # Create table description for H5 file
    class ChessTable(tb.IsDescription):
        fen = tb.StringCol(87)
        move = tb.StringCol(5)
        outcome = tb.Int8Col()


    # Create table description for HDF5 file
    class EncodedChessTable(tb.IsDescription):

        encoded_fen = tb.Int16Col(shape=(72))
        encoded_move = tb.Int16Col()
        outcome = tb.Int8Col()

    # Delete H5 file if it already exists; start anew
    if os.path.exists(os.path.join(data_folder, h5_file)):
        if overwrite:
            print("Deleting existing H5 file...")
            os.remove(os.path.join(data_folder, h5_file))
        else:
            raise ValueError("H5 file already exists. Set overwrite=True to overwrite.")

    # Create new H5 file
    h5_file = tb.open_file(
        os.path.join(data_folder, h5_file), mode="w", title="data file"
    )

    try:

        # Create table in H5 file
        table = h5_file.create_table("/", "data", ChessTable)

        # Create encoded table in H5 file
        encoded_table = h5_file.create_table(
            "/", "encoded_data", EncodedChessTable
        )

        # Create pointer to next row in these tables
        row = table.row
        encoded_row = encoded_table.row

        # Keep track of row numbers where new games begin
        new_game_index = 0
        new_game_indices = list()

        # Keep track of errors
        n_wrong_results = 0
        n_move_fen_mismatches = 0

        # Iterate through chunks
        for i in range(len(moves_files)):
            print("Now reading %s and %s...\n" % (moves_files[i], fens_files[i]))

            # Read moves and FENs in this chunk
            all_moves = open(os.path.join(data_folder, moves_files[i]), "r").read()
            all_fens = open(os.path.join(data_folder, fens_files[i]), "r").read()
            all_moves = all_moves.split("\n\n")[:-1]
            all_fens = all_fens.split("\n\n")[:-1]
            assert len(all_moves) == len(all_fens)
            print("There are %d games.\n" % len(all_moves))

            # Iterate through games in this chunk
            for j in tqdm(range(len(all_moves)), desc="Adding rows to table"):
                moves = all_moves[j].split("\n")
                result = moves.pop(-1)
                moves = [move.lower() for move in moves]
                moves.append("<end>")
                encoded_moves = [UCI_MOVES[move] for move in moves]
                fens = all_fens[j].split("\n")
                transformed_fens = [fen_transform(fen) for fen in fens]
                encoded_fens = [tokenizer.encode(t_fen) for t_fen in transformed_fens]
                transformed_fens = [fen.split() for fen in transformed_fens]#For uniformity in storing the data
                assert len(set(len(sublist) for sublist in encoded_fens)) == 1, "Non-unformity in FENS"
                results = []



                for j in range(len(moves)):
                  if result == "1-0":
                    if j % 2 == 0:
                      results.append(1)
                    else:
                      results.append(-1)
                  elif result == "0-1":
                    if j % 2 == 0:
                      results.append(-1)
                    else:
                      results.append(1)
                # Ignore game if there is a mismatch between moves and FENs
                if len(moves) != len(fens):
                    n_move_fen_mismatches += 1
                    continue  # ignore this game

                start_index = 0 if result == "1-0" else 1

                # Ignore this game if the wrong result is recorded in the source file
                if len(moves) % 2 != start_index:
                    n_wrong_results += 1
                    continue

                # Iterate through moves in this game
                for k in range(len(moves)):
                  row["fen"] = fens[k]
                  #row["transformed_fen"] = transformed_fens[k]
                  row["move"] = moves[k]
                  row["outcome"] = results[k]
                  row.append()

                  encoded_row["encoded_fen"] = encoded_fens[k]
                  encoded_row["encoded_move"] = encoded_moves[k]
                  encoded_row["outcome"] = results[k]
                  encoded_row.append()

                new_game_index += k + 1
                new_game_indices.append(new_game_index)




            table.flush()
            print("\nA total of %d datapoints have been saved to disk.\n" % table.nrows)

        print("...done.\n")

        if n_move_fen_mismatches > 0:
            print(
                "NOTE: %d game(s) excluded because number of moves and FENs did not match.\n"
                % n_move_fen_mismatches
            )
        if n_wrong_results > 0:
            print(
                "NOTE: %d game(s) (%.2f percent) excluded that had the wrong result recorded.\n"
                % (
                    n_wrong_results,
                    n_wrong_results
                    * 100.0
                    / (len(new_game_indices) + n_wrong_results + n_move_fen_mismatches),
                )
            )

        # Get indices to split at
        if val_split_fraction is not None:
            val_split_index = int(table.nrows - val_split_fraction * table.nrows)
            encoded_table.attrs.val_split_index = val_split_index
            print("Validation split index: %d\n" % val_split_index)


        print("...done.\n")

    finally:
        h5_file.close()


