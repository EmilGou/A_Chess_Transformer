    
import torch
import numpy as np
import random


def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# This is only for non-autoregressive
def get_vocab() -> dict:
    pieces = ["P","N","B","R","Q","K","p","n","b","r","q","k", "."]

    turn = ["t_w", "t_b"]
    castling = ["c_K", "c_Q", "c_k", "c_q", "nK", "nQ", "nk", "nq"]
    en_passant = [letter + number for letter in 'abcdefgh' for number in '12345678'] + ['-']


    full_move_numbers = [f'full_{i}' for i in range(1,101)]
    half_move_numbers = [f'half_{i}' for i in range(101)]

    vocab_list = pieces + turn + castling + en_passant + full_move_numbers + half_move_numbers + ["[UNK]"]

    vocab = {token: i for i, token in enumerate(vocab_list)}

    return vocab

# This is only for non-autoregressive
def fen_transform(fen):
    fen_split = fen.split(" ")

    piece_placement = fen_split[0]
    turn = fen_split[1]
    castling = fen_split[2]
    en_passant = fen_split[3]
    half_move_number = fen_split[4]
    full_move_number = fen_split[5]

    new_piece_placement = piece_placement.replace("/", "")
    for char in new_piece_placement:
        if char.isdigit():
            new_piece_placement = new_piece_placement.replace(char, '.' * int(char))
    assert len(new_piece_placement) == 64, "Invalid FEN given to fen_transform"
    new_piece_placement = ' '.join(new_piece_placement)
    new_turn = 't_w' if turn == 'w' else 't_b'

    new_castling = ["nK", "nQ", "nk", "nq"]

    if 'K' in castling:
      new_castling[0] = "c_K"
    if 'Q' in castling:
      new_castling[1] = "c_Q"
    if 'k' in castling:
      new_castling[2] = "c_k"
    if 'q' in castling:
      new_castling[3] = "c_q"

    new_castling = " ".join(new_castling)


    half_move_number= f'half_{half_move_number}'
    fullmove_number = f'full_{full_move_number}'

    transformed_fen = f'{new_turn} {new_piece_placement} {new_castling} {en_passant} {half_move_number} {fullmove_number}'

    return transformed_fen


def fix_fen(fen_str: str, default_side: str = "w") -> str:
    """
    Return a six-field FEN.  If the side-to-move field is missing or blank,
    insert `default_side` ("w" by default, set to "b" if you prefer Black).
    """
    # strip outer whitespace and collapse any double spaces
    parts = [p for p in fen_str.strip().split(' ') if p != '']

    # A legal FEN must have six tokens; five means the side-to-move is missing
    if len(parts) == 5:
        parts.insert(1, default_side)

    return " ".join(parts)


def parse_fen(fen_str: str):
  parts = fen_str.split(' ')
  position = parts[0]
  turn = parts[1]
  half_move_clock = parts[4]
  full_moves = parts[5]

  return {
      "position": position,
      "turn": turn,
      "half_move_clock": int(half_move_clock),
      "full_moves": int(full_moves)
  }



