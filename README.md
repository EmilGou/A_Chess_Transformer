# A Chess Transformer

## General/Future TODOs

- Instead of having utilities in colab notebooks, create util.py files
- Organize Repo so that there is a distinction between the autoregressive and standard approach (or even consider deleting the standard approach)

## Dataset

Data is originally from the [Lichess Elite Database](https://database.nikonoel.fr/). I made use of this [script](https://github.com/sgrvinod/chess-transformers/blob/main/chess_transformers/data/LE22c.sh) from sgrvinod which uses pgn-extract to extract the FENs, moves, and outcomes of the games within the pgn files. Note that the script applies a filter to keep only games that ended in checkmate and that used a time control of at least 5 minutes. 

The dataset consists of 26,701,685 half-moves, the FEN description of the board position that the move was played, and the outcome of the game (1 for win by white and -1 for win by black)

[Download Here](https://drive.google.com/uc?export=download&id=1XuyQCim9l1ia8VG0MVSzYgxpdJa_rEtK)

You may also download the raw [Moves](https://drive.google.com/uc?export=download&id=1BSBuF2dKOnVWuR5CNjp-o7QBYb-10JTO) and [FENS](https://drive.google.com/uc?export=download&id=1MC9UTgqE5074gVSVzrDzIJT7Hg1MHQac). 

The dataset contains two tables: data and encoded_data. The data table contains the raw FEN sequences, moves, and outcome of each game. The encoded_data contains the encoded FENs after applying a transformation, encoded moves, and outcome of each game. 

- **`Lichess_Elite_Dataset.h5`** 
    - **data**
        - fen
        - move
        - outcome
    - **encoded_data**
        - encoded_fen
        - encoded_move
        - outcome
     
## Non-Autoregressive Model

TODO: Add evaluation sections

## Autoregressive Model

### Evaluations

#### Legal move Accuracy

Results are achieved from testing the model checkpoints on one randomly samples position from 1000 games. The samples are the same across all of the checkpoints. Top-k sampling is used with k = 10. 

![accuracies](./Assets/legalacc_seed=666_topk=10_v1.png)

The distribution of mistakes broken down by piece type across various checkpoints:

![mistakes](./Assets/mistakes_epochs_pieces_topk=10_v1.png)

#### Model vs Fairy-Stockfish

Below are the results of the estimated elo differences between the v1 model and various levels of fairy-stockfish. The estimated elos for each level of fairy-stockfish, according to a random user on the Lichess forum, are:

- Level 1: under 400
- Level 2: 500
- Level 3: 800
- Level 4: 1100


![EloDiff](./Assets/elo_vs_fairystockfish_v1_topk=10_alpha=30.png)

Detailed metrics across all levels

|   Level |   Games |   Wins |   Losses |   Draws | Elo $\Delta$         |   Win Ratios |   Likelihood of Superiorities |
|---------|---------|--------|----------|---------|-------------------|--------------|-------------------------------|
|       1 |     200 |    129 |        0 |      73 | 262.60 (+-39.24)  |    0.819307  |                             1 |
|       2 |     200 |     88 |        2 |     112 | 157.97 (+-30.20)  |    0.712871  |                             1 |
|       3 |     200 |      2 |      148 |      52 | -317.36 (+-47.84) |    0.138614  |                             0 |
|       4 |     200 |      1 |      180 |      21 | -487.68 (+-78.48) |    0.0569307 |                             0 |


